# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import math
import os
import random
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

import numpy as np
import torch
from datasets import load_dataset
from jinja2 import Template
from PIL import Image
from PIL.Image import Image as ImageObject
from qwen_vl_utils.vision_process import fetch_video
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from . import torch_functional as VF
from .icd_descriptions import DescriptionLookup
from .note_truncate import chat_template_overhead, truncate_for_chat_prompt


def collate_fn(features: list[dict[str, Any]]) -> dict[str, Any]:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)
    for feature in features:
        for key, value in feature.items():
            if isinstance(value, torch.Tensor):
                tensors[key].append(value)
            else:
                non_tensors[key].append(value)

    for key, value in tensors.items():
        tensors[key] = torch.stack(value, dim=0)

    for key, value in non_tensors.items():
        non_tensors[key] = np.array(value, dtype=object)

    return {**tensors, **non_tensors}


def process_image(
    image: Union[dict[str, Any], ImageObject, str], min_pixels: Optional[int], max_pixels: Optional[int]
) -> ImageObject:
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


def process_video(
    video: str,
    min_pixels: Optional[int],
    max_pixels: Optional[int],
    video_fps: float,
    return_fps: bool = False,
    return_metadata: bool = False,
) -> Any:
    vision_info = {"video": video, "min_pixels": min_pixels, "max_pixels": max_pixels, "fps": video_fps}
    return fetch_video(vision_info, return_video_sample_fps=return_fps, return_video_metadata=return_metadata)


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        prompt_key: str = "prompt",
        answer_key: str = "answer",
        image_key: str = "images",
        video_key: str = "videos",
        image_dir: Optional[str] = None,
        video_fps: float = 2.0,
        max_prompt_length: int = 1024,
        truncation: str = "error",
        format_prompt: Optional[str] = None,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        filter_overlong_prompts: bool = True,
        filter_overlong_prompts_workers: int = 16,
        hint_pool_key: Optional[str] = None,
        max_hint_codes: int = 0,
        min_hint_codes: int = 0,
        p_inject_hint: float = 1.0,
        skip_empty_hint_pool: bool = False,
        code_weights_path: Optional[str] = None,
        hint_temperature: float = 1.0,
        hint_use_descriptions: bool = True,
        # Empty/None → resolved by DescriptionLookup.from_default to the
        # repo-relative path (./data/icd9_descriptions.json) or
        # ICD9_DESC_PATH env var.
        hint_descriptions_path: Optional[str] = None,
        hint_description_max_chars: int = 100,
        hint_include_mini_example: bool = True,
        hint_icd_version: str = "icd9",
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.image_key = image_key
        self.video_key = video_key
        self.image_dir = image_dir
        self.video_fps = video_fps
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.hint_pool_key = hint_pool_key
        self.max_hint_codes = max_hint_codes
        self.min_hint_codes = min_hint_codes
        self.p_inject_hint = p_inject_hint
        self.skip_empty_hint_pool = skip_empty_hint_pool
        self.hint_temperature = hint_temperature
        self.hint_use_descriptions = hint_use_descriptions
        self.hint_descriptions_path = hint_descriptions_path
        self.hint_description_max_chars = hint_description_max_chars
        self.hint_include_mini_example = hint_include_mini_example
        # "icd9" or "icd10" → label + mini-example shown in description hints.
        self.hint_icd_version = "icd10" if str(hint_icd_version).lower() == "icd10" else "icd9"
        self._hint_desc_lookup: Optional[DescriptionLookup] = None
        if self.hint_pool_key and self.hint_use_descriptions:
            try:
                # Empty/None hint_descriptions_path → from_default falls back
                # to the repo-relative DEFAULT_PATH (or ICD9_DESC_PATH env).
                self._hint_desc_lookup = DescriptionLookup.from_default(
                    self.hint_descriptions_path or None
                )
                logger.info(
                    "Loaded ICD descriptions for hint rendering from %s (max_chars=%d, mini_example=%s)",
                    self.hint_descriptions_path,
                    self.hint_description_max_chars,
                    self.hint_include_mini_example,
                )
            except FileNotFoundError as exc:
                logger.warning(
                    "hint_use_descriptions=true but descriptions file is unavailable: %s. "
                    "Falling back to bare-code hints.",
                    exc,
                )
        # Load code-level sampling weights if provided. {} when no file given,
        # which causes _inject_hint to fall back to uniform random.sample.
        self.code_weights: dict[str, float] = {}
        self._median_weight: float = 1.0
        if code_weights_path:
            import json as _json
            with open(code_weights_path) as _f:
                self.code_weights = _json.load(_f)
            if self.code_weights:
                import statistics as _stats
                self._median_weight = _stats.median(self.code_weights.values())
            logger.info(
                "Loaded code_weights for %d codes from %s (T=%.2f, median=%.4f)",
                len(self.code_weights), code_weights_path,
                hint_temperature, self._median_weight,
            )
        # Defensive: catch misconfigurations early. max<=0 means "no cap"
        # (use len(pool)), so only flag min>max when a positive cap exists.
        if max_hint_codes > 0 and min_hint_codes > max_hint_codes:
            raise ValueError(
                f"min_hint_codes={min_hint_codes} > max_hint_codes="
                f"{max_hint_codes}; these clamp silently in _inject_hint but "
                f"this is almost certainly a config mistake."
            )
        if min_hint_codes < 0:
            raise ValueError(f"min_hint_codes must be >= 0, got {min_hint_codes}")
        if not 0.0 <= p_inject_hint <= 1.0:
            raise ValueError(f"p_inject_hint must be in [0, 1], got {p_inject_hint}")

        if "@" in data_path:
            data_path, data_split = data_path.split("@")
        else:
            data_split = "train"

        if os.path.isdir(data_path):
            # when we use dataset builder, we should always refer to the train split
            file_type = os.path.splitext(os.listdir(data_path)[0])[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_dir=data_path, split=data_split)
        elif os.path.isfile(data_path):
            file_type = os.path.splitext(data_path)[-1][1:].replace("jsonl", "json")
            self.dataset = load_dataset(file_type, data_files=data_path, split=data_split)
        else:
            # load remote dataset from huggingface hub
            self.dataset = load_dataset(data_path, split=data_split)

        self.format_prompt = None
        if format_prompt:
            with open(format_prompt, encoding="utf-8") as f:
                self.format_prompt = f.read()

        # Sanity-check: if user asks for hint injection, make sure the parquet
        # actually carries a hint_pool column. Otherwise training would silently
        # run without hints.
        if self.hint_pool_key and len(self.dataset) > 0:
            if self.hint_pool_key not in self.dataset.column_names:
                raise ValueError(
                    f"hint_pool_key={self.hint_pool_key!r} but column not "
                    f"found in dataset; columns={self.dataset.column_names}. "
                    f"Either regenerate parquet with the hint_pool column or "
                    f"unset data.hint_pool_key."
                )

        # Phase 2: drop "mastered" samples (empty hint_pool). Inference still
        # sees the full parquet, so a previously-mastered sample re-enters
        # training when the next round's inference puts something back into
        # its hint_pool.
        if self.skip_empty_hint_pool:
            if not self.hint_pool_key:
                raise ValueError(
                    "skip_empty_hint_pool=true requires hint_pool_key to be set."
                )
            n_before = len(self.dataset)
            key = self.hint_pool_key
            self.dataset = self.dataset.filter(
                lambda ex: ex.get(key) is not None and len(ex[key]) > 0,
                desc="Skip mastered samples (empty hint_pool)",
                num_proc=filter_overlong_prompts_workers,
                load_from_cache_file=False,
            )
            n_after = len(self.dataset)
            # Print + log: this is a key Phase 2 observability signal — how
            # many samples does the model consider "mastered" this round?
            msg = (
                f"skip_empty_hint_pool: kept {n_after}/{n_before} samples "
                f"(dropped {n_before - n_after} mastered)"
            )
            print(msg, flush=True)
            logger.warning(msg)

        # Measure chat-template overhead once here (before .filter ships
        # `self` to worker subprocesses via pickle). Each worker inherits the
        # cached int — no redundant per-worker measurement, no race.
        self._template_overhead = chat_template_overhead(
            self.tokenizer, enable_thinking=False
        )
        logger.info(
            "Chat-template overhead: %d tokens; content budget: %d tokens",
            self._template_overhead, max_prompt_length - self._template_overhead,
        )

        if filter_overlong_prompts:
            # Disable HF datasets' filter cache: the default cache key hashes
            # only the function source and dataset fingerprint, so changing
            # max_hint_codes / hint_pool_key between runs would silently reuse
            # a stale filter result and let prompts slip through that no
            # longer fit with the (now different) worst-case hint length.
            # 40s re-filter per run is cheap insurance.
            n_before = len(self.dataset)
            self.dataset = self.dataset.filter(
                self._filter_overlong_prompts,
                desc=(
                    f"Filtering overlong prompts "
                    f"[mhc={self.max_hint_codes} key={self.hint_pool_key}]"
                ),
                num_proc=filter_overlong_prompts_workers,
                load_from_cache_file=False,
            )
            n_after = len(self.dataset)
            if n_before != n_after:
                msg = (
                    f"filter_overlong_prompts: dropped {n_before - n_after}/"
                    f"{n_before} samples where truncate_for_chat_prompt could "
                    f"not fit within max_prompt_length={max_prompt_length}. "
                    f"These samples have boilerplate exceeding the budget or "
                    f"extreme tokenizer drift."
                )
                print(msg, flush=True)
                logger.warning(msg)

    def _weighted_sample(self, pool: list[str], k: int) -> list[str]:
        """Weighted sample without replacement from `pool` of size `k` using
        `self.code_weights` and `self.hint_temperature`.

        p_i ∝ weight^(1/T). Numerically stable via log-space normalization
        (subtract max log-weight before exponentiating). On any numpy error
        (zero/NaN/Inf weights, fewer non-zero entries than k), falls back to
        uniform random.sample so training never crashes mid-epoch."""
        T = max(self.hint_temperature, 1e-3)
        inv_T = 1.0 / T
        # log(weight^(1/T)) = (1/T) * log(weight)
        log_w = []
        for c in pool:
            w = self.code_weights.get(c, self._median_weight)
            # Floor at 1e-12 to avoid log(0) → -inf
            log_w.append(inv_T * math.log(max(w, 1e-12)))
        m = max(log_w)
        scored = [math.exp(lw - m) for lw in log_w]
        total = sum(scored)
        if total <= 0 or not math.isfinite(total):
            return random.sample(pool, k)
        probs = [s / total for s in scored]
        try:
            return list(np.random.choice(pool, size=k, replace=False, p=probs))
        except ValueError:
            # Not enough non-zero entries (k > #non-zero) — fall back.
            return random.sample(pool, k)

    # Anchor text produced by tasks/prepare_icd_grpo.py; we insert hints
    # immediately before it so the final format instruction stays last.
    HINT_ANCHOR = "Output format:"
    # Mode-agnostic hint phrasing that frames the revealed codes as a
    # teaching signal rather than a copy-target. Works for both think and
    # no-think prompts.
    HINT_TEMPLATE = (
        "Hint: codes {codes} apply to this case. Verify each against the note "
        "and list all applicable codes (these plus any others)."
    )
    # Version-specific mini-examples. ICD-9 set (acidosis / hypertension /
    # kidney failure) and the parallel ICD-10 set, so the example always
    # matches the coding system the model is actually being trained on.
    HINT_DESC_EXAMPLE_ICD9 = (
        "Mini example:\n"
        "If the hint says:\n"
        "- 276.2 (Acidosis)\n\n"
        "and the note also supports 401.9 and 584.9, then the final answer "
        "should include both the hinted code and the other supported codes:\n"
        "<code>276.2, 401.9, 584.9</code>"
    )
    HINT_DESC_EXAMPLE_ICD10 = (
        "Mini example:\n"
        "If the hint says:\n"
        "- E87.2 (Acidosis)\n\n"
        "and the note also supports I10 and N17.9, then the final answer "
        "should include both the hinted code and the other supported codes:\n"
        "<code>E87.2, I10, N17.9</code>"
    )
    _hint_anchor_warned = False

    def _format_hint_code(self, code: str) -> str:
        if not self._hint_desc_lookup:
            return code
        desc = self._hint_desc_lookup.get(code, max_chars=self.hint_description_max_chars)
        if not desc:
            return code
        return f"{code} ({desc})"

    def _build_hint_line(self, hint_codes: list[str]) -> str:
        """Render the selected hint codes.

        With descriptions enabled this is the "E" prompt variant validated by
        tasks/probe_hint_description_effect_on_samples.py: confirmed missed
        codes + descriptions + a tiny example showing hints remain in the
        final full-code answer.
        """
        if self._hint_desc_lookup:
            icd_label = "ICD-10" if self.hint_icd_version == "icd10" else "ICD-9"
            desc_lines = "\n".join(f"- {self._format_hint_code(code)}" for code in hint_codes)
            hint_line = (
                "Training hint:\n"
                f"The following {icd_label} codes were missed previously, but they are confirmed applicable to this case.\n"
                f"Include these hinted codes in the final <code> answer, then add any other applicable {icd_label} codes supported by the note.\n\n"
                "Hinted codes:\n"
                f"{desc_lines}"
            )
            if self.hint_include_mini_example:
                example = (
                    self.HINT_DESC_EXAMPLE_ICD10
                    if self.hint_icd_version == "icd10"
                    else self.HINT_DESC_EXAMPLE_ICD9
                )
                hint_line = f"{hint_line}\n\n{example}"
            return hint_line

        return self.HINT_TEMPLATE.format(codes=", ".join(hint_codes))

    def _inject_hint(self, prompt_str: str, hint_pool, worst_case: bool = False) -> str:
        """Insert a hint line before HINT_ANCHOR when hint_pool is non-empty.

        Hint count distribution:
          k_max = len(pool)                        if max_hint_codes <= 0 (no cap)
                  min(max_hint_codes, len(pool))   otherwise
          k_min = clamp(min_hint_codes, 0, k_max)
          k ~ Uniform(k_min, k_max)

        Phase 1 config (full GT as pool): min=0, max=5 → k ~ Uniform(0, 5)
        Phase 2 config (missed as pool):  min=1, max=0 → k ~ Uniform(1, len(missed))

        worst_case=True (used by the overlong-prompt filter) forces k = k_max
        so the filter is conservative regardless of random state.
        """
        if hint_pool is None or len(hint_pool) == 0:
            return prompt_str
        pool = [str(c) for c in hint_pool if c]
        if not pool:
            return prompt_str

        k_max = len(pool) if self.max_hint_codes <= 0 else min(self.max_hint_codes, len(pool))
        k_min = max(0, min(self.min_hint_codes, k_max))

        if worst_case:
            k = k_max
        else:
            k = random.randint(k_min, k_max)
        if k == 0:
            return prompt_str

        if worst_case:
            # Worst-case (filter pass): deterministic pick, longest tokens
            # don't matter since we're just measuring prompt length.
            hint_codes = pool[:k]
        elif self.code_weights:
            hint_codes = self._weighted_sample(pool, k)
        else:
            hint_codes = random.sample(pool, k)
        hint_line = self._build_hint_line(hint_codes)
        idx = prompt_str.rfind(self.HINT_ANCHOR)
        if idx == -1:
            if not RLHFDataset._hint_anchor_warned:
                logger.warning(
                    "Hint anchor %r not found in prompt; appending hint at "
                    "the end. Final format instruction may no longer be last. "
                    "Check that prepare_icd_grpo.py produced prompts with the "
                    "expected anchor text.",
                    self.HINT_ANCHOR,
                )
                RLHFDataset._hint_anchor_warned = True
            return f"{prompt_str.rstrip()}\n\n{hint_line}"
        before = prompt_str[:idx].rstrip()
        after = prompt_str[idx:]
        return f"{before}\n\n{hint_line}\n\n{after}"

    def _build_messages(self, example: dict[str, Any], worst_case_hint: bool = False) -> list[dict[str, Any]]:
        prompt_str: str = example[self.prompt_key]
        if self.hint_pool_key and self.hint_pool_key in example:
            # worst_case_hint=True is used by the prompt-length filter, so it
            # always injects to stay conservative regardless of runtime mix.
            should_inject = worst_case_hint or random.random() < self.p_inject_hint
            if should_inject:
                prompt_str = self._inject_hint(
                    prompt_str, example[self.hint_pool_key], worst_case=worst_case_hint
                )
        if self.format_prompt:
            format_prompt = Template(self.format_prompt.strip())
            prompt_str = format_prompt.render(content=prompt_str)

        # Text-only prompts: shrink the clinical-note section if the full
        # prompt exceeds the budget. truncate_for_chat_prompt is a no-op on
        # prompts that already fit (quick length check). Image/video prompts
        # follow a different anchor path and skip truncation.
        if (self.image_key not in example and self.video_key not in example):
            prompt_str = truncate_for_chat_prompt(
                prompt_str,
                self.tokenizer,
                max_prompt_length=self.max_prompt_length,
                template_overhead=self._template_overhead,
                enable_thinking=False,
            )

        if self.image_key in example:
            # https://huggingface.co/docs/transformers/en/tasks/image_text_to_text
            content_list = []
            for i, content in enumerate(prompt_str.split("<image>")):
                if i != 0:
                    content_list.append({"type": "image"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        elif self.video_key in example:
            content_list = []
            for i, content in enumerate(prompt_str.split("<video>")):
                if i != 0:
                    content_list.append({"type": "video"})

                if content:
                    content_list.append({"type": "text", "text": content})

            return [{"role": "user", "content": content_list}]
        else:
            return [{"role": "user", "content": prompt_str}]

    def _filter_overlong_prompts(self, example: dict[str, Any]) -> bool:
        # Use worst-case hint length so the filter is conservative and
        # deterministic (runtime __getitem__ may see shorter or equal prompts).
        # With runtime truncation enabled (truncate_note_smart in
        # _build_messages), this check mainly catches the rare case where
        # boilerplate alone exceeds the budget — truncate_note_smart raises
        # ValueError in that case, which we treat as "drop".
        try:
            messages = self._build_messages(example, worst_case_hint=True)
        except ValueError:
            return False
        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example[self.image_key]
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example[self.video_key]
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            for video in videos:
                processed_videos.append(process_video(video, self.min_pixels, self.max_pixels, self.video_fps))

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            return model_inputs["input_ids"].size(-1) <= self.max_prompt_length
        else:
            input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
            return len(input_ids) <= self.max_prompt_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example: dict = self.dataset[index]
        messages = self._build_messages(example)
        example.pop(self.prompt_key, None)
        # Drop hint_pool so it doesn't get collated into the batch and
        # forwarded to reward / ckpt state (the hint has already been baked
        # into `messages`).
        if self.hint_pool_key:
            example.pop(self.hint_pool_key, None)

        if self.image_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = example.pop(self.image_key)
            if self.image_dir is not None and len(images) != 0 and isinstance(images[0], str):  # image paths
                images = [os.path.join(self.image_dir, image) for image in images]

            processed_images = [] if len(images) != 0 else None  # text-only data
            for image in images:
                processed_images.append(process_image(image, self.min_pixels, self.max_pixels))

            model_inputs = self.processor(processed_images, [prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"images": images}
        elif self.video_key in example:
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            videos = example.pop(self.video_key)
            if self.image_dir is not None and len(videos) != 0 and isinstance(videos[0], str):  # video paths
                videos = [os.path.join(self.image_dir, video) for video in videos]

            processed_videos = [] if len(videos) != 0 else None  # text-only data
            video_fps_list = []
            for video in videos:
                processed_video, video_fps = process_video(
                    video, self.min_pixels, self.max_pixels, self.video_fps, return_fps=True
                )
                processed_videos.append(processed_video)
                video_fps_list.append(video_fps)

            model_inputs = self.processor(
                videos=processed_videos, text=[prompt], add_special_tokens=False, return_tensors="pt"
            )
            if "second_per_grid_ts" in self.processor.model_input_names:
                model_inputs["second_per_grid_ts"] = [2.0 / video_sample_fps for video_sample_fps in video_fps_list]

            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]
            example["multi_modal_data"] = {"videos": videos}
        else:
            prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")[0]
            attention_mask = model_inputs.pop("attention_mask")[0]

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            # qwen-vl mrope
            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from ..models.transformers.qwen3_vl import get_rope_index
            else:
                from ..models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask,
            )  # (3, seq_length)
            text_position_ids = torch.arange(len(input_ids)).unsqueeze(0)  # (1, seq_length)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0, max=None)  # (seq_length,)

        input_ids, attention_mask, position_ids = VF.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        raw_prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        example["input_ids"] = input_ids
        example["attention_mask"] = attention_mask
        example["position_ids"] = position_ids
        example["raw_prompt_ids"] = raw_prompt_ids
        example["ground_truth"] = example.pop(self.answer_key)
        return example
