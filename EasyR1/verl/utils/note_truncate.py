"""Smart truncation for clinical-note prompts that exceed max_prompt_length.

When a discharge-summary prompt is too long, we shrink the note content rather
than dropping the sample. Strategy is tiered:

- Tier 1: if the note has a "Brief Hospital Course" section, truncate only
  that section (middle-out). This preserves Discharge Diagnoses / Medications
  at the end, which carry the most ICD-coding signal.
- Tier 2: no BHC anchor found. Fall back to whole-note middle-out truncation
  (keep head + tail, drop middle).

If the note region itself cannot be located (no "Discharge Summary:" anchor),
last-resort truncation slices the whole prompt from the end. This should not
happen with prompts produced by tasks/prepare_icd_grpo.py.
"""

from __future__ import annotations

from typing import Optional, Tuple


# Section header variants found in MIMIC-III discharge summaries (88% coverage
# across the top-50 corpus; see data probe in plan file).
HOSPITAL_COURSE_HEADERS: list[str] = [
    "Brief Hospital Course:",
    "HOSPITAL COURSE:",
    "Hospital course:",
]

# Next-section anchors that mark where the hospital-course section ends.
# Ordered loosely by frequency so common ones hit first.
HOSPITAL_COURSE_END_ANCHORS: list[str] = [
    "Medications on Admission:", "MEDICATIONS ON ADMISSION:",
    "Discharge Medications:", "DISCHARGE MEDICATIONS:",
    "Discharge Diagnosis:", "DISCHARGE DIAGNOSIS:",
    "Discharge Diagnoses:", "DISCHARGE DIAGNOSES:",
    "Discharge Condition:", "DISCHARGE CONDITION:",
    "Discharge Disposition:",
    "Discharge Instructions:", "DISCHARGE INSTRUCTIONS:",
    "Followup Instructions:", "Followup:",
    "Pertinent Results:",
    "Physical Exam:",
]

# Prompts from prepare_icd_grpo.py put the clinical note between these anchors.
NOTE_START_ANCHOR = "Discharge Summary:\n"
# Hint and Output format: each mark the end of the note, whichever appears
# first.
#
# WARNING:
# - "\n\nHint:" must match RLHFDataset.HINT_TEMPLATE prefix in dataset.py.
# Renaming this marker without updating here causes the affected block to
# leak into the "note" region and be eligible for middle-out truncation.
NOTE_END_ANCHORS: list[str] = [
    "\n\nHint:",
    "\n\nOutput format:",
]

BHC_SEPARATOR = "\n\n[... hospital course truncated ...]\n\n"
NOTE_SEPARATOR = "\n\n[... middle portion of note truncated ...]\n\n"

# Floor on the remaining section length; below this, truncation is too
# destructive to be useful and we fall back to the next tier.
MIN_TRUNCATED_SECTION_TOKENS = 100


def _find_header_at_line_start(text: str, header: str) -> int:
    """Return first index of `header` that is at start-of-text or preceded by
    a newline. Guards against matching inline occurrences like
    "... Hospital course: was complicated ..." inside prose.

    Returns -1 if no line-start occurrence exists.
    """
    start = 0
    while True:
        idx = text.find(header, start)
        if idx < 0:
            return -1
        if idx == 0 or text[idx - 1] == "\n":
            return idx
        start = idx + 1


def _find_hospital_course_bounds(note: str) -> Optional[Tuple[int, int]]:
    """Return (content_start, content_end) for the Brief Hospital Course
    section inside `note`, or None if no header matches.

    content_start is the index AFTER the header line so the header itself
    stays in the preserved prefix. Headers must appear at line start so we
    don't match inline prose like "Her hospital course: was complicated ...".
    """
    for header in HOSPITAL_COURSE_HEADERS:
        idx = _find_header_at_line_start(note, header)
        if idx < 0:
            continue
        content_start = idx + len(header)
        nearest_end = len(note)
        for anchor in HOSPITAL_COURSE_END_ANCHORS:
            i = note.find(anchor, content_start)
            if 0 <= i < nearest_end:
                nearest_end = i
        return content_start, nearest_end
    return None


def _count_tokens(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _middle_out(token_ids: list[int], budget: int, sep_ids: list[int]) -> list[int]:
    """Keep budget//2 head tokens + budget//2 tail tokens, with sep in between."""
    if budget <= len(sep_ids):
        # Budget so tight that the separator alone would overfill — just truncate.
        return token_ids[:budget]
    eff = budget - len(sep_ids)
    half = eff // 2
    return token_ids[:half] + sep_ids + token_ids[-(eff - half):]


def truncate_note_smart(
    prompt: str,
    tokenizer,
    max_tokens: int,
    safety_margin: int = 20,
) -> str:
    """Return `prompt` with its clinical-note section shrunk to fit max_tokens.

    If the full prompt already fits, returned unchanged. The caller should
    subtract chat-template overhead from max_tokens before calling, since this
    function measures the raw text only.
    """
    if _count_tokens(tokenizer, prompt) <= max_tokens:
        return prompt

    # Locate the note region. Fail loudly if the anchor is missing — silent
    # end-clip would eat the "Output format:" block (it lives at the tail),
    # producing prompts with no format instruction and wrecking the reward
    # signal in a way that's very hard to debug. Caller should either ensure
    # the prep-script anchor is present or catch this ValueError.
    note_start_idx = prompt.find(NOTE_START_ANCHOR)
    if note_start_idx < 0:
        raise ValueError(
            f"NOTE_START_ANCHOR {NOTE_START_ANCHOR!r} not found in overlong "
            f"prompt ({_count_tokens(tokenizer, prompt)} tokens); cannot "
            f"safely truncate. Check that prepare_icd_grpo.py produced the "
            f"expected anchor text."
        )
    note_start_idx += len(NOTE_START_ANCHOR)

    note_end_idx = len(prompt)
    for a in NOTE_END_ANCHORS:
        i = prompt.find(a, note_start_idx)
        if 0 <= i < note_end_idx:
            note_end_idx = i

    prefix = prompt[:note_start_idx]
    note = prompt[note_start_idx:note_end_idx]
    suffix = prompt[note_end_idx:]

    prefix_tokens = _count_tokens(tokenizer, prefix)
    suffix_tokens = _count_tokens(tokenizer, suffix)
    note_budget = max_tokens - prefix_tokens - suffix_tokens - safety_margin
    if note_budget <= MIN_TRUNCATED_SECTION_TOKENS:
        raise ValueError(
            f"Boilerplate ({prefix_tokens + suffix_tokens} tokens) leaves "
            f"only {note_budget} tokens for the clinical note; cannot "
            f"meaningfully truncate. Increase max_prompt_length or shorten "
            f"hint/format spec."
        )

    bhc_sep_ids = tokenizer.encode(BHC_SEPARATOR, add_special_tokens=False)
    note_sep_ids = tokenizer.encode(NOTE_SEPARATOR, add_special_tokens=False)

    # Tier 1: targeted Brief Hospital Course truncation.
    bhc_bounds = _find_hospital_course_bounds(note)
    if bhc_bounds is not None:
        bhc_start, bhc_end = bhc_bounds
        note_before = note[:bhc_start]
        bhc = note[bhc_start:bhc_end]
        note_after = note[bhc_end:]

        before_tokens = _count_tokens(tokenizer, note_before)
        after_tokens = _count_tokens(tokenizer, note_after)
        bhc_budget = note_budget - before_tokens - after_tokens

        if bhc_budget >= MIN_TRUNCATED_SECTION_TOKENS:
            bhc_ids = tokenizer.encode(bhc, add_special_tokens=False)
            if len(bhc_ids) > bhc_budget:
                bhc_ids = _middle_out(bhc_ids, bhc_budget, bhc_sep_ids)
            truncated_bhc = tokenizer.decode(bhc_ids, skip_special_tokens=True)
            result = prefix + note_before + truncated_bhc + note_after + suffix
            # Verify fit — decode roundtrip can drift 1-2 tokens.
            if _count_tokens(tokenizer, result) <= max_tokens:
                return result
        # bhc_budget too small, or verification failed → fall through to Tier 2.

    # Tier 2: whole-note middle-out truncation.
    note_ids = tokenizer.encode(note, add_special_tokens=False)
    note_ids = _middle_out(note_ids, note_budget, note_sep_ids)
    result = prefix + tokenizer.decode(note_ids, skip_special_tokens=True) + suffix

    # If still overflows (rare: decode drift), shave a few more tokens.
    # We loop a few times to handle compounding drift — after 3 attempts the
    # budget is so constrained that we raise so the caller sees a clear error
    # instead of silently returning an oversize prompt.
    for attempt in range(3):
        overflow = _count_tokens(tokenizer, result) - max_tokens
        if overflow <= 0:
            return result
        new_budget = note_budget - overflow - safety_margin * (attempt + 1)
        if new_budget < MIN_TRUNCATED_SECTION_TOKENS:
            break
        note_ids = tokenizer.encode(note, add_special_tokens=False)
        note_ids = _middle_out(note_ids, new_budget, note_sep_ids)
        result = prefix + tokenizer.decode(note_ids, skip_special_tokens=True) + suffix

    raise ValueError(
        f"Could not fit prompt into {max_tokens} tokens after 3 retries "
        f"(final length {_count_tokens(tokenizer, result)}). This usually "
        f"means tokenizer decode/encode drift is unusually large; consider "
        f"increasing safety_margin or max_prompt_length."
    )


def chat_template_overhead(
    tokenizer,
    enable_thinking: bool = False,
) -> int:
    """Token cost of an empty user-turn chat template.

    Pass the same `enable_thinking` flag you'll use downstream so the measured
    overhead matches reality (Qwen3 thinking variants can inject `<think>`
    tokens when enable_thinking=True).
    """
    empty = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=enable_thinking,
    )
    return len(tokenizer.encode(empty, add_special_tokens=False))


def truncate_for_chat_prompt(
    prompt: str,
    tokenizer,
    max_prompt_length: int,
    template_overhead: Optional[int] = None,
    enable_thinking: bool = False,
    safety_margin: int = 20,
) -> str:
    """Truncate `prompt` so that its chat-templated form fits in
    `max_prompt_length` tokens.

    This is the integration-level helper. It:
      1. Measures chat-template overhead (if not provided)
      2. Calls truncate_note_smart with budget = max_prompt_length - overhead
      3. Verifies the final templated length and re-shaves if decode/encode
         drift across the template boundary pushes it back over budget
      4. Returns the truncated prompt ready to be wrapped in a user message

    Caller sites: dataset.py._build_messages, inference_on_train.py.

    Raises ValueError if the prompt cannot be made to fit (e.g., boilerplate
    alone exceeds budget, or drift retry exhausted).
    """
    if template_overhead is None:
        template_overhead = chat_template_overhead(tokenizer, enable_thinking)

    budget = max_prompt_length - template_overhead
    if budget <= MIN_TRUNCATED_SECTION_TOKENS:
        raise ValueError(
            f"Chat-template overhead ({template_overhead}) leaves only "
            f"{budget} tokens for content; increase max_prompt_length."
        )

    truncated = truncate_note_smart(prompt, tokenizer, budget, safety_margin)

    # Cross-template verification: apply_chat_template(tokenize=True) may
    # produce different tokenization of specials than encode(tokenize=False),
    # so re-verify through the real rendering path the caller will use.
    templated_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": truncated}],
        add_generation_prompt=True,
        tokenize=True,
        enable_thinking=enable_thinking,
    )
    overflow = len(templated_ids) - max_prompt_length
    if overflow > 0:
        # Tighten budget and retry — drift across template boundary can be
        # a few tokens. We pay an extra safety_margin per retry.
        for attempt in range(3):
            tighter_budget = budget - overflow - safety_margin * (attempt + 1)
            if tighter_budget < MIN_TRUNCATED_SECTION_TOKENS:
                break
            truncated = truncate_note_smart(
                prompt, tokenizer, tighter_budget, safety_margin
            )
            templated_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": truncated}],
                add_generation_prompt=True,
                tokenize=True,
                enable_thinking=enable_thinking,
            )
            overflow = len(templated_ids) - max_prompt_length
            if overflow <= 0:
                return truncated
        raise ValueError(
            f"Chat-templated prompt ({len(templated_ids)} tokens) still "
            f"exceeds max_prompt_length ({max_prompt_length}) after "
            f"{attempt + 1} retries. Tokenizer drift too large."
        )
    return truncated
