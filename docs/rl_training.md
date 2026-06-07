# Stage 2/3 — GRPO and PHI (EasyR1)

These stages start from the SFT base produced by Stage 1 and optimize it with
policy-gradient RL against a **sample-level F1** ICD reward. Two entry points
ship as templates:

- **GRPO** — `EasyR1/examples/qwen2_5_1_5b_icd_grpo_top50_nohint_4gpu.sh`
- **PHI** — `EasyR1/tasks/run_iterative_hint_training_full_hintE_desc_4gpu.sh`

Everything else (dataset, label scope, backbone, β) is an env var or a
Hydra-style CLI override — no code edits. See the top-level `README.md` for the
override recipes (including MIMIC-IV / ICD-10).

After Stage 1, point these at your SFT outputs:

```bash
export SFT_MODEL_PATH=$PWD/models/qwen2.5-1.5b-icd-sft-top50        # top-50 base
export SFT_MODEL_PATH_FULL=$PWD/models/qwen2.5-1.5b-icd-sft-full    # full-code base
```

---

## Reference point — SFT only (no RL)

Evaluate the Stage-1 model directly to get the no-RL baseline number:

```bash
python EasyR1/tasks/eval_checkpoints_on_test.py \
  --base_model "$SFT_MODEL_PATH" \
  --test_parquet data/icd_grpo_top50_nothink/test.parquet \
  --output_dir EasyR1/tasks/eval_sft_top50
```

---

## GRPO (no hint)

Pure GRPO — no value critic, group-relative advantages from `rollout.n=8`
samples per prompt, standardized sample-level F1 reward.

```bash
bash EasyR1/examples/qwen2_5_1_5b_icd_grpo_top50_nohint_4gpu.sh
```

Key knobs (set in the script; override via env or CLI):

| Knob | Default |
|---|---|
| `algorithm.adv_estimator` | `grpo` |
| `algorithm.kl_coef` | 1e-3 |
| `worker.rollout.n` | 8 |
| `worker.actor.global_batch_size` | 1024 (top50) / 512 (full) |
| `worker.actor.model.lora.rank` | 8 |
| `worker.reward.reward_function_kwargs.beta` | 1.0 (F1-balanced) |
| `data.max_response_length` | 196 (top50) / 512 (full) |
| `trainer.total_epochs` | 5 (top50) / 1–3 (full) |
| `data.p_inject_hint` | 0.0 (hints disabled) |

Budget (4×H100, top-50): ~7 min/step × ~130 steps × 5 epochs ≈ 1–1.5 days.

---

## PHI — Progressive Hint Injection

A missed-code curriculum on top of GRPO. Round 0 runs the SFT checkpoint
hint-free to seed each example's missed-code pool `H = Y \ Ŷ`; each later round
trains a GRPO epoch with stochastic, description-annotated hints, then re-runs
inference to refresh the pool. Inference is always hint-free; the reward target
is always the full gold set.

```bash
export SFT_MODEL_PATH_FULL=$PWD/models/qwen2.5-1.5b-icd-sft-full
CUDA_VISIBLE_DEVICES=0,1,2,3 ROUNDS=3 NUM_GPUS=4 \
  bash EasyR1/tasks/run_iterative_hint_training_full_hintE_desc_4gpu.sh
```

Hint mechanics (paper defaults, all overridable): inject probability 0.5; 1–5
codes sampled without replacement; rare/low-recall priority with clip + temp 3.0;
ICD descriptions from `data/icd9_descriptions.json` (point `ICD9_DESC_PATH` at
`icd10_descriptions.json` for MIMIC-IV, and pass `data.hint_icd_version=icd10`).

Implemented in:
- `verl/utils/dataset.py` — hint rendering in `_build_hint_line`
- `tasks/inference_on_train.py` + `tasks/update_hint_pool.py` — the round-update loop
- `tasks/prepare_hint_data.py` — builds the `hint_pool` parquet column

---

## Eval

The same eval script scores the SFT base and every saved `global_step_*` LoRA
adapter on the test parquet, with a closed-set label space derived from the test
parquet (matching the PLM-ICD eval protocol):

```bash
python EasyR1/tasks/eval_checkpoints_on_test.py \
  --base_model "$SFT_MODEL_PATH" \
  --test_parquet data/icd_grpo_top50_nothink/test.parquet \
  --ckpt_root EasyR1/checkpoints/easy_r1/qwen2_5_1_5b_icd_grpo_top50_nohint \
  --output_dir EasyR1/tasks/eval_grpo_top50_sweep
```

---

## Macro-aware reward (future-work extension)

Any run can swap the plain F1 reward for the **weighted F-beta** reward, which
weights rare codes higher to push macro-F1 / long-tail recall:

```bash
python EasyR1/tasks/prepare_icd_reward_weights.py \
  --train_parquet data/icd_grpo_top50_nothink/train.parquet \
  --output data/code_weights_top50.json --alpha 0.5

bash EasyR1/examples/qwen2_5_1_5b_icd_grpo_top50_nohint_4gpu.sh \
  worker.reward.reward_function_kwargs.reward_type=weighted_fbeta \
  worker.reward.reward_function_kwargs.code_weight_path=$PWD/data/code_weights_top50.json
```

This corresponds to the Fβ / class-balanced reward directions discussed in the
paper's Future Directions; it is not part of the headline results.
