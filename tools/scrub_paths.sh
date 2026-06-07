#!/bin/bash
# One-time scrub script: convert absolute /data/... paths in the EasyR1
# subtree to relative ${REPO_ROOT}/... paths or env-var defaults.
#
# Run from the repo root:
#   bash tools/scrub_paths.sh
#
# After running, verify zero leaked absolute paths:
#   grep -rn "/data/" EasyR1/ --include="*.sh" --include="*.py"
#   grep -rn "/home/" EasyR1/ --include="*.sh" --include="*.py"

set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/3] Insert REPO_ROOT line after SCRIPT_DIR line in shell scripts..."
# Match the existing SCRIPT_DIR pattern; only insert when REPO_ROOT not yet defined.
while IFS= read -r f; do
    if grep -q 'REPO_ROOT=' "$f"; then continue; fi
    if grep -qE 'SCRIPT_DIR="\$\(cd ' "$f"; then
        sed -i '/SCRIPT_DIR="\$(cd /a\
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"' "$f"
    fi
done < <(find EasyR1 -name "*.sh")

echo "[2/3] Replace hardcoded /data/* paths (longer patterns first)..."
# Order matters: longer / more specific patterns must be substituted before
# their shorter prefixes, otherwise the prefix swallows them mid-replace.
find EasyR1 \( -name "*.sh" -o -name "*.py" -o -name "*.md" -o -name "*.yaml" -o -name "*.yml" -o -name "*.jinja" \) -print0 | \
    xargs -0 sed -i \
        -e 's|/data/LlamaFactory/outputs/qwen2\.5-1\.5b/icd_naive_sft_mimic3_top50|${SFT_MODEL_PATH:-${REPO_ROOT}/models/qwen2.5-1.5b-icd-sft-top50}|g' \
        -e 's|/data/LlamaFactory/outputs/qwen2\.5-1\.5b/icd-naive-sft-mimic3-full|${SFT_MODEL_PATH_FULL:-${REPO_ROOT}/models/qwen2.5-1.5b-icd-sft-full}|g' \
        -e 's|/data/LlamaFactory/outputs/qwen3-4b/icd_naive_sft_mimic3_top50|${SFT_MODEL_PATH_QWEN3:-${REPO_ROOT}/models/qwen3-4b-icd-sft-top50}|g' \
        -e 's|/data/LlamaFactory/data/icd_grpo_mimic3_top50_nothink_hint_beta1_hint5|${REPO_ROOT}/data/icd_grpo_top50_hint_beta1_hint5|g' \
        -e 's|/data/LlamaFactory/data/icd_grpo_mimic3_top50_nothink_hint_smoke|${REPO_ROOT}/data/icd_grpo_top50_hint_smoke|g' \
        -e 's|/data/LlamaFactory/data/icd_grpo_mimic3_top50_nothink_hint|${REPO_ROOT}/data/icd_grpo_top50_hint|g' \
        -e 's|/data/LlamaFactory/data/icd_grpo_mimic3_top50_nothink|${REPO_ROOT}/data/icd_grpo_top50_nothink|g' \
        -e 's|/data/LlamaFactory/data/icd_grpo_mimic3_full_hint|${REPO_ROOT}/data/icd_grpo_full_hint|g' \
        -e 's|/data/LlamaFactory/data/icd_grpo_mimic3_full|${REPO_ROOT}/data/icd_grpo_full|g' \
        -e 's|/data/LlamaFactory/data/icd_naive_sft_mimic3_top50|${REPO_ROOT}/data/icd_sft_top50|g' \
        -e 's|/data/LlamaFactory/data/icd_naive_sft_mimic3_full|${REPO_ROOT}/data/icd_sft_full|g' \
        -e 's|/data/LlamaFactory/data|${REPO_ROOT}/data|g' \
        -e 's|/data/LlamaFactory/outputs|${REPO_ROOT}/models|g' \
        -e 's|/data/LlamaFactory|${REPO_ROOT}/LlamaFactory|g' \
        -e 's|/data/anaconda3/envs/llama/bin/python|python|g' \
        -e 's|/data/EasyR1|${REPO_ROOT}/EasyR1|g' \
        -e 's|/data/ICD_Coding/icd9_descriptions.json|${ICD9_DESC_PATH:-${REPO_ROOT}/data/icd9_descriptions.json}|g' \
        -e 's|/data/ICD_Coding|${REPO_ROOT}/data|g' \
        -e 's|/data/medical-coding-reproducibility|${MED_CODING_REPRO_ROOT}|g' \
        -e 's|/data/PLM-ICD|${PLM_ICD_ROOT}|g' \
        -e 's|/data/explainable-medical-coding|${XMC_ROOT}|g' \
        -e 's|/data/hdt|${HDT_ROOT}|g' \
        -e 's|/data/icd9_top50|${REPO_ROOT}/data/icd9_top50|g' \
        -e 's|/data/icd9_full|${REPO_ROOT}/data/icd9_full|g' \
        -e 's|/data/icd10_top50|${REPO_ROOT}/data/icd10_top50|g' \
        -e 's|/data/icd10_full|${REPO_ROOT}/data/icd10_full|g'

echo "[3/3] Done. Verify:"
echo "  grep -rn '/data/' EasyR1/ --include='*.sh' --include='*.py' --include='*.md' --include='*.yaml'"
echo "  grep -rn '/home/' EasyR1/ --include='*.sh' --include='*.py'"
echo "  grep -rn 'wandb_v1_' . 2>/dev/null"
