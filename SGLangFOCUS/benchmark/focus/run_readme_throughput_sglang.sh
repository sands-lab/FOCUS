#!/usr/bin/env bash
set -u

# Run the throughput matrices described by the LMDeploy FOCUS README, using the
# SGLang implementation in this repository.
#
# Usage:
#   benchmark/focus/run_readme_throughput_sglang.sh <dataset_id> [model_id] [alpha]
#
# Environment knobs:
#   SUITES="base focus delayed block"   # subset: base, focus, delayed, block
#   BATCH_SIZES="32 64 128 256"
#   NUM_PROMPTS=5000
#   MAX_NEW_TOKENS=2048
#   RESULTS_DIR=results/focus_sglang
#   PYTHON_BIN=/path/to/python
#   LOG_LEVEL=warning

DATASET=${1:-anon8231489123/ShareGPT_Vicuna_unfiltered}
MODEL=${2:-JetLM/SDAR-8B-Chat-b32}
ALPHA=${3:-1.5}

SUITES=${SUITES:-"base focus delayed block"}
BATCH_SIZES=${BATCH_SIZES:-"32 64 128 256"}
NUM_PROMPTS=${NUM_PROMPTS:-5000}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
RESULTS_DIR=${RESULTS_DIR:-results/focus_sglang}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-flashinfer}
PYTHON_BIN=${PYTHON_BIN:-python}
LOG_LEVEL=${LOG_LEVEL:-warning}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
export PYTHONPATH="${REPO_ROOT}/python:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ "${PYTHON_BIN}" == */bin/python* ]]; then
  PYTHON_ENV_DIR=$(cd "$(dirname "${PYTHON_BIN}")/.." && pwd)
  if [[ -d "${PYTHON_ENV_DIR}/lib" ]]; then
    export LD_LIBRARY_PATH="${PYTHON_ENV_DIR}/lib:${LD_LIBRARY_PATH:-}"
  fi
fi

mkdir -p "${RESULTS_DIR}"

DATASET_ARGS=()
if [[ "${DATASET}" == *"hendrycks-MATH"* ]]; then
  DATASET_ARGS+=(--dataset-format math)
elif [[ "${DATASET}" == "openai/gsm8k" ]]; then
  DATASET_ARGS+=(--dataset-format gsm8k --hf-split test --hf-config main)
elif [[ "${DATASET}" == "anon8231489123/ShareGPT_Vicuna_unfiltered" ]]; then
  DATASET_ARGS+=(--dataset-format sharegpt --hf-data-file ShareGPT_V3_unfiltered_cleaned_split.json)
fi

dataset_name=$(basename "${DATASET}")
model_name=$(basename "${MODEL}")

has_suite() {
  local needle="$1"
  for suite in ${SUITES}; do
    if [[ "${suite}" == "${needle}" || "${suite}" == "all" ]]; then
      return 0
    fi
  done
  return 1
}

run_case() {
  local kind="$1"
  local dataset="$2"
  local model="$3"
  local block_size="$4"
  local threshold="$5"
  local delayed="$6"
  local focus="$7"
  local alpha="$8"
  local batch_size="$9"
  local label="${kind}_$(basename "${dataset}")_$(basename "${model}")_batch_${batch_size}"

  if [[ "${kind}" == "focus" ]]; then
    label="focus_$(basename "${dataset}")_$(basename "${model}")_alpha_${alpha}_batch_${batch_size}"
  elif [[ "${kind}" == "block_focus" ]]; then
    label="focus_$(basename "${dataset}")_SDAR-b${block_size}_batch_${batch_size}"
  elif [[ "${kind}" == "block_base" ]]; then
    label="base_$(basename "${dataset}")_SDAR-b${block_size}_batch_${batch_size}"
  fi

  local output_file="${RESULTS_DIR}/${label}.log"
  local error_file="${RESULTS_DIR}/${label}.err"
  local json_file="${RESULTS_DIR}/${label}.json"

  echo "========================================="
  echo "Running ${kind}: dataset=${dataset} model=${model} block=${block_size} batch=${batch_size}"
  echo "stdout=${output_file}"
  echo "stderr=${error_file}"
  echo "========================================="

  cmd=(
    "${PYTHON_BIN}" benchmark/focus/profile_throughput_sglang.py
    "${dataset}"
    "${model}"
    --model-path "${model}"
    "${DATASET_ARGS[@]}"
    --eager-mode
    --skip-tokenize
    --skip-detokenize
    --num-prompts "${NUM_PROMPTS}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --repeat-block-detect
    --repeat-block-window "${block_size}"
    --repeat-block-threshold 3
    --dllm-block-length "${block_size}"
    --dllm-confidence-threshold "${threshold}"
    --dllm-focus-alpha "${alpha}"
    --concurrency "${batch_size}"
    --attention-backend "${ATTENTION_BACKEND}"
    --log-level "${LOG_LEVEL}"
    --output-dir "${RESULTS_DIR}"
    --csv "${RESULTS_DIR}/profile_throughput_sglang.csv"
    --json-result "${json_file}"
  )

  if [[ "${delayed}" == "1" ]]; then
    cmd+=(--dllm-enable-delayed-cache)
  fi
  if [[ "${focus}" == "1" ]]; then
    cmd+=(--dllm-enable-focus)
  fi

  "${cmd[@]}" >"${output_file}" 2>"${error_file}"
  status=$?
  if [[ ${status} -eq 0 ]]; then
    echo "Completed: ${label}"
  else
    echo "Failed: ${label}; see ${error_file}"
  fi
  return ${status}
}

overall_status=0

if has_suite base; then
  for batch in ${BATCH_SIZES}; do
    run_case base "${DATASET}" "${MODEL}" 32 0.9 0 0 "${ALPHA}" "${batch}" || overall_status=$?
  done
fi

if has_suite focus; then
  for batch in ${BATCH_SIZES}; do
    run_case focus "${DATASET}" "${MODEL}" 32 0.8 1 1 "${ALPHA}" "${batch}" || overall_status=$?
  done
fi

if has_suite delayed; then
  for batch in ${BATCH_SIZES}; do
    run_case delayed_cache "${DATASET}" "${MODEL}" 32 0.8 1 0 "${ALPHA}" "${batch}" || overall_status=$?
  done
fi

if has_suite block; then
  for block in 16 64; do
    block_model="JetLM/SDAR-8B-Chat-b${block}"
    for batch in ${BATCH_SIZES}; do
      run_case block_focus "${DATASET}" "${block_model}" "${block}" 0.8 1 1 "${ALPHA}" "${batch}" || overall_status=$?
    done
    for batch in ${BATCH_SIZES}; do
      run_case block_base "${DATASET}" "${block_model}" "${block}" 0.9 0 0 "${ALPHA}" "${batch}" || overall_status=$?
    done
  done
fi

echo "========================================="
echo "SGLang README throughput rerun finished"
echo "dataset=${DATASET}"
echo "model=${MODEL}"
echo "suites=${SUITES}"
echo "batch_sizes=${BATCH_SIZES}"
echo "results=${RESULTS_DIR}"
echo "status=${overall_status}"
echo "========================================="

exit ${overall_status}
