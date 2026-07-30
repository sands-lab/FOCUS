#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 <lmdeploy|sglang> <sdar|llada2> [baseline|delayed|focus]" >&2
    exit 2
fi

BACKEND="$1"
MODEL_KEY="$2"
MODE="${3:-baseline}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OPENCOMPASS_ROOT="${OPENCOMPASS_ROOT:-${REPO_ROOT}/../opencompass-0.5.1.post1}"
RESULTS_ROOT="${RESULTS_ROOT:-${REPO_ROOT}/results/focus_sglang/opencompass_accuracy_compare}"
DATASETS="${DATASETS:-gsm8k_gen math500_gen humaneval_gen sanitized_mbpp_gen IFEval_gen}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LMDEPLOY_PYTHON="${LMDEPLOY_PYTHON:-${PYTHON_BIN}}"
SGLANG_PYTHON="${SGLANG_PYTHON:-${PYTHON_BIN}}"
OPENCOMPASS_PYTHON="${OPENCOMPASS_PYTHON:-${PYTHON_BIN}}"
PORT="${PORT:-30000}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"

mkdir -p "${RESULTS_ROOT}"

case "${MODEL_KEY}" in
    sdar)
        MODEL_PATH="JetLM/SDAR-8B-Chat-b32"
        MODEL_ABBR_BASE="sdar-8b"
        INCLUDE_SYSTEM_PROMPT="False"
        SYSTEM_PROMPT="You are a helpful assistant."
        SGLANG_META_TEMPLATE="api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
)"
        SGLANG_EXTRA_KWARGS=""
        SGLANG_TEMPERATURE="1.0"
        LMDEPLOY_TEMPERATURE="1.0"
        LMDEPLOY_STOP_WORDS=""
        MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
        OPENCOMPASS_BATCH_SIZE="${OPENCOMPASS_BATCH_SIZE:-${MAX_RUNNING_REQUESTS}}"
        ;;
    llada2)
        MODEL_PATH="inclusionAI/LLaDA2.0-mini"
        MODEL_ABBR_BASE="llada2-mini"
        INCLUDE_SYSTEM_PROMPT="True"
        SYSTEM_PROMPT="You are a helpful assistant."
        SGLANG_META_TEMPLATE="api_meta_template = dict(
    begin=dict(role='SYSTEM', prompt='You are a helpful assistant.'),
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
)"
        SGLANG_EXTRA_KWARGS="        openai_extra_kwargs=dict(stop=['<|role_end|>']),"
        SGLANG_TEMPERATURE="0.0"
        LMDEPLOY_TEMPERATURE="0.0"
        LMDEPLOY_STOP_WORDS="            stop_words=['<|role_end|>'],"
        MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-32}"
        OPENCOMPASS_BATCH_SIZE="${OPENCOMPASS_BATCH_SIZE:-${MAX_RUNNING_REQUESTS}}"
        ;;
    *)
        echo "Unknown model '${MODEL_KEY}'. Expected sdar or llada2." >&2
        exit 2
        ;;
esac

case "${MODE}" in
    baseline)
        THRESHOLD=0.9
        ENABLE_DELAYED="false"
        ENABLE_FOCUS="false"
        PY_ENABLE_DELAYED="False"
        PY_ENABLE_FOCUS="False"
        ;;
    delayed)
        THRESHOLD=0.8
        ENABLE_DELAYED="true"
        ENABLE_FOCUS="false"
        PY_ENABLE_DELAYED="True"
        PY_ENABLE_FOCUS="False"
        ;;
    focus)
        THRESHOLD=0.8
        ENABLE_DELAYED="true"
        ENABLE_FOCUS="true"
        PY_ENABLE_DELAYED="True"
        PY_ENABLE_FOCUS="True"
        ;;
    *)
        echo "Unknown mode '${MODE}'. Expected baseline, delayed, or focus." >&2
        exit 2
        ;;
esac

MODEL_CONFIG="${MODEL_KEY}_${BACKEND}_${MODE}"
MODEL_ABBR="${MODEL_ABBR_BASE}-${BACKEND}-${MODE}"

write_lmdeploy_config() {
    local path="$1"
    mkdir -p "$(dirname "${path}")"
    cat >"${path}" <<EOF
from opencompass.models import TurboMindModelwithChatTemplate


models = [
    dict(
        type=TurboMindModelwithChatTemplate,
        abbr='${MODEL_ABBR}',
        path='${MODEL_PATH}',
        backend='pytorch',
        engine_config=dict(
            tp=1,
            max_batch_size=${OPENCOMPASS_BATCH_SIZE},
            session_len=2048,
            dllm_block_length=32,
            dllm_denoising_steps=32,
            dllm_confidence_threshold=${THRESHOLD},
            dllm_unmasking_strategy='low_confidence_dynamic',
            dllm_enable_delayed_cache=${PY_ENABLE_DELAYED},
            dllm_enable_focus=${PY_ENABLE_FOCUS},
            dllm_focus_alpha=1.5,
            eager_mode=False,
        ),
        gen_config=dict(
            top_p=1.0,
            top_k=0,
            temperature=${LMDEPLOY_TEMPERATURE},
            do_sample=False,
            max_new_tokens=1024,
        ),
${LMDEPLOY_STOP_WORDS}
        include_system_prompt=${INCLUDE_SYSTEM_PROMPT},
        system_prompt='${SYSTEM_PROMPT}',
        max_seq_len=2048,
        max_out_len=1024,
        batch_size=${OPENCOMPASS_BATCH_SIZE},
        run_cfg=dict(num_gpus=1),
    )
]
EOF
}

write_sglang_config() {
    local path="$1"
    mkdir -p "$(dirname "${path}")"
    cat >"${path}" <<EOF
from opencompass.models import OpenAISDK


${SGLANG_META_TEMPLATE}


models = [
    dict(
        type=OpenAISDK,
        path='${MODEL_PATH}',
        tokenizer_path='${MODEL_PATH}',
        abbr='${MODEL_ABBR}',
        openai_api_base='http://127.0.0.1:${PORT}/v1',
        key='EMPTY',
        meta_template=api_meta_template,
        query_per_second=${MAX_RUNNING_REQUESTS},
        max_workers=${MAX_RUNNING_REQUESTS},
        max_seq_len=2048,
        max_out_len=1024,
        batch_size=${OPENCOMPASS_BATCH_SIZE},
        temperature=${SGLANG_TEMPERATURE},
${SGLANG_EXTRA_KWARGS}
        retry=10,
        verbose=False,
        run_cfg=dict(num_gpus=0),
    )
]
EOF
}

wait_for_server() {
    local url="$1"
    local log_file="$2"
    for _ in $(seq 1 180); do
        if curl -fsS "${url}/models" >/dev/null 2>&1; then
            return 0
        fi
        if [[ -f "${log_file}" ]] && grep -E "Traceback|RuntimeError|ValueError" "${log_file}" >/dev/null; then
            tail -n 80 "${log_file}" >&2
            return 1
        fi
        sleep 5
    done
    echo "Timed out waiting for SGLang server at ${url}" >&2
    [[ -f "${log_file}" ]] && tail -n 80 "${log_file}" >&2
    return 1
}

run_opencompass() {
    local python_bin="$1"
    local model_config="$2"
    local work_dir="$3"
    local log_file="$4"
    shift 4

    mkdir -p "${work_dir}"
    (
        cd "${OPENCOMPASS_ROOT}"
        PYTHONPATH="${OPENCOMPASS_ROOT}:${REPO_ROOT}/python:${PYTHONPATH:-}" \
            "$python_bin" run.py \
            --models "${model_config}" \
            --datasets ${DATASETS} \
            --work-dir "${work_dir}" \
            "$@"
    ) >"${log_file}" 2>&1
}

if [[ "${BACKEND}" == "lmdeploy" ]]; then
    RUN_DIR="${RESULTS_ROOT}/${MODEL_KEY}/lmdeploy/${MODE}"
    GENERATED_CONFIG_DIR="${RUN_DIR}/generated_configs"
    write_lmdeploy_config "${GENERATED_CONFIG_DIR}/models/${MODEL_CONFIG}.py"
    run_opencompass "${LMDEPLOY_PYTHON}" "${MODEL_CONFIG}" \
        "${RUN_DIR}/work_dir" "${RUN_DIR}/opencompass.log" \
        --config-dir "${GENERATED_CONFIG_DIR}"
elif [[ "${BACKEND}" == "sglang" ]]; then
    RUN_DIR="${RESULTS_ROOT}/${MODEL_KEY}/sglang/${MODE}"
    mkdir -p "${RUN_DIR}"
    GENERATED_CONFIG_DIR="${RUN_DIR}/generated_configs"
    write_sglang_config "${GENERATED_CONFIG_DIR}/models/${MODEL_CONFIG}.py"
    DLLM_CONFIG="${RUN_DIR}/dllm_lowconfidence.yaml"
    SERVER_LOG="${RUN_DIR}/sglang_server.log"

    cat >"${DLLM_CONFIG}" <<EOF
block_size: 32
enable_delayed_cache: ${ENABLE_DELAYED}
enable_focus: ${ENABLE_FOCUS}
focus_alpha: 1.5
threshold: ${THRESHOLD}
EOF

    PYTHONPATH="${REPO_ROOT}/python:${OPENCOMPASS_ROOT}:${PYTHONPATH:-}" \
        "${SGLANG_PYTHON}" -m sglang.launch_server \
        --model-path "${MODEL_PATH}" \
        --host 127.0.0.1 \
        --port "${PORT}" \
        --trust-remote-code \
        --mem-fraction-static "${MEM_FRACTION_STATIC}" \
        --max-running-requests "${MAX_RUNNING_REQUESTS}" \
        --attention-backend "${ATTENTION_BACKEND}" \
        --cuda-graph-backend-decode disabled \
        --cuda-graph-backend-prefill disabled \
        --dllm-algorithm LowConfidence \
        --dllm-algorithm-config "${DLLM_CONFIG}" \
        >"${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!
    trap 'kill ${SERVER_PID} >/dev/null 2>&1 || true; wait ${SERVER_PID} >/dev/null 2>&1 || true' EXIT

    wait_for_server "http://127.0.0.1:${PORT}/v1" "${SERVER_LOG}"
    SGLANG_OPENAI_API_BASE="http://127.0.0.1:${PORT}/v1" \
        run_opencompass "${OPENCOMPASS_PYTHON}" "${MODEL_CONFIG}" \
        "${RUN_DIR}/work_dir" "${RUN_DIR}/opencompass.log" \
        --config-dir "${GENERATED_CONFIG_DIR}"
else
    echo "Unknown backend '${BACKEND}'. Expected lmdeploy or sglang." >&2
    exit 2
fi
