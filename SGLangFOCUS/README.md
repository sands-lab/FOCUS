## Design Overview

FOCUS is an inference system for diffusion LLMs (DLLMs). During block
diffusion decoding, many tokens are processed even though only a small subset
is ready to be decoded. FOCUS uses attention-derived importance from early
layers to retain likely decodable tokens and evict the rest, reducing redundant
computation.

This repository implements FOCUS on top of [SGLang](https://github.com/sgl-project/sglang).
It supports SDAR and LLaDA2.0 models and preserves SGLang's OpenAI-compatible
serving API. The SDAR recipe below is the tested path.

## Key Implementation Files

- [`python/sglang/srt/dllm/config.py`](python/sglang/srt/dllm/config.py): DLLM and FOCUS configuration.
- [`python/sglang/srt/dllm/mixin/scheduler.py`](python/sglang/srt/dllm/mixin/scheduler.py): FOCUS result handling and KV-cache lifecycle.
- [`python/sglang/srt/managers/schedule_batch.py`](python/sglang/srt/managers/schedule_batch.py): DLLM prefix and block scheduling.
- [`python/sglang/srt/managers/schedule_policy.py`](python/sglang/srt/managers/schedule_policy.py): FDFO staging and admission policy.
- [`python/sglang/srt/mem_cache/common.py`](python/sglang/srt/mem_cache/common.py): KV-cache release for diffusion requests.
- [`python/sglang/srt/models/sdar.py`](python/sglang/srt/models/sdar.py) and [`python/sglang/srt/models/llada2.py`](python/sglang/srt/models/llada2.py): FOCUS-aware model execution.

## Install (CUDA)

Create a CUDA-capable Python environment that matches your PyTorch and CUDA
installation, then install this checkout:

```bash
pip install -e ./python
```

The examples below use `PYTHONPATH=python` so that the launcher always uses
this checkout.

## Run FOCUS with SGLang

Use [`configs/sdar_focus.yaml`](configs/sdar_focus.yaml) for
`JetLM/SDAR-8B-Chat-b32`. The matching
[`configs/llada2_focus.yaml`](configs/llada2_focus.yaml) is provided for
`inclusionAI/LLaDA2.0-mini`.

Start a server for SDAR:

```bash
MODEL=JetLM/SDAR-8B-Chat-b32

PYTHONPATH=python python -m sglang.launch_server \
  --model-path "$MODEL" \
  --served-model-name "$MODEL" \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 30000 \
  --tp-size 1 \
  --mem-fraction-static 0.7 \
  --disable-cuda-graph \
  --dllm-algorithm LowConfidence \
  --dllm-algorithm-config configs/sdar_focus.yaml \
  --dllm-fdfo
```

FOCUS uses delayed KV-cache commits. CUDA graphs are disabled for this mode
because the active token set changes between diffusion passes.

Send requests through SGLang's OpenAI-compatible API:

```bash
curl http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "JetLM/SDAR-8B-Chat-b32",
    "messages": [{"role": "user", "content": "What is 1 + 1?"}],
    "temperature": 0,
    "max_tokens": 32
  }'
```

To run the original DLLM or delayed-cache baseline, set `enable_focus` and
`enable_delayed_cache` to `false/false` or `false/true`, respectively.

## Benchmarking

Throughput results are written to `./results/focus_sglang`. Run the commands
from the repository root. Set `PYTHON_BIN` when the SGLang environment does
not use `python` by default.

- FOCUS throughput: [`benchmark/focus/run_readme_throughput_sglang.sh`](benchmark/focus/run_readme_throughput_sglang.sh)

  ```bash
  SUITES=focus benchmark/focus/run_readme_throughput_sglang.sh <dataset_id> <model_id> [alpha]
  ```

  Example:

  ```bash
  SUITES=focus benchmark/focus/run_readme_throughput_sglang.sh anon8231489123/ShareGPT_Vicuna_unfiltered JetLM/SDAR-8B-Chat-b32
  SUITES=focus benchmark/focus/run_readme_throughput_sglang.sh anon8231489123/ShareGPT_Vicuna_unfiltered JetLM/SDAR-8B-Chat-b32 1.8
  ```

- Original DLLM throughput:

  ```bash
  SUITES=base benchmark/focus/run_readme_throughput_sglang.sh <dataset_id> <model_id>
  ```

  Example:

  ```bash
  SUITES=base benchmark/focus/run_readme_throughput_sglang.sh anon8231489123/ShareGPT_Vicuna_unfiltered JetLM/SDAR-8B-Chat-b32
  ```

- Block size comparison for SDAR:

  ```bash
  SUITES=block benchmark/focus/run_readme_throughput_sglang.sh <dataset_id>
  ```

  This runs `JetLM/SDAR-8B-Chat-b16` and `JetLM/SDAR-8B-Chat-b64` for both
  FOCUS and Base settings.

- Delayed-cache baseline for SDAR:

  ```bash
  SUITES=delayed benchmark/focus/run_readme_throughput_sglang.sh <dataset_id>
  ```

  This runs `JetLM/SDAR-8B-Chat-b32` with delayed cache enabled and FOCUS
  disabled.

Dataset notes:

- `dataset_id` can be a Hugging Face dataset ID or a local JSON/JSONL path
  supported by [`benchmark/focus/profile_throughput_sglang.py`](benchmark/focus/profile_throughput_sglang.py).
- Hugging Face dataset IDs require the `datasets` package and network access.

## OpenCompass Evaluation

The included accuracy driver performs the complete workflow: it starts an
SGLang server, waits for it to become ready, runs OpenCompass through the
OpenAI-compatible API, then stops the server. Run one command for a selected
mode (`baseline`, `delayed`, or `focus`):

```bash
benchmark/focus/run_opencompass_accuracy_compare.sh sglang sdar focus
```

The driver uses `python` from `PATH` by default. Set `PYTHON_BIN` to use one
different environment, or set `SGLANG_PYTHON` and `OPENCOMPASS_PYTHON`
separately when needed; no separate manual server command is required. Set
`OPENCOMPASS_ROOT` if the OpenCompass checkout is not located at
`../opencompass-0.5.1.post1`.

## Citation

If you find FOCUS useful in your work, please cite:

```bibtex
@inproceedings{
liang2026focus,
title={{FOCUS}: {DLLM}s Know How to Tame Their Compute Bound},
author={Kaihua Liang and Xin Tan and An Zhong and Hong Xu and Marco Canini},
booktitle={Forty-third International Conference on Machine Learning},
year={2026},
url={https://openreview.net/forum?id=40fUEdwvH3}
}
```

## Acknowledgements

This implementation builds on [SGLang](https://github.com/sgl-project/sglang),
[OpenCompass](https://github.com/open-compass/opencompass),
[SDAR](https://github.com/JetAstra/SDAR), and
[LLaDA2.0](https://github.com/inclusionAI/LLaDA2.0).
