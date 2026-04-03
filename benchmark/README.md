# Benchmark

We provide several profiling tools to benchmark our models.

## profile with dataset

Download the dataset below or create your own dataset.

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

Profiling your model with `profile_throughput.py`

```bash
python profile_throughput.py \
 ShareGPT_V3_unfiltered_cleaned_split.json \
 /path/to/your/model \
 --concurrency 64
```

### ShareGPT (HuggingFace)

You can also pass the HuggingFace dataset ID directly (requires the `datasets` package):

```bash
python profile_throughput.py \
  anon8231489123/ShareGPT_Vicuna_unfiltered \
  /path/to/your/model \
  --dataset-format sharegpt \
  --hf-split train \
  --concurrency 64
```

If the dataset repo doesn't load via `datasets.load_dataset`, use `--hf-data-file` to point to the JSON/JSONL file.
By default, HuggingFace dataset IDs are loaded in non-streaming mode for accurate shuffling; use `--hf-streaming` to enable streaming.

### WildChat

`profile_throughput.py` also supports the WildChat dataset from HuggingFace:

```bash
python profile_throughput.py \
  allenai/WildChat \
  /path/to/your/model \
  --dataset-format wildchat \
  --hf-split train \
  --concurrency 64
```

Note: loading HuggingFace datasets requires the `datasets` package.
By default, HuggingFace dataset IDs are loaded in non-streaming mode for accurate shuffling; use `--hf-streaming` to enable streaming.

### GSM8K

`profile_throughput.py` also supports the GSM8K evaluation split from HuggingFace:

```bash
python profile_throughput.py \
  openai/gsm8k \
  /path/to/your/model \
  --dataset-format gsm8k \
  --hf-split test \
  --hf-config main \
  --concurrency 64
```

`openai/gsm8k` defaults to the `main` config and `test` split. If your local `datasets` metadata exposes the evaluation split as `validation`, the loader falls back automatically.

## profile restful api

`profile_restful_api.py` is used to do benchmark on api server.

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json

python3 profile_restful_api.py --backend lmdeploy --dataset-path ./ShareGPT_V3_unfiltered_cleaned_split.json
```
