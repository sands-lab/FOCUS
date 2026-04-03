#!/bin/bash

# Copyright (c) 2026 SANDS Lab. All rights reserved.

# Script to run baseline throughput benchmark with different batch sizes, datasets, and models
# This script executes the profile_throughput.py benchmark for baseline configuration:
# - Dataset: Specified as command line argument (first parameter)
# - Model: Specified as command line argument (second parameter)
# - Base experiment type: threshold 0.9
# - 4 batch sizes: 32, 64, 128, and 256
# Total per dataset: 4 benchmarks
# Output is saved to ./results directory
#
# Usage: ./run_baseline_throughput_evaluation.sh <dataset_id> <model_id>
# Example: ./run_baseline_throughput_evaluation.sh anon8231489123/ShareGPT_Vicuna_unfiltered JetLM/SDAR-8B-Chat-b32
#          ./run_baseline_throughput_evaluation.sh allenai/WildChat inclusionAI/LLaDA2.0-mini
#          ./run_baseline_throughput_evaluation.sh nlile/hendrycks-MATH-benchmark <model_id>
#          ./run_baseline_throughput_evaluation.sh openai/gsm8k <model_id>

# Check if dataset argument is provided
if [ -z "$1" ]; then
    echo "Error: Dataset ID is required as a command line argument"
    echo "Usage: $0 <dataset_id> <model_id>"
    echo "Example: $0 anon8231489123/ShareGPT_Vicuna_unfiltered JetLM/SDAR-8B-Chat-b32"
    exit 1
fi

# Check if model argument is provided
if [ -z "$2" ]; then
    echo "Error: Model ID is required as a command line argument"
    echo "Usage: $0 <dataset_id> <model_id>"
    echo "Example: $0 anon8231489123/ShareGPT_Vicuna_unfiltered JetLM/SDAR-8B-Chat-b32"
    exit 1
fi

# Get dataset from command line argument
DATASET=$1

# Get model from command line argument
MODEL=$2

# Dataset-specific arguments
DATASET_ARGS=()
if [[ "${DATASET}" == *"hendrycks-MATH"* ]]; then
    DATASET_ARGS+=(--dataset-format math)
elif [[ "${DATASET}" == "openai/gsm8k" ]]; then
    DATASET_ARGS+=(--dataset-format gsm8k --hf-split test --hf-config main)
fi

# Create results directory if it doesn't exist
mkdir -p ./results

# Array of batch sizes to test
BATCH_SIZES=(32 64 128 256)

# ============================================
# Common parameters for all experiments
# ============================================
# --eager-mode: Use eager execution mode
# --backend pytorch: Use PyTorch as the backend
# --skip-tokenize: Skip tokenization step
# --skip-detokenize: Skip detokenization step
# --num-prompts 5000: Target number of prompts to process; smaller datasets
#   such as GSM8K are automatically capped by available samples
# --use-uvloop: Use uvloop for async operations
# --dllm-block-length 32: DLLM block length parameter
# --dllm-denoising-steps 32: DLLM denoising steps
# --max-new-tokens 2048: Maximum number of new tokens to generate
# --repeat-block-detect: Enable repeat block detection
# --repeat-block-window 32: Window size for repeat block detection

COMMON_PARAMS="--eager-mode \
    --backend pytorch \
    --skip-tokenize \
    --skip-detokenize \
    --num-prompts 5000 \
    --use-uvloop \
    --dllm-block-length 32 \
    --dllm-denoising-steps 32 \
    --max-new-tokens 2048 \
    --repeat-block-detect \
    --repeat-block-window 32"

# ============================================
# Process the specified model
# ============================================
# Extract model name from path for output file naming
MODEL_NAME=$(basename ${MODEL})

echo "========================================="
echo "Processing model: ${MODEL}"
echo "========================================="
echo ""

# Extract dataset name from path for output file naming
DATASET_NAME=$(basename ${DATASET})

echo "========================================="
echo "Processing dataset: ${DATASET} with model: ${MODEL_NAME}"
echo "========================================="
echo ""

# ============================================
# Run Base Experiment for this dataset and model
# ============================================
echo "========================================="
echo "Starting BASE EXPERIMENT for ${DATASET_NAME} - ${MODEL_NAME}"
echo "========================================="

# Base experiment specific parameters
# --dllm-confidence-threshold 0.9: Higher confidence threshold for base experiment
# Note: delayed cache and focus are disabled by default (no flags)

BASE_PARAMS="${COMMON_PARAMS} \
    --dllm-confidence-threshold 0.9"

for BATCH_SIZE in "${BATCH_SIZES[@]}"
do
    echo "========================================="
    echo "Running BASE benchmark: ${DATASET_NAME}, ${MODEL_NAME}, batch size: ${BATCH_SIZE}"
    echo "========================================="

    # Define output file paths for stdout and stderr
    OUTPUT_FILE="./results/base_${DATASET_NAME}_${MODEL_NAME}_batch_${BATCH_SIZE}.log"
    ERROR_FILE="./results/base_${DATASET_NAME}_${MODEL_NAME}_batch_${BATCH_SIZE}.err"

    # Execute the command with the current batch size
    # --concurrency parameter controls the batch size
    # Redirect stdout to OUTPUT_FILE and stderr to ERROR_FILE
    python benchmark/profile_throughput.py ${BASE_PARAMS} "${DATASET_ARGS[@]}" --concurrency ${BATCH_SIZE} ${DATASET} ${MODEL} > ${OUTPUT_FILE} 2> ${ERROR_FILE}

    # Check if the command executed successfully
    if [ $? -eq 0 ]; then
        echo "Benchmark completed successfully"
        echo "Output saved to: ${OUTPUT_FILE}"
    else
        echo "Benchmark failed"
        echo "Check error log: ${ERROR_FILE}"
    fi

    echo ""
done

echo "========================================="
echo "Completed all benchmarks for ${DATASET_NAME} - ${MODEL_NAME}"
echo "========================================="
echo ""

echo "========================================="
echo "All batch size comparisons completed!"
echo "========================================="
echo ""
echo "Results summary:"
echo "  Dataset: ${DATASET}"
echo "  Model: ${MODEL}"
echo "  Total benchmarks run: 4 (1 experiment type × 4 batch sizes)"
echo "  Base experiments: ./results/base_${DATASET_NAME}_${MODEL_NAME}_batch_*.log"
echo "========================================="
