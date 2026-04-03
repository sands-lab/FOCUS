#!/bin/bash

# Copyright (c) 2026 SANDS Lab. All rights reserved.

# Script to run throughput benchmark with different block sizes for SDAR models
# This script executes the profile_throughput.py benchmark for multiple configurations:
# - Dataset: Specified as command line argument
# - 2 block sizes: 16 and 64 (corresponding to SDAR-8B-Chat-b{16,64} models)
#   Note: Block size 32 is covered in run_focus_throughput_evaluation.sh
# - 2 experiment types: Focus (threshold 0.8) and Base (threshold 0.9)
# - 4 batch sizes: 32, 64, 128, and 256
# Total per dataset: 2 block sizes × 2 experiment types × 4 batch sizes = 16 benchmarks
# Output is saved to ./results directory
#
# Usage: ./run_block_size_comparison.sh <dataset_id>
# Example: ./run_block_size_comparison.sh anon8231489123/ShareGPT_Vicuna_unfiltered
#          ./run_block_size_comparison.sh allenai/WildChat
#          ./run_block_size_comparison.sh openai/gsm8k

# Check if dataset argument is provided
if [ -z "$1" ]; then
    echo "Error: Dataset ID is required as a command line argument"
    echo "Usage: $0 <dataset_id>"
    echo "Example: $0 anon8231489123/ShareGPT_Vicuna_unfiltered"
    exit 1
fi

# Get dataset from command line argument
DATASET=$1

# Dataset-specific arguments
DATASET_ARGS=()
if [[ "${DATASET}" == *"hendrycks-MATH"* ]]; then
    DATASET_ARGS+=(--dataset-format math)
elif [[ "${DATASET}" == "openai/gsm8k" ]]; then
    DATASET_ARGS+=(--dataset-format gsm8k --hf-split test --hf-config main)
fi

# Create results directory if it doesn't exist
mkdir -p ./results

# Array of block sizes to test (also determines model variant and parameters)
# Note: Block size 32 is excluded as it's covered in run_focus_throughput_evaluation.sh
BLOCK_SIZES=(16 64)

# Array of batch sizes to test
BATCH_SIZES=(32 64 128 256)

# ============================================
# Common parameters for all experiments
# ============================================
# --eager-mode: Use eager execution mode
# --backend pytorch: Use PyTorch as the backend
# --skip-tokenize: Skip tokenization step
# --skip-detokenize: Skip detokenization step
# --num-prompts 5000: Number of prompts to process
# --use-uvloop: Use uvloop for async operations
# --max-new-tokens 2048: Maximum number of new tokens to generate
# --repeat-block-detect: Enable repeat block detection
# --repeat-block-window 32: Window size for repeat block detection

COMMON_PARAMS="--eager-mode \
    --backend pytorch \
    --skip-tokenize \
    --skip-detokenize \
    --num-prompts 5000 \
    --use-uvloop \
    --max-new-tokens 2048 \
    --repeat-block-detect \
    --repeat-block-window 32"

# ============================================
# Loop through each block size
# ============================================
for BLOCK_SIZE in "${BLOCK_SIZES[@]}"
do
    # Construct model name based on block size
    MODEL="JetLM/SDAR-8B-Chat-b${BLOCK_SIZE}"
    
    echo "========================================="
    echo "Processing block size: ${BLOCK_SIZE} with model: ${MODEL}"
    echo "========================================="
    echo ""
    
    # Extract dataset name from path for output file naming
    DATASET_NAME=$(basename ${DATASET})
    
    # ============================================
    # Run Focus Experiment for this block size
    # ============================================
    echo "========================================="
    echo "Starting FOCUS EXPERIMENT for block size: ${BLOCK_SIZE}"
    echo "========================================="
    
    # Focus experiment specific parameters
    # --dllm-block-length: DLLM block length (varies with block size)
    # --dllm-denoising-steps: DLLM denoising steps (varies with block size)
    # --dllm-confidence-threshold 0.8: DLLM confidence threshold
    # --dllm-enable-delayed-cache: Enable DLLM delayed cache
    # --dllm-enable-focus: Enable DLLM focus mechanism
    # --dllm-focus-alpha 1.5: DLLM focus alpha parameter
    
    FOCUS_PARAMS="${COMMON_PARAMS} \
        --dllm-block-length ${BLOCK_SIZE} \
        --dllm-denoising-steps ${BLOCK_SIZE} \
        --dllm-confidence-threshold 0.8 \
        --dllm-enable-delayed-cache \
        --dllm-enable-focus \
        --dllm-focus-alpha 1.5"
    
    for BATCH_SIZE in "${BATCH_SIZES[@]}"
    do
        echo "========================================="
        echo "Running FOCUS benchmark: block ${BLOCK_SIZE}, ${DATASET_NAME}, batch size: ${BATCH_SIZE}"
        echo "========================================="
        
        # Define output file paths for stdout and stderr
        OUTPUT_FILE="./results/focus_${DATASET_NAME}_SDAR-b${BLOCK_SIZE}_batch_${BATCH_SIZE}.log"
        ERROR_FILE="./results/focus_${DATASET_NAME}_SDAR-b${BLOCK_SIZE}_batch_${BATCH_SIZE}.err"
        
        # Execute the command with the current batch size
        # --concurrency parameter controls the batch size
        # Redirect stdout to OUTPUT_FILE and stderr to ERROR_FILE
        python benchmark/profile_throughput.py ${FOCUS_PARAMS} "${DATASET_ARGS[@]}" --concurrency ${BATCH_SIZE} ${DATASET} ${MODEL} > ${OUTPUT_FILE} 2> ${ERROR_FILE}
        
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
    
    # ============================================
    # Run Base Experiment for this block size
    # ============================================
    echo "========================================="
    echo "Starting BASE EXPERIMENT for block size: ${BLOCK_SIZE}"
    echo "========================================="
    
    # Base experiment specific parameters
    # --dllm-block-length: DLLM block length (varies with block size)
    # --dllm-denoising-steps: DLLM denoising steps (varies with block size)
    # --dllm-confidence-threshold 0.9: Higher confidence threshold for base experiment
    # Note: delayed cache and focus are disabled by default (no flags)
    
    BASE_PARAMS="${COMMON_PARAMS} \
        --dllm-block-length ${BLOCK_SIZE} \
        --dllm-denoising-steps ${BLOCK_SIZE} \
        --dllm-confidence-threshold 0.9"
    
    for BATCH_SIZE in "${BATCH_SIZES[@]}"
    do
        echo "========================================="
        echo "Running BASE benchmark: block ${BLOCK_SIZE}, ${DATASET_NAME}, batch size: ${BATCH_SIZE}"
        echo "========================================="
        
        # Define output file paths for stdout and stderr
        OUTPUT_FILE="./results/base_${DATASET_NAME}_SDAR-b${BLOCK_SIZE}_batch_${BATCH_SIZE}.log"
        ERROR_FILE="./results/base_${DATASET_NAME}_SDAR-b${BLOCK_SIZE}_batch_${BATCH_SIZE}.err"
        
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
    echo "Completed all benchmarks for block size: ${BLOCK_SIZE}"
    echo "========================================="
    echo ""
done

echo "========================================="
echo "All block size comparisons completed!"
echo "========================================="
echo ""
echo "Results summary:"
echo "  Dataset: ${DATASET}"
echo "  Total benchmarks run: 16 (2 block sizes × 2 experiment types × 4 batch sizes)"
echo "  Block sizes tested: 16, 64 (block size 32 is in run_focus_throughput_evaluation.sh)"
echo "  Focus experiments: ./results/focus_${DATASET_NAME}_SDAR-b*_batch_*.log"
echo "  Base experiments: ./results/base_${DATASET_NAME}_SDAR-b*_batch_*.log"
echo "========================================="
