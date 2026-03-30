#!/bin/bash
set -e

export TORCH_SHOW_CPP_STACKTRACES=1

PYTHONPATH=. torchrun --standalone --nproc_per_node=1 train/v2/training.py \
    --output_dir output/checkpoints/local_testing \
    --data_dir data/tokenized_data/5dbc372cdeae4c4fb44f447e66029461 \
    --gpus_per_node 1 \
    --train_batch_size 64 \
    --gradient_accumulation_steps 8 \
