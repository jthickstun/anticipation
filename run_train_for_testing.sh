set -e

#PYTHONPATH=. torchrun --standalone --nproc_per_node=1 train/v2/training.py \
#    --output_dir output/checkpoints/test_checkpoints \
#    --data_dir data/tokenized_datasets/lmd_full/b82a7a2750e3c5836ffb9bf564720cd8 \
#    --gpus_per_node 1 \
#    --train_batch_size 128 \
#    --gradient_accumulation_steps 1 \
#    --steps_per_eval 100 \
#    --use_wandb

PYTHONPATH=. python train/v2/training.py \
    --output_dir output/checkpoints/test_checkpoints \
    --data_dir data/tokenized_datasets/lmd_full/b82a7a2750e3c5836ffb9bf564720cd8 \
    --num_layers 1 \
    --gpus_per_node 1 \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --num_heads 8 \
    --hidden_dim 64 \
    --gradient_accumulation_steps 1 \
    --steps_per_eval 50 \
    --save_midi_output_after_step 25 \
    --num_events_to_generate_for_midi_inference 80 \
    --use_wandb \
    --no_ddp