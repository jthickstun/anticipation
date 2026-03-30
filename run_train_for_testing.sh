set -e
# sh run_train_for_testing.sh

#PYTHONPATH=.  torchrun --standalone --nproc_per_node=4 train/v2/training.py \
#    --output_dir output/checkpoints/test_checkpoints \
#    --data_dir data/tokenized_datasets/lmd_full/b82a7a2750e3c5836ffb9bf564720cd8 \
#    --num_layers 12 \
#    --gpus_per_node 4 \
#    --train_batch_size 256 \
#    --eval_batch_size 256 \
#    --num_heads 12 \
#    --hidden_dim 768 \
#    --gradient_accumulation_steps 2 \
#    --steps_per_eval 1000 \
#    --save_midi_output_after_step 10000 \
#    --num_events_to_generate_for_midi_inference 80 \
#    --steps_per_checkpoint 100000 \
#    --use_wandb

# Lakh alone has 1,790,841,856 tokens for AR only
# Largest FLOP we can handle is 1e17
# FLOPs:
# 1e16
# 2.15e16
# 4.68e16
# 1e17
#
#cd /home/admin/Documents/RESEARCH/anticipation
#conda activate ./env

# DEPTHS:
# 2, 4, 6, 8, 10, 12, 14
#PYTHONPATH=. python train/v2/training.py \
#    --output_dir output/checkpoints/test_checkpoints \
#    --checkpoint_path output/checkpoints/test_checkpoints/step-20 \
#    --data_dir data/tokenized_datasets/lmd_full/52b9a7aa2d5d895d6e7d25740021e560 \
#    --gpus_per_node 1 \
#    --train_batch_size 64 \
#    --eval_batch_size 64 \
#    --gradient_accumulation_steps 1 \
#    --steps_per_eval 50 \
#    --steps_per_checkpoint 20 \
#    --num_train_steps 2000000 \
#    --save_midi_output_after_step 1000000 \
#    --num_events_to_generate_for_midi_inference 80 \
#    --num_layers 2 \
#    --no_weight_tie \
#    --window_pattern "SSSL" \
#    --pos_emb "rope" \
#    --learning_rate 1e-03 \
#    --no_ddp \
#    --no_torch_compile \
#    --no_cuda_graphs \
#    --wandb_project "dgx_testing"

PYTHONPATH=.  torchrun --standalone --nproc_per_node=1 train/v2/training.py \
    --output_dir output/checkpoints/test_checkpoints \
    --data_dir data/tokenized_datasets/giga_midi/6fb2094dfa7c0d16278dfaa4a401e3b8 \
    --gpus_per_node 1 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --steps_per_eval 2000 \
    --steps_per_checkpoint 100 \
    --save_midi_output_after_step 1000000 \
    --num_events_to_generate_for_midi_inference 80 \
    --warmup_percent 0.01 \
    --num_layers 6 \
    --hidden_dim 512 \
    --intermediate_dim 1792 \
    --num_heads 8 \
    --no_weight_tie \
    --window_pattern "L" \
    --pos_emb "rope" \
    --learning_rate 1e-03 \
    --flops "1e20" \
    --mlp_style "Llama" \
    --wandb_project "dgx_testing"
