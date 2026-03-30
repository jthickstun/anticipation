#!/bin/bash
#SBATCH -p genai-thickstun-highpri --gres=gpu:2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH -t 48:00:00
#SBATCH -J train_model
#SBATCH -e output/slurm_logs/%j/stderr.err
#SBATCH -o output/slurm_logs/%j/stdout.out
set -e

# --- set up conda and activate it ---
# assuming conda binary lives here
CONDA_ACTIVATE_PATH="/share/apps/software/anaconda3/etc/profile.d/conda.sh"
if source "$CONDA_ACTIVATE_PATH" 2>/dev/null; then
  cd /home/mf867/anticipation_isolated/anticipation
  conda activate ./env
  echo "activated environment."
else
  echo "conda startup script not found."
fi


#export TORCH_SHOW_CPP_STACKTRACES=1

PYTHONPATH=. torchrun --standalone --nproc_per_node=2 train/v2/training.py \
    --output_dir "output/slurm_logs/${SLURM_JOB_ID}/checkpoints" \
    --data_dir data/tokenized_datasets/giga_midi/6fb2094dfa7c0d16278dfaa4a401e3b8 \
    --gpus_per_node 2 \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --gradient_accumulation_steps 4 \
    --steps_per_eval 1000 \
    --steps_per_checkpoint 20000 \
    --save_midi_output_after_step 1000000 \
    --num_events_to_generate_for_midi_inference 80 \
    --warmup_percent 0.01 \
    --num_layers 8 \
    --hidden_dim 768 \
    --intermediate_dim 3072 \
    --num_heads 8 \
    --no_weight_tie \
    --window_pattern "L" \
    --pos_emb "rope" \
    --learning_rate 1e-03 \
    --flops "2e20" \
    --mlp_style "Llama" \
    --use_wandb
