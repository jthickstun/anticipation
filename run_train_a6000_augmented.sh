#!/bin/bash
#SBATCH -p thickstun --gres=gpu:nvidia_rtx_6000_ada_generation:4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=36GB
#SBATCH -t 120:00:00
#SBATCH -J train_ar_local_midi_v2_augmented
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

PYTHONPATH=. torchrun --standalone --nproc_per_node=4 train/v2/training.py \
    --output_dir output/checkpoints \
    --data_dir data/tokenized_datasets/lmd_full/97ff64f775f1dc81e16b02fa9f8813d1 \
    --gpus_per_node 4 \
    --train_batch_size 256 \
    --gradient_accumulation_steps 2 \
    --use_wandb
