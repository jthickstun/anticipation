#!/bin/bash
#SBATCH -p thickstun
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --mem=50GB
#SBATCH -t 05:00:00
#SBATCH -J tokenize_dataset_transcripts
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


# we run out of space if we just write to tmp
export CUSTOM_TMP_DIR=/scratch/$USER
mkdir -p "$CUSTOM_TMP_DIR"

PYTHONPATH=. python train/v2/dataset_tokenize.py --dataset_type transcripts --settings_json_name "ar_only_local_midi_no_instr_limit_settings_87451b329323d36a658ac64ed9a8bb81.json"
