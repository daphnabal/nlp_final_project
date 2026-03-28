#!/bin/bash
# =============================================================================
# generate_final.slurm
#
# Generates stories for the final experiment for ONE schedule at a time.
# Submit one job per schedule to run all 11 in parallel:
#
#   for SCHED in fixed_temperature_0.01 fixed_temperature_0.334 \
#                fixed_temperature_0.667 fixed_temperature_1.0   \
#                fixed_temperature_1.334 fixed_temperature_1.667 \
#                fixed_temperature_2.0 increasing decreasing valley peak; do
#       sbatch --job-name=gen_${SCHED} generate_final.slurm $SCHED
#   done
#
# Resume-safe: if stories.jsonl already has partial results (e.g. from a
# previous killed run), completed (prompt_id, shadow_id) pairs are skipped
# automatically.
#
# Output:  results_final/Qwen_Qwen2.5-1.5B-Instruct_<schedule>/stories.jsonl
# Per-schedule runtime: ~4 h (900 stories × ~16 s/story on RTX3090).
# =============================================================================

#SBATCH --job-name=gen_final
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/ishaiaric/gen_final_%j.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/ishaiaric/gen_final_%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=480
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1

# First positional argument = schedule name (required)
SCHEDULE=${1:?"Usage: sbatch generate_final.slurm <schedule_name>"}

cd /vol/joberant_nobck/data/NLP_368307701_2526a/ishaiaric/nlp_final_project
export PATH="/vol/joberant_nobck/data/NLP_368307701_2526a/ishaiaric/anaconda3/envs/nlp_env/bin:$PATH"
export HF_HOME=/vol/joberant_nobck/data/NLP_368307701_2526a/ishaiaric/.cache/huggingface
source /vol/joberant_nobck/data/NLP_368307701_2526a/ishaiaric/anaconda3/etc/profile.d/conda.sh
conda activate nlp_env

which python
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
echo "Running schedule: $SCHEDULE"

python scripts/run_final_experiment.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --schedules $SCHEDULE
