#!/bin/bash
# =============================================================================
# evaluate_final_coherence.slurm
#
# Scores one schedule's 900 stories with two OPT coherence evaluators:
#   facebook/opt-125m   → key: coherence_opt_125m
#   facebook/opt-1.3b   → key: coherence_opt_1_3b
#
# Submit one job per schedule to run all 11 in parallel:
#
#   foreach SCHED (fixed_temperature_0.01 fixed_temperature_0.334 \
#                  fixed_temperature_0.667 fixed_temperature_1.0   \
#                  fixed_temperature_1.334 fixed_temperature_1.667 \
#                  fixed_temperature_2.0 increasing decreasing valley peak)
#       sbatch --job-name=eval_${SCHED} evaluate_final_coherence.slurm $SCHED
#   end
#
# Input:   results_final/Qwen_Qwen2.5-1.5B-Instruct_<schedule>/stories.jsonl
# Output:  results_final/Qwen_Qwen2.5-1.5B-Instruct_<schedule>/stories_opt_scored.jsonl
#
# Estimated runtime per schedule: ~20 min (OPT-125M: ~3 min, OPT-1.3B: ~17 min).
# Requesting 60 min as a safe upper bound.
# =============================================================================

#SBATCH --job-name=eval_coherence
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/ishaiaric/eval_coherence_%j.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/ishaiaric/eval_coherence_%j.err
#SBATCH --partition=studentkillable
#SBATCH --time=60
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=geforce_rtx_3090

# First positional argument = schedule name (required)
SCHEDULE=${1:?"Usage: sbatch evaluate_final_coherence.slurm <schedule_name>"}

cd /vol/joberant_nobck/data/NLP_368307701_2526a/ishaiaric/nlp_final_project
export PATH="/vol/joberant_nobck/data/NLP_368307701_2526a/ishaiaric/anaconda3/envs/nlp_env/bin:$PATH"
export HF_HOME=/vol/joberant_nobck/data/NLP_368307701_2526a/ishaiaric/.cache/huggingface
source /vol/joberant_nobck/data/NLP_368307701_2526a/ishaiaric/anaconda3/etc/profile.d/conda.sh
conda activate nlp_env

which python
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
echo "Scoring schedule: $SCHEDULE"

python scripts/run_opt_coherence_final.py \
    --gen_model Qwen/Qwen2.5-1.5B-Instruct \
    --schedules $SCHEDULE
