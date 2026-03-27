#!/bin/bash
# =============================================================================
# generate_final.slurm
#
# Generates all 9,900 stories for the final experiment:
#   30 prompts × 30 shadows × 11 schedules
#   (7 fixed-temperature baselines + 4 named dynamic schedules)
#
# Output:  results_final/Qwen_Qwen2.5-1.5B-Instruct_<schedule>/stories.jsonl
#
# Estimated runtime: ~8-10 h on a single A100/RTX3090 (1.5B model, fp16).
# Requesting 12 h as a safe upper bound.
# =============================================================================

#SBATCH --job-name=gen_final
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/ishaiaric/gen_final.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/ishaiaric/gen_final.err
#SBATCH --partition=studentkillable
#SBATCH --time=720
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1

cd /specific/a/home/cc/students/csguests/ishaiaric/nlp_final_project
export PATH="/vol/joberant_nobck/data/NLP_368307701_2526a/ishaiaric/anaconda3/envs/nlp_env/bin:$PATH"
export HF_HOME=/vol/joberant_nobck/data/NLP_368307701_2526a/ishaiaric/.cache/huggingface
source /vol/joberant_nobck/data/NLP_368307701_2526a/ishaiaric/anaconda3/etc/profile.d/conda.sh
conda activate nlp_env

which python
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"

# Run all 11 schedules with 30 shadow stories per prompt.
# Schedules:
#   fixed_temperature_{0.01, 0.334, 0.667, 1.0, 1.334, 1.667, 2.0}  (7 baselines)
#   increasing, decreasing, valley, peak                              (4 dynamic)
python scripts/run_final_experiment.py \
    --model Qwen/Qwen2.5-1.5B-Instruct
