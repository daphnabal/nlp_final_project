#!/bin/bash
# =============================================================================
# evaluate_final_coherence.slurm
#
# Scores all 9,900 generated stories with two OPT coherence evaluators:
#   facebook/opt-125m   → key: coherence_opt_125m
#   facebook/opt-1.3b   → key: coherence_opt_1_3b
#
# The two models are loaded and freed sequentially; peak VRAM ≤ ~4 GB.
# An RTX 3090 (24 GB) is more than sufficient.
#
# Input:   results_final/Qwen_Qwen2.5-1.5B-Instruct_<schedule>/stories.jsonl
# Output:  results_final/Qwen_Qwen2.5-1.5B-Instruct_<schedule>/stories_opt_scored.jsonl
#
# Estimated runtime: ~3-4 h (OPT-125M: ~0.5 h, OPT-1.3B: ~2.5 h).
# Requesting 5 h as a safe upper bound.
# =============================================================================

#SBATCH --job-name=eval_final_coherence
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/ishaiaric/eval_final_coherence.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/ishaiaric/eval_final_coherence.err
#SBATCH --partition=studentkillable
#SBATCH --time=300
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --constraint=geforce_rtx_3090

cd /specific/a/home/cc/students/csguests/ishaiaric/nlp_final_project
export PATH="/vol/joberant_nobck/data/NLP_368307701_2526a/ishaiaric/anaconda3/envs/nlp_env/bin:$PATH"
export HF_HOME=/vol/joberant_nobck/data/NLP_368307701_2526a/ishaiaric/.cache/huggingface
source /vol/joberant_nobck/data/NLP_368307701_2526a/ishaiaric/anaconda3/etc/profile.d/conda.sh
conda activate nlp_env

which python
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"

# Score all 11 schedules with both OPT models.
# OPT-125M is run first (fast), then OPT-1.3B.
# Both models' scores are accumulated into stories_opt_scored.jsonl.
python scripts/run_opt_coherence_final.py \
    --gen_model Qwen/Qwen2.5-1.5B-Instruct
