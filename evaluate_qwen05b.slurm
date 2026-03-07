#!/bin/bash

#SBATCH --job-name=eval_qwen05b
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2526a/ishaiaric/eval_qwen05b.out
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2526a/ishaiaric/eval_qwen05b.err
#SBATCH --partition=studentkillable
#SBATCH --time=240
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
python -c "import torch; print(torch.__version__)"

# Evaluate main schedules + sweep in one pass
python scripts/run_pipeline.py evaluate \
    --model Qwen/Qwen2.5-0.5B \
    --eval_mode all \
    --schedules fixed increasing decreasing valley peak \
                sweep_0.5 sweep_0.7 sweep_1.0 sweep_1.2 sweep_1.5
