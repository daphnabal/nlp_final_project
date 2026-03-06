#! /bin/sh

#SBATCH --job-name=nlp_evaluate
#SBATCH --output=/specific/a/home/cc/students/csguests/ishaiaric/nlp_final_project/evaluate.out
#SBATCH --error=/specific/a/home/cc/students/csguests/ishaiaric/nlp_final_project/evaluate.err
#SBATCH --partition=studentkillable
#SBATCH --time=120
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1

cd /specific/a/home/cc/students/csguests/ishaiaric/nlp_final_project

python scripts/run_pipeline.py evaluate
    