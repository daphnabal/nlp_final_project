#! /bin/sh

#SBATCH --job-name=nlp_generate
#SBATCH --output=/specific/a/home/cc/students/csguests/ishaiaric/nlp_final_project/generate.out
#SBATCH --error=/specific/a/home/cc/students/csguests/ishaiaric/nlp_final_project/generate.err
#SBATCH --partition=studentkillable
#SBATCH --time=240
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1

cd /specific/a/home/cc/students/csguests/ishaiaric/nlp_final_project

python scripts/run_pipeline.py generate
