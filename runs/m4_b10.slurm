#!/bin/bash
#SBATCH --job-name=m4_b10
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --error=m4_b10.err
#SBATCH --output=m4_b10.out
#SBATCH --partition=longrun
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:4

cd ..

module load anaconda/3
module load cuda/11.4

conda run python3 pred_exec.py m4_b10
