#!/bin/bash
#SBATCH --job-name=A4
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --error=A4.err
#SBATCH --output=A4.out
#SBATCH --partition=longrun
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:4

cd ..

module load anaconda/3
module load cuda/11.4

conda run python3 pred_exec.py A4
