import os

template = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --error={job_name}.err
#SBATCH --output={job_name}.out
#SBATCH --partition=longrun
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:4

cd ..

module load anaconda/3
module load cuda/11.4

conda run python3 pred_exec.py {job_name}
"""

for filename in os.listdir('jobs'):
    if filename.endswith('.json'):
        job_name = os.path.splitext(filename)[0]
        with open(f'{job_name}.slurm', 'w') as f:
            f.write(template.format(job_name=job_name))