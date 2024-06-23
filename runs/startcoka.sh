#!/bin/bash
for job_file in *.slurm
do
  sbatch "$job_file"
done