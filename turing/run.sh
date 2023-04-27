#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem 16GB
#SBATCH -t 24:00:00

srun -l python3 main.py resnet transformer
