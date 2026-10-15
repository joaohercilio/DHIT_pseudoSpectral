#!/bin/bash
#SBATCH --job-name=ROGALLO
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err
#SBATCH --partition=close_cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1

module purge
module load gnu13 openmpi5 fftw/3.3.10

source ../venv/bin/activate

srun --mpi=pmix --cpu-bind=cores python rogallo.py 
  
