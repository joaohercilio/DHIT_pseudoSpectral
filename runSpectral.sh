#!/bin/bash
#SBATCH --job-name=spectral_DNS
#SBATCH -o slurm.%N.%j.out
#SBATCH -e slurm.%N.%j.err
#SBATCH --partition=close_cpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1

module purge
module load gnu13 openmpi5 fftw/3.3.10 phdf5

source ../venv/bin/activate

export OMPI_MCA_mtl=^ofi
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1


srun --mpi=pmix --cpu-bind=cores python spectral.py  