#!/bin/bash
#SBATCH -J job_example
#SBATCH -p master
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH -o %J.out
#SBATCH -e %J.err

export OMP_NUM_THREADS=1

# *** add your commands in the end of the following command *** 
mpirun -np 32 --map-by core --bind-to core ### 
