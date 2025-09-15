#!/bin/bash
#SBATCH -J fcfc
#SBATCH -p master
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --hint=nomultithread
#SBATCH -o %J.out
#SBATCH -e %J.err

export OMP_NUM_THREADS=1

mpirun -n 32 --map-by core --bind-to core ~/applications/FCFC_v1_1_0_mpi/FCFC_2PT -c cfgs/fcfc_2pt.conf
