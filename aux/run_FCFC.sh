#!/bin/bash
#SBATCH -J FCFC
#SBATCH -p master
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -o %J.out
#SBATCH -e %J.err
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=1

source ~/miniforge3/bin/activate simtk-env

# mpirun -np 32 ~/applications/FCFC_mpi/FCFC_2PT_BOX -c fcfc_2pt_box.conf
mpirun -np 32  --map-by core --bind-to core ~/applications/FCFC_mpi/FCFC_2PT -c fcfc_2pt.conf
