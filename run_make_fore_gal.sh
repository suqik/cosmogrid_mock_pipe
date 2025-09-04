#!/bin/bash
#SBATCH -J make_fore_gal
#SBATCH -p master
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH -o %J.out
#SBATCH -e %J.err

export OMP_NUM_THREADS=1

pixi run mpirun -np 32 --map-by core --bind-to core python make_fore_gal.py
