#!/bin/bash
#SBATCH -J make_fore_void
#SBATCH -p master
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH -o %J.out
#SBATCH -e %J.err

export OMP_NUM_THREADS=1

pixi run python make_fore_void.py
