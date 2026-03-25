#!/bin/bash
#SBATCH -J run_angular_stats
#SBATCH -p master
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=4
#SBATCH --hint=nomultithread
#SBATCH -o job_logs/%J.out
#SBATCH -e job_logs/%J.err

export OMP_NUM_THREADS=4
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

pixi run mpirun --report-bindings --map-by slot:PE=4 --bind-to core python measurements/get_angular_stats.py