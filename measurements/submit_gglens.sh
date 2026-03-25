#!/bin/bash
#SBATCH -J run_ggl
#SBATCH -p master
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --hint=nomultithread
#SBATCH -o job_logs/%J.out
#SBATCH -e job_logs/%J.err

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

pixi run python measurements/run_gglens.py