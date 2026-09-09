#!/bin/bash
#SBATCH -J validate_shear
#SBATCH -p kshcnormal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH -o /public/home/suchen/Programs/cosmogrid_mock_pipe/fastpm_runs/job_logs/validate_shear_%j.out
#SBATCH -e /public/home/suchen/Programs/cosmogrid_mock_pipe/fastpm_runs/job_logs/validate_shear_%j.err

source /public/home/suchen/miniforge3/bin/activate mock_pipe_env

cd /public/home/suchen/Programs/cosmogrid_mock_pipe || exit 1

python fastpm_runs/validate_shape_shear.py
