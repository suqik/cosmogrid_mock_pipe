#!/bin/bash
#SBATCH -J test_mock_shape
#SBATCH -p kshcnormal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -o /public/home/suchen/Programs/cosmogrid_mock_pipe/fastpm_runs/job_logs/test_mock_shape_%j.out
#SBATCH -e /public/home/suchen/Programs/cosmogrid_mock_pipe/fastpm_runs/job_logs/test_mock_shape_%j.err

# Usage: sbatch submit_test_mock_shape.sh <cosmo_index>

source /public/home/suchen/miniforge3/bin/activate mock_pipe_env

cd /public/home/suchen/Programs/cosmogrid_mock_pipe || exit 1

python fastpm_runs/test_gen_mock_shape.py ${1:-1}
