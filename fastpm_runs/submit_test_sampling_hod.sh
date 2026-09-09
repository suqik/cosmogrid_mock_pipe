#!/bin/bash
#SBATCH -J test_sampling_hod
#SBATCH -p kshcnormal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -o /public/home/suchen/Programs/cosmogrid_mock_pipe/fastpm_runs/job_logs/test_sampling_hod_%j.out
#SBATCH -e /public/home/suchen/Programs/cosmogrid_mock_pipe/fastpm_runs/job_logs/test_sampling_hod_%j.err

# Usage: sbatch submit_test_sampling_hod.sh <cosmo_index>

IDX=${1:-1}

source /public/home/suchen/miniforge3/bin/activate mock_pipe_env

cd /public/home/suchen/Programs/cosmogrid_mock_pipe || exit 1

python fastpm_runs/test_sampling_hod.py ${IDX}
