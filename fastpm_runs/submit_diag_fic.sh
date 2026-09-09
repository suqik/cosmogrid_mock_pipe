#!/bin/bash
#SBATCH -J diag_fic
#SBATCH -p kshcnormal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -o /public/home/suchen/Programs/cosmogrid_mock_pipe/fastpm_runs/job_logs/diag_fic_%j.out
#SBATCH -e /public/home/suchen/Programs/cosmogrid_mock_pipe/fastpm_runs/job_logs/diag_fic_%j.err

# Usage: sbatch submit_diag_fic.sh <cosmo_index> [n_draws]

source /public/home/suchen/miniforge3/bin/activate mock_pipe_env

cd /public/home/suchen/Programs/cosmogrid_mock_pipe || exit 1

python fastpm_runs/diag_fic_dist.py ${1:-1} ${2:-2000}
