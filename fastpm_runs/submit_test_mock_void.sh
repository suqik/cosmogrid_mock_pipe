#!/bin/bash
#SBATCH -J test_mock_void
#SBATCH -p kshcnormal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -o /public/home/suchen/Programs/cosmogrid_mock_pipe/fastpm_runs/job_logs/test_mock_void_%j.out
#SBATCH -e /public/home/suchen/Programs/cosmogrid_mock_pipe/fastpm_runs/job_logs/test_mock_void_%j.err

# Usage: sbatch submit_test_mock_void.sh <cosmo_index> [ihod]

source /public/home/suchen/miniforge3/bin/activate mock_pipe_env

# DIVE links against conda's libmpfr.so.6 and a newer libstdc++ than the
# system provides; make both visible to the DIVE subprocess.
export LD_LIBRARY_PATH=/public/home/suchen/miniforge3/envs/mock_pipe_env/lib:${LD_LIBRARY_PATH}

cd /public/home/suchen/Programs/cosmogrid_mock_pipe || exit 1

python fastpm_runs/test_gen_mock_void.py ${1:-1} ${2:-0}
