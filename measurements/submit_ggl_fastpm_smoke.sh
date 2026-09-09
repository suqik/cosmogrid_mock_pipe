#!/bin/bash
#SBATCH -J ggl_smoke
#SBATCH -p kshcnormal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -o /public/home/suchen/Programs/cosmogrid_mock_pipe/measurements/job_logs3/smoke_%J.out
#SBATCH -e /public/home/suchen/Programs/cosmogrid_mock_pipe/measurements/job_logs3/smoke_%J.err

cd /public/home/suchen/Programs/cosmogrid_mock_pipe/measurements

source /public/home/suchen/miniforge3/bin/activate mock_pipe_env

python run_ggl_fastpm.py test 1 && \
python run_ggl_fastpm.py test 0
