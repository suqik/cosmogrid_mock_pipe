#!/bin/bash
#SBATCH -J gglens_fastpm
#SBATCH -p kshcnormal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -o /public/home/suchen/Programs/cosmogrid_mock_pipe/measurements/job_logs3/%J.out
#SBATCH -e /public/home/suchen/Programs/cosmogrid_mock_pipe/measurements/job_logs3/%J.err
#SBATCH --array=0-49

cd /public/home/suchen/Programs/cosmogrid_mock_pipe/measurements

source /public/home/suchen/miniforge3/bin/activate mock_pipe_env

python run_ggl_fastpm.py $SLURM_ARRAY_TASK_ID 1 && \
python run_ggl_fastpm.py $SLURM_ARRAY_TASK_ID 0
