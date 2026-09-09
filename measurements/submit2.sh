#!/bin/bash
#SBATCH -J gglens_measure
#SBATCH -p kshcnormal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -o job_logs2/%J.out
#SBATCH -e job_logs2/%J.err
#SBATCH --array=0-59

source /public/home/suchen/miniforge3/bin/activate mock_pipe_env

python run_ggl.py $SLURM_ARRAY_TASK_ID
