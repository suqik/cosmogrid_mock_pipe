#!/bin/bash
#SBATCH -J sample_hod
#SBATCH -p kshcnormal
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=2
#SBATCH -o /public/home/suchen/Programs/cosmogrid_mock_pipe/fastpm_runs/job_logs/sample_hod_%j.out
#SBATCH -e /public/home/suchen/Programs/cosmogrid_mock_pipe/fastpm_runs/job_logs/sample_hod_%j.err

# MPI production run: samples HOD params for all 1000 cosmologies and
# saves fastpm_runs/cosmo_hod_pairs.json (rank 0 writes the merged file).
# Usage: sbatch submit_sampling_hod.sh

source /public/home/suchen/miniforge3/bin/activate mock_pipe_env

export PATH=/opt/hpc/software/mpi/hpcx/v2.11.0/gcc-7.3.1/bin:$PATH
export LD_LIBRARY_PATH=/opt/hpc/software/mpi/hpcx/v2.11.0/gcc-7.3.1/lib:/opt/hpc/software/mpi/hwloc/lib:$LD_LIBRARY_PATH

cd /public/home/suchen/Programs/cosmogrid_mock_pipe || exit 1

mpirun -np ${SLURM_NTASKS} python fastpm_runs/run_sampling_hod.py
