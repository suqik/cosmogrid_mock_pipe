#!/bin/bash
#SBATCH -J mock_void
#SBATCH -p kshcnormal
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH -o /public/home/suchen/Programs/cosmogrid_mock_pipe/fastpm_runs/job_logs/mock_void_%j.out
#SBATCH -e /public/home/suchen/Programs/cosmogrid_mock_pipe/fastpm_runs/job_logs/mock_void_%j.err

# MPI production run: generates void catalogs from the mock galaxy catalogs
# (1000 cosmologies x 10 HODs) using DIVE.
# Output: /public/share/ace66so15x/suchen/FastPM/MockCatalogs/Voids/*.fits
# Usage: sbatch submit_mock_void.sh

source /public/home/suchen/miniforge3/bin/activate mock_pipe_env

# DIVE links against conda's libmpfr.so.6 and a newer libstdc++ than the
# system provides; make both visible to the DIVE subprocess.
export LD_LIBRARY_PATH=/public/home/suchen/miniforge3/envs/mock_pipe_env/lib:${LD_LIBRARY_PATH}

export PATH=/opt/hpc/software/mpi/hpcx/v2.11.0/gcc-7.3.1/bin:$PATH
export LD_LIBRARY_PATH=/opt/hpc/software/mpi/hpcx/v2.11.0/gcc-7.3.1/lib:/opt/hpc/software/mpi/hwloc/lib:${LD_LIBRARY_PATH}

cd /public/home/suchen/Programs/cosmogrid_mock_pipe || exit 1

mpirun -np ${SLURM_NTASKS} python fastpm_runs/run_mock_void.py
