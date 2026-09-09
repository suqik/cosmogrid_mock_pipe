#!/bin/bash
#SBATCH -J mock_gal
#SBATCH -p kshcnormal
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH -o /public/home/suchen/Programs/cosmogrid_mock_pipe/fastpm_runs/job_logs/mock_gal_%j.out
#SBATCH -e /public/home/suchen/Programs/cosmogrid_mock_pipe/fastpm_runs/job_logs/mock_gal_%j.err

# MPI production run: generates mock galaxy catalogs for all 1000 cosmologies
# x 10 HOD parameter sets (from fastpm_runs/cosmo_hod_pairs.json).
# Output: /public/share/ace66so15x/suchen/FastPM/MockCatalogs/Gals/*.fits
# Usage: sbatch submit_mock_gal.sh

source /public/home/suchen/miniforge3/bin/activate mock_pipe_env

export PATH=/opt/hpc/software/mpi/hpcx/v2.11.0/gcc-7.3.1/bin:$PATH
export LD_LIBRARY_PATH=/opt/hpc/software/mpi/hpcx/v2.11.0/gcc-7.3.1/lib:/opt/hpc/software/mpi/hwloc/lib:$LD_LIBRARY_PATH

cd /public/home/suchen/Programs/cosmogrid_mock_pipe || exit 1

mpirun -np ${SLURM_NTASKS} python fastpm_runs/run_mock_gal.py
