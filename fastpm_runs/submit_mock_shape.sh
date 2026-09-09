#!/bin/bash
#SBATCH -J mock_shape
#SBATCH -p kshcnormal
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH -o /public/home/suchen/Programs/cosmogrid_mock_pipe/fastpm_runs/job_logs/mock_shape_%j.out
#SBATCH -e /public/home/suchen/Programs/cosmogrid_mock_pipe/fastpm_runs/job_logs/mock_shape_%j.err

# MPI production run: generates KiDS1000-North-like shape catalogs (tomo3/4/5,
# shape noise + photo-z error) for all 1000 cosmologies.
# Output: /public/share/ace66so15x/suchen/FastPM/MockCatalogs/Shapes/*.fits
# Usage: sbatch submit_mock_shape.sh

source /public/home/suchen/miniforge3/bin/activate mock_pipe_env

export PATH=/opt/hpc/software/mpi/hpcx/v2.11.0/gcc-7.3.1/bin:$PATH
export LD_LIBRARY_PATH=/opt/hpc/software/mpi/hpcx/v2.11.0/gcc-7.3.1/lib:/opt/hpc/software/mpi/hwloc/lib:$LD_LIBRARY_PATH

cd /public/home/suchen/Programs/cosmogrid_mock_pipe || exit 1

mpirun -np ${SLURM_NTASKS} python fastpm_runs/run_mock_shape.py
