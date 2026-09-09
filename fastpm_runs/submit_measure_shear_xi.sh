#!/bin/bash
#SBATCH -J measure_shear_xi
#SBATCH -p kshcnormal
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH -o /public/home/suchen/Programs/cosmogrid_mock_pipe/fastpm_runs/job_logs/measure_shear_xi_%j.out
#SBATCH -e /public/home/suchen/Programs/cosmogrid_mock_pipe/fastpm_runs/job_logs/measure_shear_xi_%j.err

# Measure xi+/xi- (signal only, no variance) for all 1000 shape catalogs.
# Output: fastpm_runs/shear_xi_all.npz

source /public/home/suchen/miniforge3/bin/activate mock_pipe_env

export PATH=/opt/hpc/software/mpi/hpcx/v2.11.0/gcc-7.3.1/bin:$PATH
export LD_LIBRARY_PATH=/opt/hpc/software/mpi/hpcx/v2.11.0/gcc-7.3.1/lib:/opt/hpc/software/mpi/hwloc/lib:${LD_LIBRARY_PATH}

cd /public/home/suchen/Programs/cosmogrid_mock_pipe || exit 1

mpirun -np ${SLURM_NTASKS} python fastpm_runs/measure_shear_xi_all.py
