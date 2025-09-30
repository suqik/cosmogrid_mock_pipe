#!/bin/bash
#SBATCH -J mock_pipe
#SBATCH -p master
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH -o job_logs/%J.out
#SBATCH -e job_logs/%J.err

export OMP_NUM_THREADS=1

# pixi run mpirun -np 32 --map-by core --bind-to core python make_back_gal.py ### make background galaxies 
# pixi run mpirun -np 32 --map-by core --bind-to core python make_fore_gal.py ### make foreground galaxies
pixi run mpirun -np 32 --map-by core --bind-to core python make_fore_void.py ### make foreground voids
# pixi run mpirun -np 32 --map-by core --bind-to core python make_void_random.py ### make random voids
# pixi run python run_dsigma.py
