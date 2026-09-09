#!/bin/bash
#SBATCH -J ggl_flips
#SBATCH -p kshcnormal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH -o /public/home/suchen/Programs/cosmogrid_mock_pipe/measurements/job_logs3/flips_%J.out
#SBATCH -e /public/home/suchen/Programs/cosmogrid_mock_pipe/measurements/job_logs3/flips_%J.err

# ggl shear-convention systematic: all 4 (flip_g1, flip_g2) combos x 2 weight modes,
# cosmo_000000 (part=test), per-HOD jackknife errors.

cd /public/home/suchen/Programs/cosmogrid_mock_pipe/measurements || exit 1

source /public/home/suchen/miniforge3/bin/activate mock_pipe_env

for f1 in 0 1; do
    for f2 in 0 1; do
        echo "===== flip_g1=$f1 flip_g2=$f2 ====="
        python run_ggl_fastpm.py test 1 $f1 $f2 || exit 1
        python run_ggl_fastpm.py test 0 $f1 $f2 || exit 1
    done
done

echo "ALL FLIP COMBOS DONE"
