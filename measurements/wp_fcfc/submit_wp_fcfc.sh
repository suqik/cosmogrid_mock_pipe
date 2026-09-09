#!/bin/bash
#SBATCH -J wp_fcfc
#SBATCH -p kshcnormal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH -o /public/home/suchen/Programs/cosmogrid_mock_pipe/measurements/wp_fcfc/job_%j.out
#SBATCH -e /public/home/suchen/Programs/cosmogrid_mock_pipe/measurements/wp_fcfc/job_%j.err

# w_p (projected 2PCF) of the FastPM LOWZ-NGC mock, cosmo_000000, 10 HODs.
# RR is computed once and shared across HODs (OVERWRITE=1 keeps pair files).

export LD_LIBRARY_PATH=/public/home/suchen/miniforge3/envs/mock_pipe_env/lib:/opt/hpc/software/mpi/hpcx/v2.11.0/gcc-7.3.1/lib:/opt/hpc/software/mpi/hwloc/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=32

cd /public/home/suchen/Programs/cosmogrid_mock_pipe/measurements/wp_fcfc || exit 1

FCFC=/public/home/suchen/applications/FCFC_1.1.0/FCFC_2PT
DATA_FMT="/public/share/ace66so15x/suchen/FastPM/MockCatalogs/Gals/cosmo_000000_realization_0000_HOD_%d_boss_lowz_north_2dflens_south.fits"
OUT=$PWD
RAND=$OUT/rand_lowz_500k.fits

for h in $(seq 0 9); do
    DATA=$(printf "$DATA_FMT" "$h")
    sed -e "s|@HOD@|$h|g" -e "s|@OUT@|$OUT|g" \
        -e "s|@DATA@|$DATA|g" -e "s|@RAND@|$RAND|g" \
        wp_fcfc.conf.tpl > wp_fcfc_hod$h.conf
    echo "===== HOD $h ====="
    $FCFC -c wp_fcfc_hod$h.conf || exit 1
done

echo "ALL DONE"
