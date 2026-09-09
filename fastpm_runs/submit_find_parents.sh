#!/bin/bash
#SBATCH -J find_parents
#SBATCH -p kshcnormal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --exclude=a04r3n05
#SBATCH -o /public/home/suchen/Programs/cosmogrid_mock_pipe/fastpm_runs/job_logs/find_parents_%A_%a.out
#SBATCH -e /public/home/suchen/Programs/cosmogrid_mock_pipe/fastpm_runs/job_logs/find_parents_%A_%a.err

# Usage: sbatch submit_find_parents.sh <cosmo_index>
#   e.g. sbatch submit_find_parents.sh 0
# Or as an array over cosmologies:
#   sbatch --array=0-999 submit_find_parents.sh

IDX=${1:-$SLURM_ARRAY_TASK_ID}

BASE=/public/share/ace66so15x/suchen/FastPM/Cosmology/L1000_N1024_1000cosmo
COSMO_DIR=${BASE}/cosmo${IDX}/a_0.7692/rstar
FIND_PARENTS=/public/home/suchen/applications/rockstar/util/find_parents

cd ${COSMO_DIR} || { echo "ERROR: cannot cd to ${COSMO_DIR}"; exit 1; }

echo "=== $(date) start find_parents for cosmo${IDX} ==="
echo "input:  $(pwd)/out_0.list"
echo "output: $(pwd)/out_0_wsub.list"

${FIND_PARENTS} out_0.list 1000 > out_0_wsub.list
status=$?

echo "=== $(date) finished with status ${status} ==="
echo "lines in  out_0.list      : $(wc -l < out_0.list)"
echo "lines in  out_0_wsub.list : $(wc -l < out_0_wsub.list)"
echo "host halos (PID=-1)       : $(awk '!/^#/ && $42==-1' out_0_wsub.list | wc -l)"
echo "subhalos  (PID>=0)        : $(awk '!/^#/ && $42>=0' out_0_wsub.list | wc -l)"

# Validate the output, then remove the original halo file to save storage.
nin=$(grep -vc '^#' out_0.list)
nout=$(grep -vc '^#' out_0_wsub.list)
if [ ${status} -eq 0 ] && [ -s out_0_wsub.list ] && [ "${nin}" -eq "${nout}" ]; then
    rm out_0.list
    echo "validated (${nout} halos) -> removed original out_0.list"
else
    echo "ERROR: output invalid (status=${status}, input halos=${nin}, output halos=${nout}); keeping out_0.list" >&2
    exit 1
fi

exit 0
