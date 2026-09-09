#!/bin/bash
#SBATCH -J find_parents_batch
#SBATCH -p kshcnormal
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --exclude=a04r3n05
#SBATCH --time=12:00:00
#SBATCH -o /public/home/suchen/Programs/cosmogrid_mock_pipe/fastpm_runs/job_logs/find_parents_batch_%A_%a.out
#SBATCH -e /public/home/suchen/Programs/cosmogrid_mock_pipe/fastpm_runs/job_logs/find_parents_batch_%A_%a.err

# Submit 20 tasks; each task loops over 50 cosmologies:
#   sbatch --array=0-19 submit_find_parents_batch.sh
# Task i handles cosmo indices [i*NPER, (i+1)*NPER).
# Rerunning a task is safe: finished cosmologies are skipped, and a
# killed run only leaves a .tmp file that is cleaned up on resume.

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
NPER=50
BASE=/public/share/ace66so15x/suchen/FastPM/Cosmology/L1000_N1024_1000cosmo
FIND_PARENTS=/public/home/suchen/applications/rockstar/util/find_parents

START=$((TASK_ID * NPER))
END=$((START + NPER - 1))
OK=0
FAIL=0
FAILED_LIST=""

run_one() {
    local idx=$1
    local dir=${BASE}/cosmo${idx}/a_0.7692/rstar
    [ -d "${dir}" ] || { echo "[cosmo${idx}] ERROR: dir not found: ${dir}"; return 1; }
    cd "${dir}" || { echo "[cosmo${idx}] ERROR: cannot cd ${dir}"; return 1; }

    # Already processed (resumable)?
    if [ -f out_0_wsub.list ]; then
        if [ -f out_0.list ]; then
            # Output exists but the original was never removed: verify, then clean up.
            local nin nout
            nin=$(grep -vc '^#' out_0.list)
            nout=$(grep -vc '^#' out_0_wsub.list)
            if [ "${nin}" -eq "${nout}" ]; then
                rm out_0.list
                echo "[cosmo${idx}] $(date +%T) already done -> removed leftover out_0.list"
            else
                echo "[cosmo${idx}] WARNING: out_0_wsub.list exists but counts mismatch (${nout} vs ${nin}); keeping out_0.list"
            fi
        else
            echo "[cosmo${idx}] $(date +%T) already done -> skip"
        fi
        return 0
    fi
    [ -f out_0.list ] || { echo "[cosmo${idx}] ERROR: no out_0.list"; return 1; }

    rm -f out_0_wsub.list.tmp
    echo "[cosmo${idx}] $(date +%T) start (input halos: $(grep -vc '^#' out_0.list))"
    if ! ${FIND_PARENTS} out_0.list 1000 > out_0_wsub.list.tmp; then
        echo "[cosmo${idx}] ERROR: find_parents failed (exit $?)"
        rm -f out_0_wsub.list.tmp
        return 1
    fi
    local nin nout
    nin=$(grep -vc '^#' out_0.list)
    nout=$(grep -vc '^#' out_0_wsub.list.tmp)
    if [ -s out_0_wsub.list.tmp ] && [ "${nin}" -eq "${nout}" ]; then
        mv out_0_wsub.list.tmp out_0_wsub.list
        rm out_0.list
        echo "[cosmo${idx}] $(date +%T) done -> out_0_wsub.list (${nout} halos), original removed"
    else
        echo "[cosmo${idx}] ERROR: output invalid (input halos=${nin}, output halos=${nout}); keeping out_0.list"
        rm -f out_0_wsub.list.tmp
        return 1
    fi
}

for idx in $(seq ${START} ${END}); do
    if run_one ${idx}; then
        OK=$((OK + 1))
    else
        FAIL=$((FAIL + 1))
        FAILED_LIST="${FAILED_LIST} ${idx}"
    fi
done

echo "=== $(date) task ${TASK_ID} summary: OK=${OK} FAIL=${FAIL} ==="
[ ${FAIL} -eq 0 ] || echo "failed cosmologies:${FAILED_LIST}"
[ ${FAIL} -eq 0 ]
