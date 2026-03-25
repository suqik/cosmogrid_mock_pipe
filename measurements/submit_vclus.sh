#!/bin/bash
#SBATCH -J mock_pipe
#SBATCH -p master
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=14
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH -o job_logs/%J.out
#SBATCH -e job_logs/%J.err

export OMP_NUM_THREADS=1
partname=ah
folder="/data2/suchen/CosmoGrid/fiducial_suits"
filelist=$folder/void_filelist_part_$partname
count=0

start=$(date +%s)

while read -r file; do
    file_abspath=$folder/Void/$file
    if [ -f "$file_abspath" ]; then
        echo "Processing $file_abspath"
        # filename=$(basename "$file_abspath")
        name="${file%.*}"
        output=$(pixi run python get_clustering.py $file_abspath)
        OMEGA_M=$(echo $output | awk '{print $1}')
        EOS_W=$(echo $output | awk '{print $2}')
        RMIN=$(echo $output | awk '{print $3}')
        RMAX=$(echo $output | awk '{print $4}')

        mpirun -np 14 --map-by core --bind-to core /home/suchen/applications/FCFC_v1_1_0_mpi/FCFC_2PT \
        -c cfgs/fcfc_2pt.conf \
        --select "[\${Rv} > $RMIN && \${Rv} < $RMAX, \${Rv} > $RMIN && \${Rv} < $RMAX]" \
        --omega-m $OMEGA_M \
        --eos-w $EOS_W \
        --pair-output "[tmp/pair_$partname.dd, tmp/pair_$partname.dr, tmp/pair_$partname.rr]"\
        --cf-output "tmp/2pcf_$partname.dat"\
        --mp-output results/vcluster/fiducial/$name.dat\
        < /dev/null

        # count=$((count + 1))
        # if [ $count -ge 5 ]; then
        #     break
        # fi

    fi
done < "$filelist"

end=$(date +%s)
echo "Elapsed time: $(( end - start )) seconds"