'''
Script to choose a best LOS that has the least repetitive objects.
'''
import numpy as np
import healpy as hp
from scipy.spatial.transform import Rotation as R
import pymangle
from loguru import logger

from mpi4py import MPI

def get_repeat_ratio(galcone_id):
    repeat_ratio = len(np.unique(galcone_id))/len(galcone_id)
    return 1-repeat_ratio

def process_routine(galcone, m, rank, num=10, verbose=False):
    positions = galcone[:,:-1]
    galcone_id = galcone[:,-1]

    rotators = R.random(num=num, rng=np.random.default_rng(rank))
    repeat_ratios = []
    euler_angles = rotators.as_euler('zyx', degrees=True)

    for inum, irot in enumerate(rotators):
        if verbose:
            if rank == 0:
                print(f"Try rotation {inum}", flush=True)
        positions_rot = irot.apply(positions)
        galcone_ra_rot, galcone_dec_rot = hp.vec2ang(positions_rot, lonlat=True)
        
        mask = m.contains(galcone_ra_rot, galcone_dec_rot)
        weights = m.weight(galcone_ra_rot[mask], galcone_dec_rot[mask])
        select = (weights > 0)

        repeat_ratio = get_repeat_ratio(galcone_id[mask][select])
        repeat_ratios.append(repeat_ratio)

    return repeat_ratios, euler_angles

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        logger.info("Load data.")

    galcone = np.load("aux/galcone.npy")
    m = pymangle.Mangle("catalogs/masks/mask_DR12v5_CMASSLOWZTOT_North.ply")

    if rank == 0:
        logger.info("Data processing.")
    local_repeat_ratio, local_euler_angles = process_routine(galcone, m, rank, num=100, verbose=True)
    comm.Barrier()

    if rank == 0:
        logger.info("Gathering data.")
    ### gathering local_repeat_ratio and local_euler_angles
    repeat_ratio = comm.gather(local_repeat_ratio, root=0)
    euler_angles = comm.gather(local_euler_angles, root=0)

    if rank == 0:
        logger.info("Saving to file.")
        np.savetxt("aux/tried_LOS_repeat_ratio.txt", np.c_[np.concatenate(euler_angles), np.concatenate(repeat_ratio)], fmt="%.3f %.3f %.3f %.5f")