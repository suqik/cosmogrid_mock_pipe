from joblib import Parallel, delayed
import numpy as np
import pyccl as ccl
import json
from loguru import logger
import h5py
import treecorr
import os
import sys
sys.path.append("/home/suchen/Program/CosmoGrid/")

from utils.io_func import *
from utils.mkfore_utils import bounded_kde_transform, resample_bounded

def my_alloc(lst, n_groups):
    """
    Separate list to n groups
    """
    length = len(lst)
    base = length // n_groups
    remainder = length % n_groups

    # 每组的大小 = base + 1 (前 remainder 组)，否则 base
    sizes = np.full(n_groups, base, dtype=int)
    sizes[:remainder] += 1

    # 计算每组的起止下标
    idx = np.cumsum(np.concatenate(([0], sizes)))
    return [lst[idx[i]: idx[i + 1]] for i in range(n_groups)]

def split_jackknife_patches(catalog_radec, centers=None, njk=120):
    cat = treecorr.Catalog(ra=catalog_radec[:, 0], dec=catalog_radec[:, 1], ra_units='deg', dec_units='deg')
    field = cat.getNField()
    if centers is not None:
        labels = field.kmeans_assign_patches(centers)
        return labels
    elif njk is not None:
        labels, centers = field.run_kmeans(njk)
        return labels, centers
    else:
        raise ValueError("Either centers or njk must be specified.")
    
def process_fore_cat(fore_cat, labels, cosmo_ccl, odir):
    hubble = cosmo_ccl.to_dict()['h']
    Ngal = len(fore_cat)
    jk_labels = np.unique(labels)

    ### calculate essential values
    if IS_VOID:
        data_dst = np.zeros((Ngal, 10), dtype=np.float32)
    else:
        data_dst = np.zeros((Ngal, 9), dtype=np.float32)
        
    data_dst[:, 0] = fore_cat['ra']
    data_dst[:, 1] = np.deg2rad(fore_cat['ra'])
    data_dst[:, 2] = fore_cat['dec']
    data_dst[:, 3] = np.deg2rad(fore_cat['dec'])
    data_dst[:, 6] = fore_cat['z']

    data_dst[:, 4] = np.cos(data_dst[:, 3])
    data_dst[:, 5] = np.sin(data_dst[:, 3])
    data_dst[:, 7] = ccl.comoving_radial_distance(cosmo_ccl, 1./(1+data_dst[:, 6])) * hubble
    # if not weights, set to 1
    data_dst[:, 8] = fore_cat['w'] # 1
    if IS_VOID:
        data_dst[:, 9] = fore_cat['Rv'] # 1

    min_src_num = 10000
    result_foreground_cat_path = f"{odir}"
    if not os.path.isdir(result_foreground_cat_path):
        os.makedirs(result_foreground_cat_path)

    expos_avail_sub = []
    expos_count = 0

    for ilabel in jk_labels:
        idx_group = ilabel == labels
        sub_data = data_dst[idx_group]

        ground_src_num = idx_group.sum()

        if ground_src_num <= min_src_num:
            expos_name = f"{ilabel:d}-0"
            expos_path = result_foreground_cat_path + "/%s.hdf5" % expos_name
            h5f_expos = h5py.File(expos_path, "w")
            h5f_expos["/data"] = sub_data
            h5f_expos.close()

            expos_avail_sub.append("%s\t%s\t%d\t%d\n"
                                    % (expos_path, expos_name, ground_src_num, ilabel))
            expos_count += 1
        else:
            m, n = divmod(ground_src_num, min_src_num)
            nums_distrib = my_alloc([1 for i in range(ground_src_num)], m)
            nums = [sum(nums_distrib[i]) for i in range(m)]
            nums_st = [sum(nums[:i]) for i in range(m)]
            for count in range(m):
                expos_name = f"{ilabel:d}-{count:d}"
                expos_path = result_foreground_cat_path + "/%s.hdf5" % expos_name
                h5f_expos = h5py.File(expos_path, "w")
                h5f_expos["/data"] = sub_data[nums_st[count]: nums_st[count] + nums[count]]
                h5f_expos.close()

                expos_avail_sub.append("%s\t%s\t%d\t%d\n"
                                        % (expos_path, expos_name, nums[count], ilabel))
                expos_count += 1

    with open(result_foreground_cat_path + "/fg_src_list.dat", "w") as f:
        f.writelines(expos_avail_sub)

def process_one_tile(task, back_cat, expo_ra_edges, expo_dec_edges, cosmo_ccl, hubble, background_path):
    i, j, expo_num = task

    slt = (
        (back_cat['ra'] >= expo_ra_edges[i]) &
        (back_cat['ra'] < expo_ra_edges[i + 1]) &
        (back_cat['dec'] >= expo_dec_edges[j]) &
        (back_cat['dec'] < expo_dec_edges[j + 1])
    )
    src_data = back_cat[slt]
    src_num = len(src_data)
    if src_num == 0:
        return None  # skip empty tile

    curr_ra_min = src_data['ra'].min()
    curr_ra_max = src_data['ra'].max()
    ra_center = (curr_ra_min + curr_ra_max) / 2.
    dra = (curr_ra_max - curr_ra_min)/2.

    curr_dec_min = src_data['dec'].min()
    curr_dec_max = src_data['dec'].max()
    dec_center = (curr_dec_min + curr_dec_max) / 2.
    cos_dec_center = np.cos(np.deg2rad(dec_center))
    ddec = (curr_dec_max - curr_dec_min)/2.

    expo_pos = np.array([
        ra_center, dec_center, cos_dec_center,
        np.sqrt((dra*cos_dec_center)**2 + ddec**2)
    ], dtype=np.float32)

    dst_data = np.zeros((src_num, 11), dtype=np.float32)
    dst_data[:,0] = src_data['g1']
    dst_data[:,1] = src_data['g2']
    dst_data[:,2] = src_data['ra']
    dst_data[:,3] = np.deg2rad(src_data['ra'])
    dst_data[:,4] = src_data['dec']
    dst_data[:,5] = np.deg2rad(src_data['dec'])
    dst_data[:,6] = np.cos(np.deg2rad(src_data['dec']))
    dst_data[:,7] = np.sin(np.deg2rad(src_data['dec']))
    dst_data[:,8] = src_data['z']
    dst_data[:,9] = 0.01
    dst_data[:,10] = ccl.comoving_radial_distance(cosmo_ccl, 1./(1+src_data['z'])) * hubble

    expo_dst_path = f"{background_path}/src_expo{expo_num}.hdf5"
    with h5py.File(expo_dst_path, "w") as h5f_dst:
        h5f_dst["/expo_pos"] = expo_pos
        h5f_dst["/data"] = dst_data

    info_line = f"{expo_dst_path}\t{expo_num}\t{src_num}\t{expo_pos[0]:.6f}\t{expo_pos[1]:.6f}\t{expo_pos[2]:.6f}\t{expo_pos[3]:.6f}\n"
    return (expo_num, info_line)

def process_back_cat_parallel(back_cat, cosmo_ccl, odir, n_jobs=28, logger=None):

    hubble = cosmo_ccl.to_dict()['h']
    global_ramin, global_ramax = back_cat['ra'].min(), back_cat['ra'].max()
    global_decmin, global_decmax = back_cat['dec'].min(), back_cat['dec'].max()

    expo_ra_interval = 1.0 # deg
    expo_dec_interval = 1.0 # deg

    expo_ra_edges = np.arange(np.floor(global_ramin), np.ceil(global_ramax) + expo_ra_interval/2., expo_ra_interval)
    expo_dec_edges = np.arange(np.floor(global_decmin), np.ceil(global_decmax) + expo_dec_interval/2., expo_dec_interval)

    background_path = f"{odir}"
    os.makedirs(background_path, exist_ok=True)

    tasks = []
    expo_num = 0
    for i in range(len(expo_ra_edges) - 1):
        for j in range(len(expo_dec_edges) - 1):
            tasks.append((i, j, expo_num))
            expo_num += 1

    if logger is not None:
        logger.info(f"Total tiles: {len(tasks)}, running on {n_jobs} cores...")

    results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
        delayed(process_one_tile)(
            task, back_cat, expo_ra_edges, expo_dec_edges,
            cosmo_ccl, hubble, background_path
        )
        for task in tasks
    )

    results = [r for r in results if r is not None]
    results.sort(key=lambda x: x[0])  # sort by expo_num

    with open(f"{background_path}/bg_src_list.dat", "w") as f:
        f.writelines([r[1] for r in results])

    # print(f"Done. {len(results)} non-empty exposures written to {background_path}")

if __name__ == "__main__":

    # >>> ================         basic definitions        ====================== <<<
    ## CosmoGrid mocks
    ### define basic filename formats
    sim_fmt = "/data3/suchen/CosmoGridV1/grid/cosmo_{:06d}/run_0/"
    odir_fmt = "/data2/suchen/.cata/CosmoGrid/{:s}/cosmo_{:06d}" # one for lens or rand or srcs, the other for cosmology

    ### load cosmology
    with open("/data3/suchen/CosmoGridV1/grid/dirnames.txt", "r") as f:
        dirnames = f.readlines()
        cosmo_labels_tot = [int(i.strip("\n").split("_")[1]) for i in dirnames]

    ### load ngals
    with open("/data2/suchen/CosmoGrid/fix_HOD/ngals.json", "r") as f:
        ngal_dict = json.load(f)

    ### load random radec, prepare for generating random catalogs
    sim_void_rand_radec = np.load("/data2/suchen/CosmoGrid/Rand/boss_cmasslowztot_north_radec.npy")
    rand_lens_ratio = 5 # Nrand/Nlens

    ### define Rvmean list, for saving Rvmeans for each mock
    Rvmean_list = []

    ### read cosmo labels
    logger.info("Read cosmo labels")

    hod_param_fname = "cfgs/hod/hod_5params_dict.json"

    hod_params_dict = get_hod_params(hod_param_fname)
    cosmo_labels = []

    for icosmo_str in hod_params_dict.keys():
        if len(hod_params_dict[icosmo_str]) > 0:
            cosmo_labels.append(int(icosmo_str[5:]))

    for idx, icosmo in enumerate(cosmo_labels[1:]):
        # >>> below should be put into the loops
        # icosmo = 1
        # idx = 0
        logger.info("Process cosmo_{:06d}".format(icosmo))
        ### load cosmology
        cosmo_ccl = get_cosmo_from_file(f"/data3/suchen/CosmoGridV1/grid/cosmo_{icosmo:06d}/run_0/params.yml")

        # >>> ================    prepare background catalog    ====================== <<<

        try:
            sim_shear_cat = np.load(f"/data2/suchen/CosmoGrid/Shape/sigma0.3_kids_ngal/cosmo_{icosmo:06d}_run_0_kids_north_tomo4.npy")
        except:
            logger.info(f"Skip cosmo_{icosmo:06d}")
            # continue

        odir = odir_fmt.format("back_cat", icosmo)

        # logger.info("Process source catalog")
        # process_back_cat_parallel(sim_shear_cat, cosmo_ccl, odir, n_jobs=16, logger=logger)

        # >>> ==================   prepare foreground catalog   ====================== <<<
        IS_VOID = True # if void catalog
        lens_dirbase = "fore_cat_wrv"
        rand_dirbase = "fore_cat_rand_wrv"

        ### generate dir for saving lens catalog
        odir = odir_fmt.format(lens_dirbase, icosmo)
        if not os.path.isdir(odir):
            os.makedirs(odir)

        logger.info("Load lens catalog")
        sim_void_cat = np.load(f"/data2/suchen/CosmoGrid/fix_HOD_Void/cosmo_{icosmo:06d}_run_0_HOD_0_run_0_boss_north.npy")

        ### get z_Rv bounds for building p(z,Rv)
        zmin = np.minimum(sim_void_cat['z'].min(), 0.2)
        zmax = np.maximum(sim_void_cat['z'].max(), 0.4)
        Rvmin = sim_void_cat['Rv'].min()
        Rvmax = sim_void_cat['Rv'].max()

        z_rv_bounds = [(zmin, zmax), (Rvmin, Rvmax)]

        logger.info("Total voids (original): {}".format(len(sim_void_cat)))

        ### cut voids depending on rescaled void size: Rv*n^(1/3)
        curr_ngal = ngal_dict[f'cosmo{icosmo:06d}']
        scaled_Rv = sim_void_cat['Rv']*np.cbrt(curr_ngal*1e-4) # Rv * n^(1/3)

        logger.debug("scaled Rmin: {:.2f}, scaled Rmax: {:.2f}".format(np.min(scaled_Rv), np.max(scaled_Rv)))

        slt = (scaled_Rv > 1.2) & (scaled_Rv < 1.8)
        sim_void_cat = sim_void_cat[slt]

        curr_nvoids = len(sim_void_cat)
        logger.info("Total voids: {}".format(curr_nvoids))
        curr_nrands = int(curr_nvoids*rand_lens_ratio)

        Rvmean = np.mean(sim_void_cat['Rv'])
        Rvmean_list.append(Rvmean)

        ### separate jackknife regions
        fore_radec = np.c_[sim_void_cat['ra'], sim_void_cat['dec']]
        ncen = 128

        logger.info("Split jackknife patches")
        if not os.path.exists(f"{odir}/jkf_centers_cosmo{icosmo:06d}.npy"):
            labels, centers = split_jackknife_patches(fore_radec, njk=ncen)
            np.save(f"{odir}/jkf_centers_cosmo{icosmo:06d}.npy", centers)
        else:
            centers = np.load(f"{odir}/jkf_centers_cosmo{icosmo:06d}.npy")
            labels = split_jackknife_patches(fore_radec, centers=centers)

        # logger.info("Process lens catalog")
        # process_fore_cat(sim_void_cat, labels, cosmo_ccl, odir)

        ### prepare random catalog
        ### generate dir for saving random catalog
        odir = odir_fmt.format(rand_dirbase, icosmo)
        if not os.path.isdir(odir):
            os.makedirs(odir)

        #### downsample original random to a desired number
        sim_void_rancat = sim_void_rand_radec[np.random.choice(len(sim_void_rand_radec), curr_nrands, replace=False)]

        logger.info("Building KDE of p(z,Rv) and sampling")
        #### build KDE of p(z,Rv)
        zRv_KDE = bounded_kde_transform(np.c_[sim_void_cat['z'], sim_void_cat['Rv']], z_rv_bounds)
        #### generate (z,Rv) from KDE
        sim_void_rancat['z'], sim_void_rancat['Rv'] = resample_bounded(zRv_KDE, len(sim_void_rancat), z_rv_bounds)

        logger.info("Split jackknife patches")
        ### get jackknife labels using the same patch centers
        fore_radec = np.c_[sim_void_rancat['ra'], sim_void_rancat['dec']]
        labels = split_jackknife_patches(fore_radec, centers=centers)

        logger.info("Process random catalog")
        process_fore_cat(sim_void_rancat, labels, cosmo_ccl, odir)

        logger.info("Done.")

'''          ===========================    Duplicated functions    ========================        '''

# def generate_jackknife_centers(catalog_radec, njk=120):
#     km = kmeans_sample(catalog_radec, njk, maxiter=100, tol=1e-4, method='fast')
#     centers = km.centers
#     return centers

# def find_jackknife_labels(catalog_radec, centers):
#     labels = kmeans_radec.find_nearest(catalog_radec, centers)
#     return labels

# def process_back_cat(back_cat, cosmo_ccl, odir):
#     hubble = cosmo_ccl.to_dict()['h']

#     global_ramin = back_cat['ra'].min()
#     global_ramax = back_cat['ra'].max()
#     global_decmin = back_cat['dec'].min()
#     global_decmax = back_cat['dec'].max()

#     expo_ra_interval = 1. # deg
#     expo_dec_interval = 1. # deg

#     expo_ra_edges = np.arange(np.floor(global_ramin), np.ceil(global_ramax) + expo_ra_interval/2., expo_ra_interval)
#     expo_dec_edges = np.arange(np.floor(global_decmin), np.ceil(global_decmax) + expo_dec_interval/2., expo_dec_interval)

#     assert expo_ra_edges[0] < global_ramin
#     assert expo_ra_edges[-1] > global_ramax
#     assert expo_dec_edges[0] < global_decmin
#     assert expo_dec_edges[-1] > global_decmax

#     background_path = f"{odir}"
#     if not os.path.isdir(background_path):
#         os.makedirs(background_path)

#     expo_num = 0
#     expo_avail_sub = []
#     for i in trange(len(expo_ra_edges) - 1):
#         for j in range(len(expo_dec_edges) - 1):
#             slt = ((back_cat['ra'] >= expo_ra_edges[i]) & (back_cat['ra'] < expo_ra_edges[i + 1]) & (back_cat['dec'] >= expo_dec_edges[j]) & (back_cat['dec'] < expo_dec_edges[j + 1]))
#             src_data = back_cat[slt]
#             src_num = len(src_data)

#             if src_num != 0:
#                 curr_ra_min = src_data['ra'].min()
#                 curr_ra_max = src_data['ra'].max()
#                 ra_center = (curr_ra_min + curr_ra_max) / 2.
#                 dra = (curr_ra_max - curr_ra_min)/2.

#                 curr_dec_min = src_data['dec'].min()
#                 curr_dec_max = src_data['dec'].max()
#                 dec_center = (curr_dec_min + curr_dec_max) / 2.
#                 cos_dec_center = np.cos(np.deg2rad(dec_center))
#                 ddec = (curr_dec_max - curr_dec_min)/2.

#                 expo_pos = np.array([ra_center, dec_center, cos_dec_center,
#                                     np.sqrt((dra*cos_dec_center)**2 + ddec**2)], dtype=np.float32)
                
#                 # e1, e2, RA, RA_radian, DEC, DEC_radian, COS(DEC), SIN(DEC), Z, Z_ERR, COMOVING DISTANCE
#                 dst_data = np.zeros((src_num, 11), dtype=np.float32)
#                 dst_data[:,0] = src_data['g1']
#                 dst_data[:,1] = src_data['g2']
#                 dst_data[:,2] = src_data['ra']
#                 dst_data[:,3] = np.deg2rad(src_data['ra'])
#                 dst_data[:,4] = src_data['dec']
#                 dst_data[:,5] = np.deg2rad(src_data['dec'])
#                 dst_data[:,6] = np.cos(np.deg2rad(src_data['dec']))
#                 dst_data[:,7] = np.sin(np.deg2rad(src_data['dec']))
#                 dst_data[:,8] = src_data['z']
#                 dst_data[:,9] = 0.01
#                 dst_data[:,10] = ccl.comoving_radial_distance(cosmo_ccl, 1./(1+src_data['z'])) * hubble

#                 expo_dst_path = f"{background_path}/src_expo{expo_num}.hdf5"
#                 h5f_dst = h5py.File(expo_dst_path, "w")
#                 h5f_dst["/expo_pos"] = expo_pos
#                 h5f_dst["/data"] = dst_data
#                 h5f_dst.close()

#                 expo_avail_sub.append("%s\t%s\t%d\t%f\t%f\t%f\t%f\n"
#                                         % (expo_dst_path, expo_num, src_num, expo_pos[0], expo_pos[1],
#                                             expo_pos[2],
#                                             expo_pos[3]))
                
#                 expo_num += 1

#     with open(f"{background_path}/bg_src_list.dat", "w") as f:
#         f.writelines(expo_avail_sub)