'''
Script to measure gal clus w(theta), ggl gamma_t(theta), cosmic shear xi+/xi-
'''

import sys
sys.path.append("/home/suchen/Program/CosmoGrid/")

import numpy as np
from loguru import logger
import treecorr
from utils.io_func import *
from utils.mkfore_utils import *

def find_matching_samples_table(ref_cat, match_cat, nside_x=64, nside_y=64):
    
    ref_xy = np.vstack([ref_cat['ra'], ref_cat['dec']]).T
    match_xy = np.vstack([match_cat['ra'], match_cat['dec']]).T

    all_points = np.vstack([ref_xy, match_xy])
    xmin, ymin = np.min(all_points, axis=0)
    xmax, ymax = np.max(all_points, axis=0)

    x_edges = np.linspace(xmin, xmax, nside_x + 1)
    y_edges = np.linspace(ymin, ymax, nside_y + 1)

    def compute_indices(data):
        ix = np.digitize(data[:, 0], x_edges) - 1
        iy = np.digitize(data[:, 1], y_edges) - 1
        return ix, iy

    ref_ix, ref_iy = compute_indices(ref_xy)
    match_ix, match_iy = compute_indices(match_xy)

    def valid_mask(ix, iy, n_x, n_y):
        return (ix >= 0) & (ix < n_x) & (iy >= 0) & (iy < n_y)

    ref_mask = valid_mask(ref_ix, ref_iy, nside_x, nside_y)
    match_mask = valid_mask(match_ix, match_iy, nside_x, nside_y)

    ref_cells = set(zip(ref_ix[ref_mask], ref_iy[ref_mask]))
    match_cells = set(zip(match_ix[match_mask], match_iy[match_mask]))
    common_cells = ref_cells & match_cells

    match_selected = [i for i, (ix, iy) in enumerate(zip(match_ix, match_iy)) if (ix, iy) in common_cells]

    return match_cat[match_selected]

def process_gal_rand_catalog(cosmo_label, hod_label, gal_dir, rand_dir, selection='boss'):
    if selection != 'boss' and selection != '2dflens':
        raise ValueError("selection must be `boss` or `2dflens`")
    
    ### load galaxy catalog and random catalog
    logger.info("Loading galaxy catalog and random catalog")
    gal_cat = np.load(gal_dir+f"cosmo_{cosmo_label:06d}_run_0_HOD_{hod_label}_run_0_boss_north_2dflens_south.npy")
    rand_cat = np.load(rand_dir+"bosslowz_north_2dflens_south_radec.npy")

    ### select used area
    logger.info(f"Selecting {selection} part")
    if selection == "boss":
        gal_cat = gal_cat[gal_cat['survey'] != 3]
        rand_cat = rand_cat[rand_cat['survey'] != 3]
    if selection == "2dflens":
        gal_cat = gal_cat[gal_cat['survey'] == 3]
        rand_cat = rand_cat[rand_cat['survey'] == 3]

    return gal_cat, rand_cat

def process_shape_catalog(cosmo_label, tomo_bin, shape_dir, selection='kids_north'):
    if selection != 'kids_north' and selection != 'kids_south':
        raise ValueError("selection must be `kids_north` or `kids_south`")
    if tomo_bin < 3 or tomo_bin > 5:
        raise ValueError("tomo_bin must be in [3,5]")
    
    ### load shape catalog
    logger.info(f"Loading {selection} shape catalog, tomographic bin {tomo_bin:d}")
    shape_cat = np.load(shape_dir+f"cosmo_{cosmo_label:06d}_run_0_{selection}_tomo{tomo_bin:d}.npy")

    return shape_cat

def wtheta_single_run(gal_cat, rand_cat, min_sep, max_sep, nbins,
                       sep_units='arcmin', bin_type='Log', rr_object=None, ncpu=1):

    ### initial treecorr catalog
    tc_gcat = treecorr.Catalog(ra=gal_cat['ra'], dec=gal_cat['dec'], ra_units='degrees', dec_units='degrees')
    tc_rcat = treecorr.Catalog(ra=rand_cat['ra'], dec=rand_cat['dec'], ra_units='degrees', dec_units='degrees')

    ### initialize pair counter
    dd = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units=sep_units, bin_type=bin_type)
    dr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units=sep_units, bin_type=bin_type)
    if rr_object is not None:
        rr = rr_object
    else:
        rr = treecorr.NNCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units=sep_units, bin_type=bin_type)

    ### pair counting
    logger.info("Counting DD")
    dd.process(tc_gcat, num_threads=ncpu)
    logger.info("Counting DR")
    dr.process(tc_gcat, tc_rcat, num_threads=ncpu)
    if rr_object is not None:
        logger.info("Use input RR")
    else:
        logger.info("Counting RR")
        rr.process(tc_rcat, num_threads=ncpu)
    
    ### 2pcf estimation
    logger.info("Estimating 2pcf")
    wtheta, _ = dd.calculateXi(dr=dr, rr=rr)

    return dd.rnom, wtheta, rr

def gammat_single_run(gal_cat, rand_cat, shape_cat, min_sep, max_sep, nbins,
                       sep_units='arcmin', bin_type='Log', flip_g1=True, ncpu=1):
    
    ### initial treecorr catalog
    tc_gcat = treecorr.Catalog(ra=gal_cat['ra'], dec=gal_cat['dec'], ra_units='degrees', dec_units='degrees')
    tc_rcat = treecorr.Catalog(ra=rand_cat['ra'], dec=rand_cat['dec'], ra_units='degrees', dec_units='degrees')
    tc_bcat = treecorr.Catalog(ra=shape_cat['ra'], dec=shape_cat['dec'], g1=shape_cat['g1'], g2=shape_cat['g2'], flip_g1=flip_g1, ra_units='degrees', dec_units='degrees')

    ### initialize pair counter
    dg = treecorr.NGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units=sep_units, bin_type=bin_type)
    rg = treecorr.NGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units=sep_units, bin_type=bin_type)

    ### pair counting
    logger.info(f"Counting DG")
    dg.process(tc_gcat, tc_bcat, num_threads=ncpu)
    logger.info(f"Counting RG")
    rg.process(tc_rcat, tc_bcat, num_threads=ncpu)
    
    ### 2pcf estimation
    logger.info(f"Estimating 2pcf")
    gammat, _, _ = dg.calculateXi(rg=rg)

    return dg.rnom, gammat

def cosmic_shear_single_run(shape_cat1, shape_cat2, min_sep, max_sep, nbins,
                       sep_units='arcmin', bin_type='Log', ncpu=1):
    
    ### initial treecorr catalog
    tc_scat1 = treecorr.Catalog(ra=shape_cat1['ra'], dec=shape_cat1['dec'], g1=shape_cat1['g1'], g2=shape_cat1['g2'], flip_g2=True, ra_units='degrees', dec_units='degrees')
    tc_scat2 = treecorr.Catalog(ra=shape_cat2['ra'], dec=shape_cat2['dec'], g1=shape_cat2['g1'], g2=shape_cat2['g2'], flip_g2=True, ra_units='degrees', dec_units='degrees')

    ### initialize pair counter
    gg = treecorr.GGCorrelation(min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units=sep_units, bin_type=bin_type)

    ### pair counting
    logger.info("Counting GG")
    gg.process(tc_scat1, tc_scat2, num_threads=ncpu)

    ### 2pcf estimation
    logger.info("Estimating 2pcf")
    xip = gg.xip
    xim = gg.xim

    return gg.rnom, xip, xim

if __name__ == "__main__":
    import json
    import sys

    NPARTS = int(sys.argv[1])
    PART = int(sys.argv[2])
    MODE = "gammat"
    Ncpu_part = 8

    hod_param_fname = "/home/suchen/Program/CosmoGrid/cfgs/hod/hod_5params_dict_Nsat1000.json"
    cosmo_labels_tot = get_cosmo_name_list_process(hod_param_fname)

    ncosmo_per_part = int(len(cosmo_labels_tot)/NPARTS)
    if PART == NPARTS-1:
        cosmo_labels = cosmo_labels_tot[PART*ncosmo_per_part:]
    else:
        cosmo_labels = cosmo_labels_tot[PART*ncosmo_per_part:(PART+1)*ncosmo_per_part]

    logger.info(f"Processing {cosmo_labels[0]} to {cosmo_labels[-1]}")

    # =============   For test   ===========
    cosmo_labels = [204707]
    # ======================================
    
    nhod_per_cosmo = 1

    ### in order to get cosmo & hod params
    with open(hod_param_fname, "r") as f:
        hod_param_dict = json.load(f)

    if MODE == "wtheta" or MODE == "gammat":
        ### lowz sample
        # lens_survey = "lowz"
        # zlens_mins = [0.2, 0.3, 0.4]
        # zlens_maxs = [0.3, 0.4, 0.5]

        ### cmass sample
        lens_survey = "cmass"
        zlens_mins = [0.4, 0.5]
        zlens_maxs = [0.5, 0.6]

        lens_survey_selection = "boss"

        gal_dir = f"/data2/suchen/CosmoGrid/Nsat1000_wrsd/HOD_{lens_survey}/"
        rand_dir = "/data2/suchen/CosmoGrid/Rand/"

    if MODE == "gammat" or MODE == "cosmic_shear":
        srcs_survey = "kids"
        srcs_survey_selection = "kids_north"
        srcs_tomo_zmins = [0.5, 0.7, 0.9]
        srcs_tomos = [3,4,5]
        
        if srcs_survey == "kids":
            shape_dir = f"/data2/suchen/CosmoGrid/Shape/KiDS_ngal_suits/"

    ### output file format
    if MODE == "wtheta":
        out_dir_fmt = "results/wtheta/{}_Nsat1000_z{:.1f}_{:.1f}/"
        out_fmt = out_dir_fmt + "cosmo_{:06d}_HOD_{:d}_wtheta.npz"

        for zlen_bin in range(len(zlens_mins)):
            out_dir = out_dir_fmt.format(lens_survey, zlens_mins[zlen_bin], zlens_maxs[zlen_bin])
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

    if MODE == "gammat":
        ### srcs tomography
        # out_dir_fmt = "results/gammat/{}_{}_high_ngal_z{:.1f}_{:.1f}_tomo{:d}/" # lens_survey, srcs_survey, zlen_min, zlen_max, tomo_bin
        # out_fmt = out_dir_fmt + "cosmo_{:06d}_HOD_{:d}_gammat.npz"

        # for zlen_bin in range(len(zlens_mins)):
        #     for tomo_bin in range(len(srcs_tomos)):
        #         if srcs_tomo_zmins[tomo_bin] < zlens_mins[zlen_bin]:
        #             continue
        #         else:
        #             out_dir = out_dir_fmt.format(lens_survey, srcs_survey, zlens_mins[zlen_bin], zlens_maxs[zlen_bin], srcs_tomos[tomo_bin])
        #             if not os.path.exists(out_dir):
        #                 os.makedirs(out_dir)

        ### srcs combine
        out_dir_fmt = "results/gammat/{}_{}_Nsat1000_z{:.1f}_{:.1f}/" # lens_survey, srcs_survey, zlen_min, zlen_max
        out_fmt = out_dir_fmt + "cosmo_{:06d}_HOD_{:d}_gammat.npz"

        for zlen_bin in range(len(zlens_mins)):
            out_dir = out_dir_fmt.format(lens_survey, srcs_survey, zlens_mins[zlen_bin], zlens_maxs[zlen_bin])
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

    if MODE == "cosmic_shear":
        out_dir_fmt = "results/cosmic_shear/{}_tomo{}_{}/"
        out_fmt = out_dir_fmt + "cosmo_{:06d}_cosmic_shear.npz"

        for tomo_bin_i in range(len(srcs_tomos)):
            for tomo_bin_j in range(tomo_bin_i, len(srcs_tomos)):
                out_dir = out_dir_fmt.format(srcs_survey, srcs_tomos[tomo_bin_i], srcs_tomos[tomo_bin_j])
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

    ### setup of measurements
    min_sep = 1.0
    max_sep = 300.0
    nbins = 20

    sep_units = 'arcmin'

    global_count = 0


    ###   Main Process   ###


    for cosmo_idx, icosmo in enumerate(cosmo_labels):

        if MODE == "wtheta" or MODE == "gammat":

            for ihod in range(nhod_per_cosmo):

                gal_cat_tot, rand_cat = process_gal_rand_catalog(icosmo, ihod, gal_dir, rand_dir, selection=lens_survey_selection)

                for zlens_bin in range(len(zlens_mins)):

                    zlens_min = zlens_mins[zlens_bin]
                    zlens_max = zlens_maxs[zlens_bin]
                    
                    lens_radial_selection = ((gal_cat_tot['z'] > zlens_min) & (gal_cat_tot['z'] < zlens_max))
                    gal_cat = gal_cat_tot[lens_radial_selection]
                
                    if MODE == "wtheta":

                        if global_count == 0:
                            theta_bin, wtheta, rr = wtheta_single_run(
                                gal_cat, rand_cat, 
                                min_sep=min_sep, max_sep=max_sep, nbins=nbins, sep_units='arcmin', bin_type='Log', 
                                rr_object=None,
                                ncpu=Ncpu_part)
                            
                            np.savez(out_fmt.format(lens_survey, zlens_min, zlens_max, icosmo, ihod), theta_bin=theta_bin, wtheta=wtheta)
                        else:
                            theta_bin, wtheta, rr = wtheta_single_run(
                                gal_cat, rand_cat, 
                                min_sep=min_sep, max_sep=max_sep, nbins=nbins, 
                                sep_units='arcmin', bin_type='Log', 
                                rr_object=rr,
                                ncpu=Ncpu_part)

                            np.savez(out_fmt.format(lens_survey, zlens_min, zlens_max, icosmo, ihod), theta_bin=theta_bin, wtheta=wtheta)

                        global_count += 1

                    elif MODE == "gammat":
                        shape_cat = []
                        for srcs_bin in range(len(srcs_tomos)):
                            if srcs_tomo_zmins[srcs_bin] < zlens_min:
                                continue
                            else:
                                srcs_bin_label = srcs_tomos[srcs_bin]
                                tmp = process_shape_catalog(icosmo, srcs_bin_label, shape_dir, selection=srcs_survey_selection)
                                shape_cat.append(tmp)

                        shape_cat_for_match = shape_cat[0]
                        shape_cat = np.concatenate(shape_cat)

                        logger.info("Matching shape catalogs and lens catalog")

                        gal_cat_matched = find_matching_samples_table(shape_cat_for_match, gal_cat, nside_x=500, nside_y=300)
                        rand_cat_matched = find_matching_samples_table(shape_cat_for_match, rand_cat, nside_x=500, nside_y=300)

                        theta_bin, gammat = gammat_single_run(
                            gal_cat_matched, rand_cat_matched, shape_cat, 
                            min_sep=min_sep, max_sep=max_sep, nbins=nbins, 
                            sep_units='arcmin', bin_type='Log',
                            ncpu=Ncpu_part)

                        logger.info(f"save to file")

                        np.savez(out_fmt.format(lens_survey, srcs_survey, zlens_min, zlens_max, icosmo, ihod), theta_bin=theta_bin, gammat=gammat)
    
        elif MODE == "cosmic_shear":
            
            shape_cat_list = []

            for srcs_tomo_bin in srcs_tomos:
                shape_cat_list.append(
                    process_shape_catalog(icosmo, srcs_tomo_bin, shape_dir, selection=srcs_survey_selection)
                )

            for tomo_bin_i in range(len(srcs_tomos)):

                shape_cat1 = shape_cat_list[tomo_bin_i]

                for tomo_bin_j in range(tomo_bin_i, len(srcs_tomos)):

                    shape_cat2 = shape_cat_list[tomo_bin_j]

                    theta_bin, xi_plus, xi_minus = cosmic_shear_single_run(
                        shape_cat1, shape_cat2, 
                        min_sep=min_sep, max_sep=max_sep, nbins=nbins, 
                        sep_units='arcmin', bin_type='Log',
                        ncpu=Ncpu_part)
                    
                    np.savez(out_fmt.format(srcs_survey, srcs_tomos[tomo_bin_i], srcs_tomos[tomo_bin_j], icosmo), theta_bin=theta_bin, xi_plus=xi_plus, xi_minus=xi_minus)
