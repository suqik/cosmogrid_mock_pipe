import numpy as np
import os
import json

from container import *
from calculator import *

NJOBS = 8

wdir = "/Users/suqikuai777/Workspace/cosmogrid_mock_pipe/"
data_dirbase = "/Users/suqikuai777/Dataspace/FastPM/MockCatalogs"

lens_dir = f"{data_dirbase}/Gals"
lens_fmt = "cosmo_{:06d}_realization_0000_HOD_{:d}_boss_lowz_north_2dflens_south.fits"
rand_dir = f"{data_dirbase}/Rand/DS20"
srcs_dir = f"{data_dirbase}/Shape/kids1000_north_3tomos"
srcs_fmt = "cosmo_{:06d}_realization_0002_kids1000_north_3tomos.fits"
out_fmt = wdir + "results/ggl_fpm/boss_ngc_kids1000_3tomos/cosmo{:06d}_HOD{:d}_ggl.fits"

cosmo_param_info_file = "/Users/suqikuai777/Dataspace/FastPM/cosmo_list.txt"
cosmo_param_info = {}
with open(cosmo_param_info_file, "r") as f:
    ori_infos = f.readlines()

for line_idx, line in enumerate(ori_infos):
    if line_idx == 0:
        fix_param_info = line.strip("#")
        fix_param_list = fix_param_info.split(" ")
        hubble = float(fix_param_list[1].split("=")[1])
        Ob0 = float(fix_param_list[2].split("=")[1])
        ns = float(fix_param_list[3].split("=")[1])
        continue

    if line_idx == 1:
        continue
    
    if line_idx >= 2:
        cosmo_param_info[f'cosmo{(line_idx-2):06d}'] = {}
        cosmo_param_info[f'cosmo{(line_idx-2):06d}']['Om'] = float(line.split(" ")[0])
        cosmo_param_info[f'cosmo{(line_idx-2):06d}']['h'] = hubble
        cosmo_param_info[f'cosmo{(line_idx-2):06d}']['w'] = -1.0


def get_cosmo_dict(icosmo, cosmo_param_info):
    cosmo_dict = {}
    curr_info = cosmo_param_info[f'cosmo{icosmo:06d}']
    cosmo_dict['Om0'] = curr_info['Om']
    cosmo_dict['H0'] = curr_info['h']*100.0
    cosmo_dict['w0'] = curr_info['w']
    return cosmo_dict

if __name__ == "__main__":

    cosmo_labels = [0]

    ggl_config = GGLConfig(
        rp_min=1.0,
        rp_max=40.0,
        rp_bins=13,
        rp_unit='mpc_h',
        bin_type='log',
        flip_g1=False,
        flip_g2=True,
        wRSD=False, # for void lensing, should always be False
        wSN=False,
        wPhZ=False
    )

    ggl_instance = GGLCalculator(config=ggl_config)

    mock_rand_boss = SurveyData.load_cosmogrid_rand(f"{rand_dir}/boss_cmasslowztot_ngc_z0.2_0.4_official.fits")

    for idx, icosmo in enumerate(cosmo_labels):
        cosmo_dict = get_cosmo_dict(icosmo, cosmo_param_info)
        mock_shape_kids = SurveyData.load_cosmogrid_shape(os.path.join(srcs_dir, srcs_fmt.format(icosmo)))
        if idx == 0:
            mock_rand_boss_matched = mock_rand_boss.match_to_reference(mock_shape_kids, nside=256, in_place=False)

        for ihod in range(1):
            mock_gal = SurveyData.load_cosmogrid_galaxy(os.path.join(lens_dir, lens_fmt.format(icosmo, ihod)))
            mock_gal_boss = mock_gal[mock_gal.survey != 3]

            mock_gal_boss_matched = mock_gal_boss.match_to_reference(mock_shape_kids, nside=256, in_place=False)

            lens_table = ggl_instance.mk_lens_cat(mock_gal_boss_matched)

            if ihod == 0:
                srcs_table = ggl_instance.mk_srcs_cat(mock_shape_kids)
                rand_table = ggl_instance.mk_lens_cat(mock_rand_boss_matched)

            if ggl_config.rp_unit == "Rv":
                _ = ggl_instance.get_Rv_mean_mpch(mock_gal_boss_matched)

            lens_table = ggl_instance.compute_pairs(cosmo_dict, lens_table, srcs_table, n_jobs=NJOBS)

            if ihod == 0:
                rand_table = ggl_instance.compute_pairs(cosmo_dict, rand_table, srcs_table, n_jobs=NJOBS)
            
            esd = ggl_instance.stack_signals(lens_table, rand_table)

            esd.write(out_fmt.format(icosmo, ihod), overwrite=True)
