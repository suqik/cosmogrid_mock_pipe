import numpy as np
import os
import json
import sys

from container import *
from calculator import *

part = int(sys.argv[1])

data_dirbase = "/public/share/ace66so15x/suchen/CosmoGrid"

lens_dir = f"{data_dirbase}/Free_NGAL_wrsd/HOD_cmass/Voids"
lens_fmt = "cosmo_{:06d}_run_0_HOD_{}_run_0_boss_north_2dflens_south.fits"
rand_dir = f"{data_dirbase}/Rand/DS20"
srcs_dir = f"{data_dirbase}/Shape/kids1000_north_2tomos"
srcs_fmt = "cosmo_{:06d}_run_0_kids_north_2tomos.fits"
out_fmt = "./results/vl/boss_ngc_kids1000_2tomos_wSN/cosmo{:06d}_HOD{:d}_ggl.fits"

with open(f"{data_dirbase}/cosmo_label_param.json", "r") as f:
    cosmo_param_info = json.load(f)

cosmo_label_tot = list(cosmo_param_info.keys())
ngal_list = np.loadtxt(f"{data_dirbase}/Free_NGAL_wrsd/HOD_cmass/ngals_list.txt")
ngal_dict = {}
for idx, icosmo in enumerate(cosmo_param_info.keys()):
    ngal_dict[icosmo] = ngal_list[idx*10:(idx+1)*10]

def get_cosmo_dict(icosmo, cosmo_param_info):
    cosmo_dict = {}
    curr_info = cosmo_param_info[f'cosmo{icosmo:06d}']
    cosmo_dict['Om0'] = curr_info['Om']
    cosmo_dict['H0'] = curr_info['h']*100.0
    cosmo_dict['w0'] = curr_info['w']
    return cosmo_dict

cosmo_labels = []
with open("./cosmo_label_groups/part_{:02d}".format(part)) as f:
    for ilabel in f.readlines():
        cosmo_labels.append(int(ilabel.strip("\n").split("_")[1]))

ggl_config = GGLConfig(
    rp_min=0.1,
    rp_max=3.0,
    rp_bins=13,
    rp_unit='Rv',
    bin_type='log',
    flip_g1=True,
    wRSD=False, # for void lensing, should always be False
    wSN=True,
    wPhZ=False
)

ggl_instance = GGLCalculator(config=ggl_config)

mock_rand_boss = SurveyData.load_cosmogrid_rand(f"{rand_dir}/boss_cmass_ngc_z0.4_0.6_official.fits")

for idx, icosmo in enumerate(cosmo_labels):
    cosmo_dict = get_cosmo_dict(icosmo, cosmo_param_info)
    mock_shape_kids = SurveyData.load_cosmogrid_shape(os.path.join(srcs_dir, srcs_fmt.format(icosmo)))
    if idx == 0:
        mock_rand_boss_matched = mock_rand_boss.match_to_reference(mock_shape_kids, nside=256, in_place=False)

    for ihod in range(10):
        mock_void = SurveyData.load_cosmogrid_void(os.path.join(lens_dir, lens_fmt.format(icosmo, ihod)))
        mock_void_boss = mock_void[mock_void.survey != 3]

        ngal_curr = ngal_dict[f'cosmo{icosmo:06d}'][ihod]

        rescaled_Rv = mock_void_boss.Rv * np.cbrt(ngal_curr)
        mock_void_boss = mock_void_boss[rescaled_Rv > 1.0]

        mock_void_boss_matched = mock_void_boss.match_to_reference(mock_shape_kids, nside=256, in_place=False)

        lens_table = ggl_instance.mk_lens_cat(mock_void_boss_matched)

        if ihod == 0:
            srcs_table = ggl_instance.mk_srcs_cat(mock_shape_kids)
            rand_table = ggl_instance.mk_lens_cat(mock_rand_boss_matched)

        if ggl_config.rp_unit == "Rv":
            _ = ggl_instance.get_Rv_mean_mpch(mock_void_boss_matched)

        lens_table = ggl_instance.compute_pairs(cosmo_dict, lens_table, srcs_table, n_jobs=28)
        rand_table = ggl_instance.compute_pairs(cosmo_dict, rand_table, srcs_table, n_jobs=28)
        
        esd = ggl_instance.stack_signals(lens_table, rand_table)

        esd.write(out_fmt.format(icosmo, ihod), overwrite=True)
