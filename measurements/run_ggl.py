import numpy as np
import json

from container import *
from calculator import *

lens_dir = "/data2/suchen/CosmoGrid/Free_NGAL_wrsd/HOD_cmass/grid/Gals"
rand_dir = "/data2/suchen/CosmoGrid/Rand/DS20"
srcs_dir = "/data2/suchen/CosmoGrid/Shape/KiDS1000_North"
out_fmt = "./results/ggl/boss_ngc_kids1000_2tomos/cosmo{:06d}_HOD{:d}_ggl.fits"

with open("/data3/suchen/CosmoGridV1/grid_info/cosmo_label_param.json", "r") as f:
    cosmo_param_info = json.load(f)

def get_cosmo_dict(icosmo, cosmo_param_info):
    cosmo_dict = {}
    curr_info = cosmo_param_info[f'cosmo{icosmo:06d}']
    cosmo_dict['Om0'] = curr_info['Om']
    cosmo_dict['H0'] = curr_info['h']*100.0
    cosmo_dict['w0'] = curr_info['w']
    return cosmo_dict

cosmo_labels = [1]

ggl_config = GGLConfig(
    rp_min=1.0,
    rp_max=40.0,
    rp_bins=13,
    rp_unit='mpc',
    bin_type='log',
    flip_g1=True,
    wRSD=False
)

ggl_instance = GGLCalculator(config=ggl_config)

mock_rand_boss = SurveyData.load_cosmogrid_rand(f"{rand_dir}/boss_cmass_ngc_z0.4_0.6_official.fits")

for idx, icosmo in enumerate(cosmo_labels):
    cosmo_dict = get_cosmo_dict(icosmo, cosmo_param_info)
    mock_shape_kids = SurveyData.load_cosmogrid_shape(f"{srcs_dir}/cosmo_{icosmo:06d}_run_0_kids_north_2tomos.fits")
    if idx == 0:
        mock_rand_boss_matched = mock_rand_boss.match_to_reference(mock_shape_kids, nside=256, in_place=False)

    for ihod in range(2):
        mock_gal = SurveyData.load_cosmogrid_galaxy(f"{lens_dir}/cosmo_{icosmo:06d}_run_0_HOD_{ihod}_run_0_boss_north_2dflens_south.fits")
        mock_gal_boss = mock_gal.apply_condition_cut("survey != 3", in_place=False)

        mock_gal_boss_matched = mock_gal_boss.match_to_reference(mock_shape_kids, nside=256, in_place=False)

        lens_table = ggl_instance.compute_pairs(
            cosmo_dict, 
            lens_cat=mock_gal_boss_matched, 
            srcs_cat=mock_shape_kids,
            n_jobs=28
        )

        if ihod == 0:
            rand_table = ggl_instance.compute_pairs(
                cosmo_dict, 
                lens_cat=mock_rand_boss_matched, 
                srcs_cat=mock_shape_kids,
                n_jobs=28
            )
        
        esd = ggl_instance.stack_signals(lens_table, rand_table)

        esd.write(out_fmt.format(icosmo, ihod), overwrite=True)
