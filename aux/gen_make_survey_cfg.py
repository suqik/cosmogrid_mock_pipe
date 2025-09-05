'''
Script to generate config for make-survey code
'''

import sys
sys.path.append("/home/suchen/Program/CosmoGrid/")
import numpy as np

from utils.io_func import get_cosmo_from_file

make_survey_param = {
    "f---translate": [-900, -900, 0],
    "f---lbox": 900.0,
    "f---omega_m": 0.3,
    "f---omega_l": 0.7,
    "f---hubble": 1.0,
    "f---redshift_input": 0.3,
    "i---redshift_space": 0,
    "f---min_redshift": 0.2,
    "f---max_redshift": 0.33,
    "s---file_skymask": "/home/suchen/Program/CosmoGrid/catalogs/BOSS_LOWZ/mask_DR12v5_LOWZ_North.ply",
    "f---min_sky_weight": 0.5, 
    "i---downsample_sky": 0,
    "i---powspec": 1e4,
    "i---make_info": 0
}

def write_make_survey_cfg(param_dict, ofile):
    with open(ofile, "w+") as f:
        for ori_key, value in param_dict.items():
            val_type, key = ori_key.split("---")
            if key == "translate":
                f.write(f"{key}  {value[0]:.1f}, {value[1]:.1f}, {value[2]:.1f}\n")
            elif val_type == "f":
                f.write(f"{key}  {value:.1f}\n")
            elif val_type == "i":
                f.write(f"{key}  {int(value):d}\n")
            elif val_type == "s":
                f.write(f"{key}  {value}\n")

with open("/data3/suchen/CosmoGridV1/grid/dirnames.txt", "r") as f:
    dirnames = f.readlines()
    cosmo_labels = [int(i.strip("\n").split("_")[1]) for i in dirnames]

cpar_fmt = "/data3/suchen/CosmoGridV1/grid/cosmo_{:06d}/run_0/params.yml"
cfg_fmt = "cfgs/make-survey/make_survey_cosmo_{:06d}.cfg"

for icosmo_label in cosmo_labels:
    cosmo_pars = get_cosmo_from_file(cpar_fmt.format(icosmo_label), otype='dict')
    curr_make_survey_param = make_survey_param.copy()
    curr_make_survey_param["f---omega_m"] = cosmo_pars["Om"]
    curr_make_survey_param["f---omega_l"] = 1 - cosmo_pars["Om"]

    write_make_survey_cfg(curr_make_survey_param, cfg_fmt.format(icosmo_label))