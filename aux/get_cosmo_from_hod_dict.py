'''
Get cosmological parameter from hod dictionary
'''

import sys
sys.path.append('/home/suchen/Program/CosmoGrid/')
import json
from tqdm import tqdm

from utils.io_func import *

hod_dict_path = "cfgs/hod/hod_5params_dict.json"
out_path = "cfgs/hod/hod_5params_dict_wcosmo.json"

sim_fmt = "/data3/suchen/CosmoGridV1/grid/cosmo_{:06d}/run_0/"

hod_params_dict = get_hod_params(hod_dict_path)

cosmo_labels = []
for icosmo_str in hod_params_dict.keys():
    if len(hod_params_dict[icosmo_str]) > 0:
        cosmo_labels.append(int(icosmo_str[5:]))

for icosmo in tqdm(cosmo_labels, desc='Processing'):
    cosmo_dict = get_cosmo_from_file(sim_fmt.format(icosmo) + "params.yml", otype='dict')
    hod_params_dict['cosmo{:06d}'.format(icosmo)]['cpar'] = cosmo_dict

with open(out_path, 'w') as f:
    json.dump(hod_params_dict, f)