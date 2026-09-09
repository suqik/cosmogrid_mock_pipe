import numpy as np
import sys
import os

from astropy.table import vstack
from container import *
from calculator import *

part = sys.argv[1]
wSN = bool(int(sys.argv[2]))
flip_g1 = bool(int(sys.argv[3])) if len(sys.argv) > 3 else True
flip_g2 = bool(int(sys.argv[4])) if len(sys.argv) > 4 else True

data_dirbase = "/public/share/ace66so15x/suchen/FastPM/MockCatalogs"

lens_dir = f"{data_dirbase}/Gals"
lens_fmt = "cosmo_{:06d}_realization_0000_HOD_{}_boss_lowz_north_2dflens_south.fits"
srcs_dir = f"{data_dirbase}/Shapes"
srcs_fmt = "cosmo_{:06d}_realization_0002_kids1000_north_3tomos.fits"
rand_fname = "/public/share/ace66so15x/suchen/CosmoGrid/Rand/DS20/boss_cmass_ngc_z0.4_0.6_official.fits"
cosmo_par_fname = "/public/home/suchen/Programs/Simtool/Pipeline/cfgs/fiducial/cosmo_list.txt"

out_dir = ("./results/ggl/boss_ngc_kids1000_3tomos_wSN" if wSN
           else "./results/ggl/boss_ngc_kids1000_3tomos")
out_dir += f"_f{int(flip_g1)}{int(flip_g2)}"
out_fmt = out_dir + "/cosmo{:06d}_HOD{:d}_ggl.fits"

tomos = (3, 4, 5)


def load_cosmo_params(fname):
    '''
    Parse the FastPM cosmology file (two header lines + rows of
    OmegaM, S8). Fixed: hubble, Omegab, ns; w0 = -1.
    '''

    with open(fname, "r") as stream:
        fixed_line = stream.readline().strip()
        varying_line = stream.readline().strip()

    fixed = {}
    for item in fixed_line[1:].split():
        name, value = item.split("=", 1)
        fixed[name] = float(value)

    varying_names = varying_line[1:].split()
    rows = np.loadtxt(fname, comments="#", ndmin=2)

    return fixed, varying_names, rows


def get_cosmo_dict(icosmo):
    selected = dict(zip(varying_names, rows[icosmo]))
    cosmo_dict = {}
    cosmo_dict['Om0'] = float(selected['OmegaM'])
    cosmo_dict['H0'] = fixed['hubble'] * 100.0
    cosmo_dict['w0'] = -1.0
    return cosmo_dict


fixed, varying_names, rows = load_cosmo_params(cosmo_par_fname)

if part == "test":
    part_file = "./cosmo_label_groups/part_fastpm_test"
else:
    part_file = "./cosmo_label_groups/part_fastpm_{:02d}".format(int(part))

cosmo_labels = []
with open(part_file) as f:
    for ilabel in f.readlines():
        cosmo_labels.append(int(ilabel.strip("\n").split("_")[1]))

ggl_config = GGLConfig(
    rp_min=1.0,
    rp_max=40.0,
    rp_bins=13,
    rp_unit='mpc_h',
    bin_type='log',
    flip_g1=flip_g1, # shear convention combos: f00 none, f10 g1 only, f01 g2 only, f11 both
    flip_g2=flip_g2,
    wRSD=True,
    wSN=wSN,
    wPhZ=False,
    njk=100, # jackknife fields (must be > 2)
)

ggl_instance = GGLCalculator(config=ggl_config)

### reference footprint: fixed to cosmo0 source catalog
ref_shape = SurveyData.load_cosmogrid_shape(os.path.join(srcs_dir, srcs_fmt.format(0)))

### random catalog: only RA/DEC are used; redshift not needed (z = 0.3 placeholder)
randcat = Table.read(rand_fname)
mock_rand_boss = SurveyData(
    ra=randcat['RA'],
    dec=randcat['DEC'],
    z=np.full(len(randcat), 0.3),
    w=np.ones(len(randcat)),
)
mock_rand_boss_matched = mock_rand_boss.match_to_reference(ref_shape, nside=256)

for icosmo in cosmo_labels:
    cosmo_dict = get_cosmo_dict(icosmo)
    mock_shape_kids = SurveyData.load_cosmogrid_shape(
        os.path.join(srcs_dir, srcs_fmt.format(icosmo)))

    ### collect per-(HOD, tomo) results, write one file per HOD at the end
    esd_by_hod = [[] for _ in range(10)]

    for tomo in tomos:
        mock_shape_t = mock_shape_kids.apply_condition_cut(f"tomo == {tomo}")

        srcs_table = ggl_instance.mk_srcs_cat(mock_shape_t)
        rand_table = ggl_instance.mk_lens_cat(mock_rand_boss_matched)

        for ihod in range(10):
            mock_gal = SurveyData.load_cosmogrid_galaxy(
                os.path.join(lens_dir, lens_fmt.format(icosmo, ihod)))
            mock_gal_boss = mock_gal[mock_gal.survey != 3]
            mock_gal_boss_matched = mock_gal_boss.match_to_reference(ref_shape, nside=256)

            lens_table = ggl_instance.mk_lens_cat(mock_gal_boss_matched)
            lens_table = ggl_instance.compute_pairs(cosmo_dict, lens_table, srcs_table, n_jobs=28)

            if ihod == 0:
                rand_table = ggl_instance.compute_pairs(cosmo_dict, rand_table, srcs_table, n_jobs=28)

            esd = ggl_instance.stack_signals(lens_table, rand_table)
            ### jackknife error per (HOD, tomo); the 10 HODs have different
            ### HOD parameters so they are not averaged
            jk_cov = ggl_instance.estimate_jackknife_cov(lens_table, rand_table)
            esd['ds_err_jk'] = np.sqrt(np.diag(jk_cov))
            esd['tomo'] = np.full(len(esd), tomo, dtype=int)
            esd_by_hod[ihod].append(esd)

    ### one file per (cosmo, HOD), rows = rp bins x 3 tomos
    for ihod in range(10):
        esd_all = vstack(esd_by_hod[ihod])
        esd_all.write(out_fmt.format(icosmo, ihod), overwrite=True)
