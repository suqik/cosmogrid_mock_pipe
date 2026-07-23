'''
Utils used in making background samples
'''

import numpy as np
from scipy.spatial.transform import Rotation as R
import healpy as hp
import pymangle
from .io_func import bgal_type
import warnings

def make_nofz(zctrs, nz):
    zedges = 0.5*(zctrs[1:] + zctrs[:-1])
    nz = nz[1:-1]
    nofz = {}
    nofz['zedges'] = zedges
    nofz['nz'] = nz

    return nofz

def gen_angle_positions_from_healpix(ngal:float, mask:np.ndarray):
    # get RA DEC of HEALPIX-type footprint
    nside = hp.npix2nside(len(mask))
    ra, dec = hp.pix2ang(nside, np.argwhere(mask != 0).flatten(), lonlat=True)
    
    # get effective galaxy numbers
    sample_area = (ra.max() - ra.min()) * (dec.max() - dec.min()) # only valid for KiDS
    Ngal = int(np.around(ngal * sample_area * 60**2)) # ngal is in arcmin^-2

    # sample RA DEC
    sampled_ra = np.random.uniform(low=ra.min(), high=ra.max(), size=Ngal)
    cos_sampled_dec = np.random.uniform(low=np.cos(np.deg2rad(90-dec.min())), high=np.cos(np.deg2rad(90-dec.max())), size=Ngal)
    sampled_dec = np.rad2deg(np.arccos(cos_sampled_dec))
    sampled_dec = 90. - sampled_dec

    sample_pix = hp.ang2pix(nside, sampled_ra, sampled_dec, lonlat=True)
    picked_pix_in_sample = np.isin(sample_pix, np.argwhere(mask!=0).flatten())

    Ngal = np.sum(picked_pix_in_sample)
    picked_ra = sampled_ra[picked_pix_in_sample]
    picked_dec = sampled_dec[picked_pix_in_sample]

    return picked_ra, picked_dec

def gen_angle_positions_from_mangle(ngal:float, mask:pymangle.Mangle):
    sample_area = (mask.areas * mask.weights).sum() # deg^2
    Ngal = int(np.around(ngal * sample_area * 60**2))
    sampled_ra, sampled_dec = mask.genrand(Ngal)
    sample_weights = mask.weight(sampled_ra, sampled_dec)
    picked_idx = (sample_weights > 0)
    picked_ra = sampled_ra[picked_idx]
    picked_dec = sampled_dec[picked_idx]

    return picked_ra, picked_dec

def gen_redshifts_from_nofz(Ngal:float, nofz:dict|float|list, photo_z_err=None, seed=None):
    if isinstance(nofz, dict):
        # sample redshift
        zsamples = []
        zedges = nofz['zedges']
        nz = nofz['nz']

        for i in range(len(nz)-1):
            iNgal = int(Ngal*nz[i])
            zsamples.append(np.random.uniform(low=zedges[i], high=zedges[i+1], size=iNgal))

        zsamples = np.concatenate(zsamples)

    if isinstance(nofz, float):
        zsamples = np.ones(Ngal) * nofz
    if isinstance(nofz, list):
        zsamples = np.random.uniform(low=nofz[0], high=nofz[1], size=Ngal)

    if photo_z_err is not None:
        rng = np.random.default_rng(seed=seed)
        sigma_z = rng.normal(loc=0.0, scale=photo_z_err, size=(len(zsamples),))
        zph_samples = zsamples + sigma_z
        ### require the true Zs are always larger than 0
        phys_cut = (zph_samples > 0)
        zph_samples = zph_samples[phys_cut]
        zsamples = zsamples[phys_cut]

    else:
        zph_samples = zsamples

    return zsamples, zph_samples

def assign_shear_vals(cat_ra, cat_dec, cat_z, shear_map_dict:dict, sigma_e:float, seed=None):
    shell_zctrs = np.array([shear_map_dict[f'shell{i}']['redshift'] for i in range(len(shear_map_dict))])
    shell_zmax = shell_zctrs[-1]

    deltaz_max = shell_zmax - shear_map_dict[f'shell{len(shear_map_dict)-2}']['redshift']
    shell_zedges = 0.5 * (shell_zctrs[1:] + shell_zctrs[:-1])
    shell_zedges = np.append(0, np.append(shell_zedges, shell_zmax + deltaz_max))

    zcut = cat_z < shell_zmax + deltaz_max
    cat_ra = cat_ra[zcut]
    cat_dec = cat_dec[zcut]
    cat_z = cat_z[zcut]
    Ngal = zcut.sum()

    g1_pure = np.zeros_like(cat_z)
    g2_pure = np.zeros_like(cat_z)
    for ishell in range(len(shell_zedges) - 1):
        select = (cat_z >= shell_zedges[ishell]) & (cat_z < shell_zedges[ishell+1])
        selected_ra = cat_ra[select]
        selected_dec = cat_dec[select]

        selected_g1 = hp.get_interp_val(shear_map_dict[f'shell{ishell}']['gamma1'], selected_ra, selected_dec, lonlat=True)
        selected_g2 = hp.get_interp_val(shear_map_dict[f'shell{ishell}']['gamma2'], selected_ra, selected_dec, lonlat=True)

        g1_pure[select] = selected_g1
        g2_pure[select] = selected_g2

    if sigma_e is not None:
        rng = np.random.default_rng(seed=seed)
        n1n2 = rng.normal(loc=0, scale=sigma_e, size=(Ngal,2))
        n_complex = n1n2[:,0] + n1n2[:,1]*1j
        del n1n2
        g_complex = g1_pure + g2_pure*1j
        e_complex = (g_complex + n_complex) / (1 + n_complex*np.conj(g_complex))
        g1_noise = np.real(e_complex)
        g2_noise = np.imag(e_complex)
    else:
        g1_noise = g1_pure
        g2_noise = g2_pure

    return g1_pure, g2_pure, g1_noise, g2_noise

### sampling and add shape noise
def gen_random_positions(ngal:float, mask:np.ndarray, nofz:dict|float|list, photo_z_err=None, seed=None, logger=None) -> np.ndarray:
    # get RA DEC of footprint
    nside = hp.npix2nside(len(mask))
    ra, dec = hp.pix2ang(nside, np.argwhere(mask != 0).flatten(), lonlat=True)
    
    # get effective galaxy numbers
    sample_area = (ra.max() - ra.min()) * (dec.max() - dec.min())
    Ngal = int(np.around(ngal * sample_area * 60**2)) # ngal is in arcmin^-2
    if logger is not None:
        logger.info(f"Generating {Ngal} galaxies")
    
    # sample RA DEC
    sampled_ra = np.random.uniform(low=ra.min(), high=ra.max(), size=Ngal)
    cos_sampled_dec = np.random.uniform(low=np.cos(np.deg2rad(90-dec.min())), high=np.cos(np.deg2rad(90-dec.max())), size=Ngal)
    sampled_dec = np.rad2deg(np.arccos(cos_sampled_dec))
    sampled_dec = 90. - sampled_dec

    sample_pix = hp.ang2pix(nside, sampled_ra, sampled_dec, lonlat=True)
    picked_pix_in_sample = np.isin(sample_pix, np.argwhere(mask!=0).flatten())

    Ngal = np.sum(picked_pix_in_sample)
    picked_ra = sampled_ra[picked_pix_in_sample]
    picked_dec = sampled_dec[picked_pix_in_sample]

    bg_galcat = np.empty(Ngal, dtype=bgal_type)
    bg_galcat['ra'] = picked_ra
    bg_galcat['dec'] = picked_dec

    if isinstance(nofz, dict):
        # sample redshift
        zsamples = []
        zedges = nofz['zedges']
        nz = nofz['nz']

        for i in range(len(nz)-1):
            iNgal = int(Ngal*nz[i])
            zsamples.append(np.random.uniform(low=zedges[i], high=zedges[i+1], size=iNgal))

        zsamples = np.concatenate(zsamples)
        Ngal = len(zsamples)

        id_alive = np.random.choice(np.arange(len(bg_galcat)), Ngal, replace=False)
        bg_galcat = bg_galcat[id_alive]
        bg_galcat['z_true'] = zsamples

    if isinstance(nofz, float):
        bg_galcat['z_true'] = np.ones(Ngal) * nofz
    if isinstance(nofz, list):
        bg_galcat['z_true'] = np.random.uniform(low=nofz[0], high=nofz[1], size=Ngal)

    if photo_z_err is not None:
        rng = np.random.default_rng(seed=seed)
        sigma_z = rng.normal(loc=0.0, scale=photo_z_err, size=(len(bg_galcat),))
        bg_galcat['z'] = bg_galcat['z_true'] + sigma_z
        bg_galcat['sigz'] = photo_z_err
        ### require the true Zs are always larger than 0
        phys_cut = (bg_galcat['z'] > 0)
        bg_galcat = bg_galcat[phys_cut]

    else:
        bg_galcat['z'] = bg_galcat['z_true']
        bg_galcat['sigz'] = 0.0

    return bg_galcat

def get_gal_shear(bg_galcat:np.ndarray, shear_map_dict:dict, sigma_e:float=None, seed=None) -> np.ndarray:
    shell_zctrs = np.array([shear_map_dict[f'shell{i}']['redshift'] for i in range(len(shear_map_dict))])
    shell_zmax = shell_zctrs[-1]

    deltaz_max = shell_zmax - shear_map_dict[f'shell{len(shear_map_dict)-2}']['redshift']
    shell_zedges = 0.5 * (shell_zctrs[1:] + shell_zctrs[:-1])
    shell_zedges = np.append(0, np.append(shell_zedges, shell_zmax + deltaz_max))

    zcut = bg_galcat['z'] < shell_zmax + deltaz_max
    bg_galcat = bg_galcat[zcut]
    Ngal = len(bg_galcat)

    for ishell in range(len(shell_zedges) - 1):
        select = (bg_galcat['z'] >= shell_zedges[ishell]) & (bg_galcat['z'] < shell_zedges[ishell+1])
        selected_ra = bg_galcat['ra'][select]
        selected_dec = bg_galcat['dec'][select]

        selected_g1 = hp.get_interp_val(shear_map_dict[f'shell{ishell}']['gamma1'], selected_ra, selected_dec, lonlat=True)
        selected_g2 = hp.get_interp_val(shear_map_dict[f'shell{ishell}']['gamma2'], selected_ra, selected_dec, lonlat=True)

        bg_galcat['g1_pure'][select] = selected_g1
        bg_galcat['g2_pure'][select] = selected_g2

    if sigma_e is not None:
        rng = np.random.default_rng(seed=seed)
        n1n2 = rng.normal(loc=0, scale=sigma_e, size=(Ngal,2))
        n_complex = n1n2[:,0] + n1n2[:,1]*1j
        del n1n2
        g_complex = bg_galcat["g1_pure"] + bg_galcat["g2_pure"]*1j
        e_complex = (g_complex + n_complex) / (1 + n_complex*np.conj(g_complex))
        bg_galcat["g1"] = np.real(e_complex)
        bg_galcat["g2"] = np.imag(e_complex)
    else:
        bg_galcat["g1"] = bg_galcat["g1_pure"]
        bg_galcat["g2"] = bg_galcat["g2_pure"]

    bg_galcat['w'] = np.ones(Ngal)

    return bg_galcat

def rotate_pix(pix, nside, rot_degrees):
    r = R.from_euler('zyx', rot_degrees, degrees=True)
    norm_vec_x, norm_vec_y, norm_vec_z = hp.pix2vec(nside=nside, ipix=pix)
    norm_vec = np.c_[norm_vec_x, norm_vec_y, norm_vec_z]
    new_vec = r.apply(norm_vec)
    pix_new = hp.vec2pix(nside=nside, x=new_vec[:,0], y=new_vec[:,1], z=new_vec[:,2])
    
    return pix_new
