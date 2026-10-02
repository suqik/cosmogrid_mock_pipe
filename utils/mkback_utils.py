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
    nz = np.asarray(nz, dtype=float)[1:-1]
    if not np.all(np.isfinite(nz)) or np.any(nz < 0):
        raise ValueError("n(z) values must be finite and non-negative")
    normalization = nz.sum()
    if normalization <= 0:
        raise ValueError("n(z) must contain positive probability")
    nz = nz / normalization
    nofz = {}
    nofz['zedges'] = zedges
    nofz['nz'] = nz

    return nofz

def gen_angle_positions_from_healpix(ngal:float, mask:np.ndarray):
    """Sample an exact surface density inside a HEALPix footprint.

    ``ngal`` is in arcmin^-2.  The requested catalog size is derived from
    the area of the non-zero mask pixels, rather than from a rectangular
    RA/Dec bounding box.  Candidate positions are drawn uniformly in solid
    angle from the smallest of the ordinary and RA-wrapped bounding boxes
    and rejected when they fall outside the mask.
    """
    nside = hp.npix2nside(len(mask))
    footprint_pixels = np.flatnonzero(mask != 0)
    if len(footprint_pixels) == 0:
        raise ValueError("HEALPix footprint contains no usable pixels")

    pixel_area_deg2 = hp.nside2pixarea(nside) * (180.0 / np.pi) ** 2
    footprint_area_deg2 = len(footprint_pixels) * pixel_area_deg2
    target_count = int(np.rint(ngal * footprint_area_deg2 * 60.0**2))
    if target_count == 0:
        return np.empty(0), np.empty(0)

    pixel_ra, pixel_dec = hp.pix2ang(
        nside, footprint_pixels, lonlat=True
    )
    raw_bounds = (float(pixel_ra.min()), float(pixel_ra.max()))
    shifted_ra = np.where(pixel_ra < 180.0, pixel_ra + 360.0, pixel_ra)
    shifted_bounds = (float(shifted_ra.min()), float(shifted_ra.max()))
    if shifted_bounds[1] - shifted_bounds[0] < raw_bounds[1] - raw_bounds[0]:
        ra_min, ra_max = shifted_bounds
    else:
        ra_min, ra_max = raw_bounds

    padding = np.rad2deg(hp.max_pixrad(nside))
    ra_min -= padding
    ra_max += padding
    ra_width = ra_max - ra_min
    if ra_width >= 360.0:
        ra_min = 0.0
        ra_width = 360.0
    dec_min = max(-90.0, float(pixel_dec.min()) - padding)
    dec_max = min(90.0, float(pixel_dec.max()) + padding)

    sin_dec_min = np.sin(np.deg2rad(dec_min))
    sin_dec_max = np.sin(np.deg2rad(dec_max))
    bounding_area_deg2 = (
        np.deg2rad(ra_width)
        * (sin_dec_max - sin_dec_min)
        * (180.0 / np.pi) ** 2
    )
    acceptance = min(1.0, footprint_area_deg2 / bounding_area_deg2)

    ra_parts = []
    dec_parts = []
    remaining = target_count
    max_batch = 5_000_000
    coordinate_dtype = bgal_type.fields["ra"][0]
    while remaining:
        batch_size = min(
            max_batch,
            max(1_000, int(np.ceil(1.1 * remaining / acceptance))),
        )
        sampled_ra = (
            ra_min + np.random.uniform(size=batch_size) * ra_width
        ) % 360.0
        sampled_sin_dec = np.random.uniform(
            low=sin_dec_min, high=sin_dec_max, size=batch_size
        )
        sampled_ra = sampled_ra.astype(coordinate_dtype)
        sampled_ra %= coordinate_dtype.type(360.0)
        sampled_dec = np.rad2deg(np.arcsin(sampled_sin_dec)).astype(
            coordinate_dtype
        )
        sample_pix = hp.ang2pix(
            nside, sampled_ra, sampled_dec, lonlat=True
        )
        accepted = mask[sample_pix] != 0
        take = min(remaining, int(accepted.sum()))
        if take:
            accepted_indices = np.flatnonzero(accepted)[:take]
            ra_parts.append(sampled_ra[accepted_indices])
            dec_parts.append(sampled_dec[accepted_indices])
            remaining -= take

    return np.concatenate(ra_parts), np.concatenate(dec_parts)

def gen_angle_positions_from_mangle(ngal:float, mask:pymangle.Mangle):
    sample_area = (mask.areas * mask.weights).sum() # deg^2
    Ngal = int(np.around(ngal * sample_area * 60**2))
    sampled_ra, sampled_dec = mask.genrand(Ngal)
    sample_weights = mask.weight(sampled_ra, sampled_dec)
    picked_idx = (sample_weights > 0)
    picked_ra = sampled_ra[picked_idx]
    picked_dec = sampled_dec[picked_idx]

    return picked_ra, picked_dec

def gen_redshifts_from_nofz(Ngal:float, nofz:dict|float|list, photo_z_err=None, seed=None, zmax=None):
    if isinstance(nofz, dict):
        zedges = np.asarray(nofz['zedges'], dtype=float)
        nz = np.asarray(nofz['nz'], dtype=float)
        upper_edges = zedges[1:].copy()

        if zmax is not None:
            widths = np.diff(zedges)
            covered_widths = np.clip(
                np.minimum(upper_edges, zmax) - zedges[:-1],
                0.0,
                widths,
            )
            nz = nz * covered_widths / widths
            upper_edges = np.minimum(upper_edges, zmax)
        normalization = nz.sum()
        if normalization <= 0:
            raise ValueError("n(z) has no probability in the requested range")
        probabilities = nz / normalization
        counts = np.random.multinomial(int(Ngal), probabilities)
        zsamples = np.concatenate([
            np.random.uniform(
                low=zedges[i], high=upper_edges[i], size=count
            )
            for i, count in enumerate(counts)
            if count
        ])

    elif isinstance(nofz, float):
        zsamples = np.ones(int(Ngal)) * nofz
    elif isinstance(nofz, list):
        zsamples = np.random.uniform(
            low=nofz[0], high=nofz[1], size=int(Ngal)
        )
    else:
        raise TypeError("nofz must be a dictionary, float, or list")

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
