'''
Utils used in making background samples
'''

import numpy as np
from scipy.spatial.transform import Rotation as R
import healpy as hp
from .io_func import bgal_type
def make_nofz(zctrs, nz):
    zedges = 0.5*(zctrs[1:] + zctrs[:-1])
    nz = nz[1:-1]
    nofz = {}
    nofz['zedges'] = zedges
    nofz['nz'] = nz

    return nofz

### sampling and add shape noise
def gen_gal_positions(ngal:float, mask:np.ndarray, nofz:dict) -> np.ndarray:
    # get RA DEC of footprint
    nside = hp.npix2nside(len(mask))
    ra, dec = hp.pix2ang(nside, np.argwhere(mask != 0).flatten(), lonlat=True)
    
    # get effective galaxy numbers
    sample_area = (ra.max() - ra.min()) * (dec.max() - dec.min())
    Ngal = int(np.around(ngal * sample_area * 60**2)) # ngal is in arcmin^-2
    ###   For test   ###
    Ngal = 1_000_000
    ####################
    
    # sample RA DEC
    sampled_ra = np.random.uniform(low=ra.min(), high=ra.max(), size=Ngal)
    cos_sampled_dec = np.random.uniform(low=np.cos(np.deg2rad(90-dec.min())), high=np.cos(np.deg2rad(90-dec.max())), size=Ngal)
    sampled_dec = np.rad2deg(np.arccos(cos_sampled_dec))
    sampled_dec = 90. - sampled_dec

    # ####################################################   For test   ###########################################################
    # sampled_ra = np.random.uniform(low=0, high=360, size=Ngal)
    # cos_sampled_dec = np.random.uniform(low=-1, high=1, size=Ngal)
    # sampled_dec = np.rad2deg(np.arccos(cos_sampled_dec))
    # sampled_dec = 90. - sampled_dec
    # #############################################################################################################################

    sample_pix = hp.ang2pix(nside, sampled_ra, sampled_dec, lonlat=True)
    picked_pix_in_sample = np.isin(sample_pix, np.argwhere(mask!=0).flatten())
    # #######################   For test   #######################
    # picked_pix_in_sample = np.ones(len(sample_pix)).astype(bool)
    ##############################################################

    Ngal = np.sum(picked_pix_in_sample)
    picked_ra = sampled_ra[picked_pix_in_sample]
    picked_dec = sampled_dec[picked_pix_in_sample]

    bg_galcat = np.empty(Ngal, dtype=bgal_type)
    bg_galcat['ra'] = picked_ra
    bg_galcat['dec'] = picked_dec

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
    bg_galcat['z'] = zsamples
    # #############   For test   #############
    # bg_galcat['z'] = np.ones(len(bg_galcat))
    # ########################################

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

        bg_galcat['g1'][select] = selected_g1
        bg_galcat['g2'][select] = selected_g2

    if sigma_e is not None:
        rng = np.random.default_rng(seed=seed)
        n1n2 = rng.normal(loc=0, scale=sigma_e, size=(Ngal,2))
        n_complex = n1n2[:,0] + n1n2[:,1]*1j
        del n1n2
        g_complex = bg_galcat["g1"] + bg_galcat["g2"]*1j
        e_complex = (g_complex + n_complex) / (1 + n_complex*np.conj(g_complex))
        bg_galcat["g1"] = np.real(e_complex)
        bg_galcat["g2"] = np.imag(e_complex)

    bg_galcat['w'] = np.ones(Ngal)

    return bg_galcat

def rotate_pix(pix, nside, rot_degrees):
    r = R.from_euler('zyx', rot_degrees, degrees=True)
    norm_vec_x, norm_vec_y, norm_vec_z = hp.pix2vec(nside=nside, ipix=pix)
    norm_vec = np.c_[norm_vec_x, norm_vec_y, norm_vec_z]
    new_vec = r.apply(norm_vec)
    pix_new = hp.vec2pix(nside=nside, x=new_vec[:,0], y=new_vec[:,1], z=new_vec[:,2])
    
    return pix_new





''' >>>===============================   Duplicated   ==================================<<< '''




# # >>>=================   sampling background galaxy   ======================<<<
# def uniform_spherical_sampling(N:int, lonlat:bool=True) -> tuple[np.ndarray, np.ndarray]:
#     '''
#     Generate uniform spherical sampling.

#     Parameters:
#     ----------
#     N: int
#         The number of samples.
#     lonlat: bool
#         Whether to return the samples in longitude and latitude. Default is True.

#     Returns:
#     -------
#     theta: np.ndarray
#         The polar angle of the samples. If `lonlat` is True, it is actually RA.
#     phi: np.ndarray
#         The azimuthal angle of the samples. If `lonlat` is True, it is actually DEC.
#     '''
#     indices = np.arange(0, N, dtype=float) + 0.5
#     theta = np.arccos(1 - 2 * indices / N)  # Polar angle
#     phi = 2 * np.pi * indices / N          # Azimuthal angle

#     if lonlat:
#         dec = np.rad2deg(theta)
#         ra  = np.rad2deg(phi)
#         dec = 90. - dec
#         return ra, dec
#     else:
#         return theta, phi

# def uniform_spherical_random_sampling(N:int, ra_minmax:list, dec_minmax:list, seed:int=None, lonlat:bool=True) -> tuple[np.ndarray, np.ndarray]:
#     '''
#     Generate uniform spherical sampling.

#     Parameters:
#     ----------
#     N: int
#         The number of samples.
#     ra_minmax: list
#         The minimum and maximum of RA.
#     dec_minmax: list
#         The minimum and maximum of DEC.
#     seed: int
#         The seed for the random number generator. Default is None.
#     lonlat: bool
#         Whether to return the samples in longitude and latitude. Default is True.

#     Returns:
#     -------
#     theta: np.ndarray
#         The polar angle of the samples. If `lonlat` is True, it is actually RA.
#     phi: np.ndarray
#         The azimuthal angle of the samples. If `lonlat` is True, it is actually DEC.
#     '''
#     rng = np.random.default_rng(seed=seed)

#     # phi = rng.uniform(0, 2 * np.pi, N)
#     # cos_theta = rng.uniform(-1, 1, N)
#     # theta = np.arccos(cos_theta)

#     phi = rng.uniform(np.deg2rad(ra_minmax[0]), np.deg2rad(ra_minmax[1]), N)
#     cos_theta = rng.uniform(np.cos(np.deg2rad(90. - dec_minmax[0])), np.cos(np.deg2rad(90. - dec_minmax[1])), N)
#     theta = np.arccos(cos_theta)

#     if lonlat:
#         dec = np.rad2deg(theta)
#         ra  = np.rad2deg(phi)
#         dec = 90. - dec
#         return ra, dec
#     else:
#         return theta, phi
    
# def construct_background_galaxies(
#         Ngal:int, redshift:float, sigma_redshift:float, 
#         g1_map:np.ndarray, g2_map:np.ndarray, mask:np.ndarray=None, weight:np.ndarray=None,
#         interp:str="bilinear", sigma_n:float=None, seed:int=None
#     ) -> np.ndarray:
#     '''
#     Construct background galaxies from shear map.

#     Parameters:
#     ----------
#     Ngal: int
#         The number of galaxies to generate.
#     redshift: float
#         The redshift of the galaxies.
#     sigma_redshift: float
#         The redshift uncertainty, sigma_z.
#     g1_map: np.ndarray
#         The HEALPix map of e1.
#     g2_map: np.ndarray
#         The HEALPix map of e2.
#     mask: np.ndarray
#         The HEALPix map of the mask. Can be float.
#     weight: np.ndarray
#         The weight of the galaxies considering observational systematics.
#         If is None, will be unity. This is used when analyzing simulations.
#     interp: str
#         The interpolation method to use. Default is "bilinear".
#     sigma_n: float
#         The uncertainty of shape measurements, sigma_e.
#     seed: int
#         The seed for the random number generator. Default is None.

#     Returns:
#     -------
#     gal_cat: np.ndarray
#         The catalog of background galaxies in SWOT format.
#     '''
    
#     if len(g1_map) != len(g2_map):
#         raise ValueError(f"Sizes of input gamma1 map ({len(g1_map)}) and gamma2 map ({len(g2_map)}) do not match!")
#     nside = hp.npix2nside(len(g1_map))

#     if mask is not None:
#         ### get ra_minmax and dec_minmax
#         mask_nside = hp.npix2nside(len(mask))
#         nomask_angles = hp.pix2ang(mask_nside, np.argwhere(mask != 0), lonlat=True)
#         ra_minmax = [np.min(nomask_angles[0]), np.max(nomask_angles[0])]
#         dec_minmax = [np.min(nomask_angles[1]), np.max(nomask_angles[1])]
#     else:
#         ra_minmax = [0, 360]
#         dec_minmax = [-90, 90]

#     gal_cat = np.empty((Ngal,), dtype=bgal_type)
#     gal_cat["ra"], gal_cat["dec"] = uniform_spherical_random_sampling(Ngal, ra_minmax, dec_minmax, lonlat=True)
#     ### remove the galaxies where mask is 0
#     if mask is not None:
#         gal_in_mask_pix = hp.ang2pix(mask_nside, gal_cat["ra"], gal_cat["dec"], lonlat=True)
#         gal_cat = gal_cat[mask[gal_in_mask_pix] != 0]
#         Ngal = len(gal_cat)

#     gal_cat["z"] = redshift*np.ones(Ngal)
#     gal_cat["sigz"] = sigma_redshift*np.ones(Ngal)
#     if interp == "ngp":
#         gal_pix = hp.ang2pix(nside, gal_cat["ra"], gal_cat["dec"], lonlat=True)
#         gal_cat["g1"]  = g1_map[gal_pix]
#         gal_cat["g2"]  = g2_map[gal_pix]
#     if interp == "bilinear":
#         gal_cat["g1"]  = hp.get_interp_val(g1_map, gal_cat["ra"], gal_cat["dec"], lonlat=True)
#         gal_cat["g2"]  = hp.get_interp_val(g2_map, gal_cat["ra"], gal_cat["dec"], lonlat=True)

#     if sigma_n is not None:
#         rng = np.random.default_rng(seed=seed)
#         n1n2 = rng.normal(loc=0, scale=sigma_n, size=(Ngal,2))
#         n_complex = n1n2[:,0] + n1n2[:,1]*1j
#         del n1n2
#         g_complex = gal_cat["g1"] + gal_cat["g2"]*1j
#         e_complex = (g_complex + n_complex) / (1 + n_complex*np.conj(g_complex))
#         gal_cat["g1"] = np.real(e_complex)
#         gal_cat["g2"] = np.imag(e_complex)

#     if weight is None :
#         gal_cat["w"] = np.ones((Ngal,))
#     else:
#         gal_cat["w"] = weight
#     return gal_cat