'''
Utils usd in constructing foreground samples
'''

import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial.transform import Rotation as R
import healpy as hp
import pyccl as ccl
from typing import Union, Tuple
from .hod_utils import *

# >>>======================   Calculate Halo Mass Function   =====================<<<
def get_abundance(data, bins=10, isedge=True, bin_scale='linear', density=False):
    if type(bins) is int:
        edges = bins
    if type(bins) is np.ndarray:
        if isedge:
            edges = bins
        else:
            if bin_scale == 'log':
                width = bins[1]/bins[0]
                edges = np.append(bins/np.sqrt(width),bins[-1]*np.sqrt(width))
            else:
                width = bins[1]-bins[0]
                edges = np.append(bins-width/2.,bins[-1]+width/2.)
    n, fedges = np.histogram(data, edges, density=density)
    return n, fedges

def get_HMF(mass, bins, boxsize, isedge=False, bin_scale='linear', ifcum=True):
    mctr_list = []
    logNm_list = []

    if type(mass) is not list:
        mass = [mass]
    if type(boxsize) is not list:
        boxsize = [boxsize]*len(mass)
    elif len(boxsize) != len(mass):
        raise ValueError("The size of boxsize list must be the same as mass list!")
    
    for imass, ibox in zip(mass, boxsize):
        ni, mi = get_abundance(imass, bins, isedge=isedge, bin_scale=bin_scale, density=False)
        if bin_scale == 'linear':
            mctr = (mi[1:] + mi[:-1])/2.
        else:
            mctr = np.sqrt(mi[1:]*mi[:-1])
        if ifcum:
            ilogNm = (np.cumsum(ni[::-1])[::-1]/ibox/ibox/ibox)
        else:
            ilogNm = (ni/ibox/ibox/ibox)

        mctr_list.append(mctr)
        logNm_list.append(ilogNm)

    if len(mctr_list) == 1:
        return mctr_list[0], logNm_list[0]
    else:
        return mctr_list, logNm_list
# >>>=============================================================================<<<

# >>>===========================   HOD help functions   ==========================<<<
def get_ngal(
        halo_mass, Lbox, redshift,
        model_lb, model_params_names, hod_param_vals, 
        ):
    '''
    Calculate theoretical predictions of ngal given HMF.
    '''
    if isinstance(hod_param_vals, list) or isinstance(hod_param_vals, np.ndarray):
        model_params_dict = dict(zip(model_params_names, hod_param_vals))
    elif isinstance(hod_param_vals, dict):
        model_params_dict = hod_param_vals.copy()
    else:
        raise ValueError("hod_param_vals should be list or dict")

    Mmin = halo_mass.min()
    Mmax = halo_mass.max()

    nMbin = 30
    dlgM = np.log10(Mmax/Mmin)/(nMbin-1)
    Mbin_edges = np.logspace(np.log10(Mmin)-dlgM, np.log10(Mmax)+dlgM, nMbin)

    massbin, NM = get_HMF(halo_mass, Mbin_edges, boxsize=Lbox, isedge=True, bin_scale='log', ifcum=False)

    tmp_dict = model_params_dict.copy()
    tmp_dict['fic'] = 1.0

    if model_lb == 0:
        ctr = MWCens(redshift=redshift)
    elif model_lb == 2 or model_lb == 3:
        ctr = MWCens_IC(redshift=redshift)
    elif model_lb == 4:
        ctr = ABMWCens_IC(redshift=redshift)

    ctr.param_dict = tmp_dict
    Nctr = ctr.mean_occupation(prim_haloprop=massbin)

    if model_lb == 0 or model_lb == 2:
        sat = MWSats(redshift=redshift, cenocc_model=ctr, modulate_with_cenocc=True)
    elif model_lb == 3:
        sat = MWSats2(redshift=redshift, cenocc_model=ctr, modulate_with_cenocc=True)
    elif model_lb == 4:
        sat = ABMWSats(redshift=redshift, cenocc_model=ctr, modulate_with_cenocc=True)

    sat.param_dict = tmp_dict
    Nsat = sat.mean_occupation(prim_haloprop=massbin)

    ngal_mock = (np.sum(Nctr*NM) + np.sum(Nsat*NM))

    Nsat_frac = Nsat.sum()/(Nctr+Nsat).sum()

    return ngal_mock, Nsat_frac

# >>>=============================================================================<<<


def Sph2Cart(cosmo_ccl:ccl.Cosmology, **kwargs) -> np.ndarray:
    hubble = cosmo_ccl.to_dict()['h']
    if 'pos' in kwargs.keys():
        ra = kwargs['pos'][:,0]
        dec = kwargs['pos'][:,1]
        z = kwargs['pos'][:,2]
    elif 'ra' in kwargs.keys() and 'dec' in kwargs.keys() and 'z' in kwargs.keys():
        ra = kwargs['ra']
        dec = kwargs['dec']
        z = kwargs['z']
    else:
        raise IOError("Need either `pos` or `ra dec z` as inputs!")
    
    chi_radial = ccl.comoving_radial_distance(cosmo_ccl, 1./(1+z)) # Mpc
    chi_radial *= hubble # Mpc/h
    pos = hp.ang2vec(ra, dec, lonlat=True) # Actually norm of position
    pos = (pos.T * chi_radial).T

    return pos

def Cart2Sph(cosmo_ccl:ccl.Cosmology, **kwargs) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hubble = cosmo_ccl.to_dict()['h']
    if 'pos' in kwargs.keys():
        pos = kwargs['pos']
    elif 'x' in kwargs.keys() and 'y' in kwargs.keys() and 'z' in kwargs.keys():
        pos = np.array([kwargs['x'], kwargs['y'], kwargs['z']])
    else:
        raise IOError("Need either `pos` or `x y z` as inputs!")
    
    chi_radial = np.linalg.norm(pos, axis=1) / hubble # Mpc
    phys_cut = (chi_radial > 1e-5) & (chi_radial < 5000)
    pos = pos[phys_cut]

    redshifts = 1./ccl.scale_factor_of_chi(cosmo_ccl, chi_radial[phys_cut]) - 1.
    ra, dec = hp.vec2ang(pos, lonlat=True)

    return ra, dec, redshifts, phys_cut

# >>>===================   Constructing foreground lightcone   ===================<<<
def box_recenter(pos:np.ndarray, center:Union[tuple,list,np.ndarray], boxsize:float) -> np.ndarray:
    '''
    Re-center the coordinates `pos` to a box centered at `center` with edge length `boxsize`.
    
    Parameters:
    ----------
    pos: np.ndarray
        The original coordinates to be re-centered.
    center: tuple or list or np.ndarray
        The center of the box. Should be in the format of (x, y, z).
    boxsize: float
        The size of the box edge.
    
    Returns:
    -------
    local_pos: np.ndarray
        The re-centered coordinates.
    '''
    local_pos = ((pos-np.array(center)*boxsize)+boxsize) % boxsize
    return local_pos

def push_box(pos, shift, boxsize):
    local_pos = (pos + boxsize) % boxsize
    local_pos += shift*boxsize

    return local_pos
    
def cut_shell_one_box(pos:np.ndarray, gid:np.ndarray, 
                      boxsize:float, 
                      shift:Union[tuple,list,np.ndarray], 
                      rmin:float, rmax:float,
                      other_props:list=None
                      ) -> Union[np.ndarray, tuple]:
    '''
    Cut out a shell within a given range from the coordinates.

    Parameters:
    ----------
    pos: np.ndarray
        The original coordinates to be cut.
    gid: np.ndarray
        The global galaxy IDs to be cut.
    boxsize: float
        The size of the box edge.
    shift: tuple or list or np.ndarray
        The shift to be applied to the coordinates.
    rmin: float
        The minimum radius of the shell.
    rmax: float
        The maximum radius of the shell.
    other_props: [Optional] list of np.ndarray
        Other properties to be transform simutaneously.
        Default is None.

    Returns:
    -------
    local_pos: np.ndarray
        The coordinates after cutting out the shell.
    local_prop: np.ndarray
        Other properties after cutting out the shell.
    '''

    # local_pos = (pos + boxsize) % boxsize
    # local_pos += shift*boxsize
    local_pos = push_box(pos, shift, boxsize)
    
    cut = ((np.linalg.norm(local_pos,axis=1)>rmin)&(np.linalg.norm(local_pos,axis=1)<rmax))
    local_pos = local_pos[cut]
    local_gid = gid[cut]

    if other_props is not None:
        local_props = []
        if not isinstance(other_props, list):
            raise ValueError("`other_props` should be a list of np.ndarray!")
        for iprop in other_props:
            local_props.append(iprop[cut])

        return np.c_[local_pos, local_gid], local_props

    else:
        return np.c_[local_pos, local_gid]

def _expand_box_indices(indices_positive):
    """
    Expand box indices in the positive octant to all eight octants.

    A positive-octant box index i represents [i, i + 1].
    Its mirror image is therefore [-i - 1, -i].
    """
    if indices_positive.shape[0] == 0:
        return np.empty((0, 3), dtype=np.int64)

    signs = np.array(
        np.meshgrid(
            [-1, 1],
            [-1, 1],
            [-1, 1],
            indexing="ij",
        )
    ).reshape(3, -1).T

    expanded_indices = np.where(
        signs[None, :, :] > 0,
        indices_positive[:, None, :],
        -indices_positive[:, None, :] - 1,
    )

    expanded_indices = expanded_indices.reshape(-1, 3)

    return np.unique(expanded_indices, axis=0).astype(np.int64)


def get_cross_box_indice(boxsize:float, chi_min:float, chi_max:float) -> np.ndarray:
    """
    Search boxes associated with the spherical shell

        chi_min <= chi <= chi_max.

    Boxes are divided into two groups:

    1. crossing_indices:
       Boxes crossed by the chi_min or chi_max spherical surface.
       Particles in these boxes must be filtered individually.

    2. inside_indices:
       Boxes lying completely inside the spherical shell.
       All particles in these boxes can be retained directly.

    Parameters
    ----------
    boxsize : float
        Side length of one box.
    chi_min : float
        Inner radius of the spherical shell.
    chi_max : float
        Outer radius of the spherical shell.

    Returns
    -------
    crossing_indices : ndarray of shape (N_crossing, 3)
        Indices of boxes crossing either spherical boundary.

    inside_indices : ndarray of shape (N_inside, 3)
        Indices of boxes completely inside the spherical shell.
    """
    if boxsize <= 0:
        raise ValueError("boxsize must be positive.")

    if chi_min < 0:
        raise ValueError("chi_min must be non-negative.")

    if chi_max < chi_min:
        raise ValueError("chi_max must be greater than or equal to chi_min.")

    # Work in units of boxsize.
    radius_min = chi_min / boxsize
    radius_max = chi_max / boxsize

    # Search the positive octant first.
    nmax = int(np.ceil(radius_max))

    if nmax == 0:
        empty = np.empty((0, 3), dtype=np.int64)
        return empty, empty.copy()

    search_indices_1d = np.arange(nmax, dtype=np.int64)

    ix, iy, iz = np.meshgrid(
        search_indices_1d,
        search_indices_1d,
        search_indices_1d,
        indexing="ij",
    )

    search_indices = np.column_stack(
        (
            ix.ravel(),
            iy.ravel(),
            iz.ravel(),
        )
    )

    # For a positive-octant box [i, i+1] × [j, j+1] × [k, k+1]:
    #
    # nearest corner  = [i, j, k]
    # farthest corner = [i+1, j+1, k+1]
    #
    # Squared distances are sufficient, avoiding sqrt.
    nearest_dist_sq = np.sum(
        search_indices.astype(float) ** 2,
        axis=1,
    )

    farthest_dist_sq = np.sum(
        (search_indices.astype(float) + 1.0) ** 2,
        axis=1,
    )

    radius_min_sq = radius_min**2
    radius_max_sq = radius_max**2

    # Completely contained in the shell.
    inside_choice = (
        (nearest_dist_sq >= radius_min_sq)
        & (farthest_dist_sq <= radius_max_sq)
    )

    # Crosses the inner spherical boundary.
    cross_inner_choice = (
        (nearest_dist_sq < radius_min_sq)
        & (farthest_dist_sq > radius_min_sq)
    )

    # Crosses the outer spherical boundary.
    cross_outer_choice = (
        (nearest_dist_sq < radius_max_sq)
        & (farthest_dist_sq > radius_max_sq)
    )

    crossing_choice = cross_inner_choice | cross_outer_choice

    crossing_positive = search_indices[crossing_choice]
    inside_positive = search_indices[inside_choice]

    # Expand positive-octant indices to all octants.
    crossing_indices = _expand_box_indices(crossing_positive)
    inside_indices = _expand_box_indices(inside_positive)

    return crossing_indices, inside_indices

def make_lightcone_tiles(
        position:np.ndarray, boxsize:float, 
        chi_min:float, chi_max:float, 
        ctr:Union[tuple,list,np.ndarray,int]=[0,0,0],
        other_props:list=None) -> Union[np.ndarray, tuple]:
    '''
    Make a lightcone from the given coordinates.

    Parameters:
    ----------
    position: np.ndarray
        The positions of the particles. Should be in Cartesian coordinates.
    boxsize: float
        The size of the box edge.
    chi_min: float
        The minimum radius of the slice.
    chi_max: float
        The maximum radius of the slice.
    ctr: tuple or list or np.ndarray or int
        The center of the box. Default is (0,0,0).
        If is int, will be broadcast to three dimensions.
    other_props: [Optional] list of np.ndarray
        Other properties of the galaxy, for example velocities.
        Default is None.

    Returns:
    -------
    lightcone: np.ndarray
        The positions of particles in the lightcone.
    lightcone_prop: np.ndarray
        Other properties of particles in the lightcone.
    '''
    if other_props is not None:
        if not isinstance(other_props, list) and isinstance(other_props, np.ndarray):
            other_props = [other_props]
        for iprop, other_prop in enumerate(other_props):
            if other_prop.shape[0] != position.shape[0]:
                raise ValueError(f"The shape of the position ({position.shape[0]}) and the {iprop}-th property ({other_prop.shape[0]}) do not match!")
    
    num_props = len(other_props)

    if type(ctr) is int:
        ctr = [ctr]*3
    crossing_indice, inside_indice = get_cross_box_indice(boxsize, chi_min, chi_max)
    pos_rectr = box_recenter(position, ctr, boxsize)
    gid = np.arange(len(position)) # Global ID of each tracer, start from 0
    
    lightcone = []
    
    if other_props is not None:
        tmp_lightcone_props = {}
        for iprop in range(num_props):
            tmp_lightcone_props[f"prop{iprop}"] = []

        for idx in range(len(crossing_indice)):
            tmp_pos_id, tmp_prop = cut_shell_one_box(pos_rectr, gid, boxsize, crossing_indice[idx], chi_min, chi_max, other_props=other_props)
            lightcone.append(tmp_pos_id)
            for iprop in range(num_props):
                tmp_lightcone_props[f"prop{iprop}"].append(tmp_prop[iprop])
        
        for idx in range(len(inside_indice)):
            tmp_pos_id = np.c_[push_box(pos_rectr, inside_indice[idx], boxsize), gid]
            tmp_prop = other_props
            lightcone.append(tmp_pos_id)
            for iprop in range(num_props):
                tmp_lightcone_props[f"prop{iprop}"].append(tmp_prop[iprop])
        
        lightcone = np.vstack(lightcone)
        lightcone_props = []

        for iprop in range(num_props):
            # print(tmp_lightcone_props[f"prop{iprop}"][0].shape, tmp_lightcone_props[f"prop{iprop}"][1].shape)
            lightcone_props.append(np.concatenate(tmp_lightcone_props[f"prop{iprop}"]))
        # lightcone_prop = np.vstack(lightcone_prop)

        return lightcone, lightcone_props

    else:   
        for idx in range(len(crossing_indice)):
            tmp_pos_id = cut_shell_one_box(pos_rectr, gid, boxsize, crossing_indice[idx], chi_min, chi_max)
            lightcone.append(tmp_pos_id)

        for idx in range(len(inside_indice)):
            tmp_pos_id = np.c_[push_box(pos_rectr, inside_indice[idx], boxsize), gid]
            lightcone.append(tmp_pos_id)
        lightcone = np.vstack(lightcone)

        return lightcone

def cat2shell(nside:int, **kwargs):
    '''
    Convert a catalog to a HEALPix shell.

    Parameters:
    ----------
    nside: int
        The HEALPix Nside parameter.
    **kwargs: dict
        The keyword arguments for the shell function.
        Can be: 
        - pos: np.ndarray, (N,3) representing (x,y,z)
               in Cartesian coordinates.
        - (ra,dec): np.ndarray, (N,) representing RA (DEC)
               in Spherical coordinates.
        - (theta,phi): np.ndarray, (N,) representing theta (phi)
               in Spherical coordinates.

    Returns:
    -------
    shell: np.ndarray
        The HEALPix shell.
    '''

    if "pos" in kwargs.keys():
        pos = kwargs["pos"]
        part_pix = hp.vec2pix(nside, pos[:,0], pos[:,1], pos[:,2])
    
    elif "ra" in kwargs.keys() and "dec" in kwargs.keys():
        ra = kwargs["ra"]
        dec = kwargs["dec"]
        part_pix = hp.ang2pix(nside, ra, dec, lonlat=True)

    elif "theta" in kwargs.keys() and "phi" in kwargs.keys():
        theta = kwargs["theta"]
        phi = kwargs["phi"]
        part_pix = hp.ang2pix(nside, theta, phi, lonlat=False)

    else:
        raise ValueError("Should provide either [pos] or (ra,dec) or (theta,phi)!")
        
    shell = np.zeros(12*nside**2).astype(np.int64)
    np.add.at(shell, part_pix, 1)

    return shell

def rotate_lightcone(galcone, rot_degrees, inv=False, icoord='radec'):
    '''
    Apply rotation on lightcone particles.

    Parameters:
    ----------
    galcone: np.ndarray
        The lightcone particles in (x,y,z).
    rot_degrees: list or np.ndarray
        The rotation angle in degrees.
    inv: bool
        Whether to apply the inverse rotation. Default is True.
    icoord: str
        The coordinate system of the input lightcone. 
        Can be `vec` or `radec`. Default is 'radec'.

    Returns:
    -------
    galcone_rot: np.ndarray
        The rotated lightcone particles in (x,y,z).
    '''

    if icoord != 'vec' and icoord != 'radec':
        raise ValueError("icoord should be either 'vec' or 'radec'!")

    r = R.from_euler('zyx', rot_degrees, degrees=True)
    if inv:
        r = r.inv()
    
    if icoord == 'vec':
        galcone_rot = r.apply(galcone)
    if icoord == 'radec':
        galcone_rot = galcone.copy()
        galcone_rot['ra'] , galcone_rot['dec'] = hp.rotator.rotateDirection(
            rotmat=r.as_matrix(),
            theta=galcone['ra'],
            phi=galcone['dec'],
            lonlat=True
        )
        galcone_rot['ra'] = np.where(
            galcone_rot['ra'] < 0,
            galcone_rot['ra'] + 360,
            galcone_rot['ra']
        )

    return galcone_rot

# >>>=============================================================================<<<

# >>>============   Apply survey geometry and radial selection  ==================<<<

def apply_boss_geometry(galcone, geom_polygon, masks):
    mask = geom_polygon.contains(galcone["ra"], galcone["dec"])
    galcone_boss = galcone[mask]
    galcone_boss["w"] = geom_polygon.weight(galcone_boss["ra"], galcone_boss["dec"])
    
    select = galcone_boss["w"] > 0
    galcone_boss = galcone_boss[select]

    for ipoly in range(len(masks)):
        mask = masks[ipoly].contains(galcone_boss["ra"], galcone_boss["dec"])
        tot_mask = mask if ipoly == 0 else tot_mask | mask
    galcone_boss = galcone_boss[~tot_mask]

    return galcone_boss

def apply_boss_lowze2e3_trim(galcone, lowz_polygon):
    trim = lowz_polygon.weight(galcone["ra"], galcone["dec"]) == 0
    galcone_trimmed = galcone[trim]

    return galcone_trimmed

def apply_2dflens_geometry(galcone, mask_weight_maps, interp=True):
    mask_map = mask_weight_maps[0]
    weight_map = mask_weight_maps[1]
    ### apply mask
    nside = hp.npix2nside(len(mask_map))

    galcone_pix = hp.ang2pix(nside, galcone["ra"], galcone["dec"], lonlat=True)
    select = mask_map[galcone_pix] > 0
    galcone_2dflens = galcone[select]

    if interp:
        galcone_2dflens["w"] = hp.get_interp_val(weight_map, galcone_2dflens["ra"], galcone_2dflens["dec"], lonlat=True)
    else:
        galcone_2dflens["w"] = weight_map[galcone_pix]
    
    select = galcone_2dflens["w"] > 0
    galcone_2dflens = galcone_2dflens[select]

    return galcone_2dflens

def apply_nz(galcone, nofz_info, nofz_method, norm=False, add_rsd=False):
    zedges = nofz_info['zedges']
    shell_vol = nofz_info['shell_vol']
    nz_ref = nofz_info['nz_ref']

    if add_rsd:
        z_mock = galcone["zrsd"]
    else:
        z_mock = galcone["z"]
        
    Nz_mock, _ = np.histogram(z_mock, zedges)
    Nz_target = nz_ref * shell_vol
    downsample_rate = Nz_target / Nz_mock
    downsample_rate = np.nan_to_num(downsample_rate, nan=0.0, posinf=0.0, neginf=0.0)
    downsample_rate = np.clip(downsample_rate, 0, 1)

    if norm:
        downsample_rate /= np.max(downsample_rate)

    downsample_rate = np.clip(downsample_rate, 0, 1)
    
    galcone_dsampled = []

    for ibin in range(len(zedges)-1):
        zmin, zmax = zedges[ibin], zedges[ibin+1]
        in_bin = (z_mock >= zmin) & (z_mock < zmax)
        gal_in_bin = galcone[in_bin]

        if len(gal_in_bin) != 0:
            if nofz_method == "downsample":
                mask = np.random.choice(np.arange(len(gal_in_bin)), size=int(downsample_rate[ibin]*len(gal_in_bin)), replace=False)
            if nofz_method == "rank":
                ### descending rank of gal in bin
                ranked_idx = np.argsort(gal_in_bin, order="host_halo_mvir")[::-1]
                mask = ranked_idx[:Nz_target[ibin]]

            galcone_dsampled.append(gal_in_bin[mask])
            if nofz_method == "const":
                pass

    galcone_dsampled = np.concatenate(galcone_dsampled)

    return galcone_dsampled

def make_nofz_info(nofz_info, survey_name, zedges, shell_vol, nz_ref):
    nofz_info[survey_name] = {}
    nofz_info[survey_name]['zedges'] = zedges
    nofz_info[survey_name]['shell_vol'] = shell_vol
    nofz_info[survey_name]['nz_ref'] = nz_ref

    return nofz_info

def make_nofz_from_sample(samples:np.ndarray, bins:Union[int, list, tuple]=30, rng:tuple=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate histogram (x, pdf) from given sample.

    Parameters:
    ----------
    samples : array-like
        Array of samples.
    bins : int or sequence, optional
        Bins or edges of histogram
    range : tuple, optional
        (min, max), minimum and maximum of histogram range
    
    Return: q
    ----------
    x : ndarray
        Bin center
    pdf : ndarray
        Normalized pdf
    """
    counts, bin_edges = np.histogram(samples, bins=bins, range=rng, density=False)
    bin_widths = np.diff(bin_edges)
    x = 0.5 * (bin_edges[:-1] + bin_edges[1:])  # bin center
    pdf = counts / (np.sum(counts) * bin_widths)  # normalization

    return x, pdf

def sample_from_histogram(N:int, x:np.ndarray, pdf:np.ndarray) -> np.ndarray:
    """
    Sampling N samples from given histogram (x, pdf).

    Parameters:
    ----------
    N : int
        Number of samples.
    x : array-like
        Bin center of histogram
    pdf : array-like
        Pdf, can be normalized or not
    
    Return:
    ----------
    samples : ndarray
        Sampled array.
    """
    x = np.asarray(x)
    pdf = np.asarray(pdf)

    # 归一化 pdf
    pdf = pdf / np.sum(pdf)

    # 计算CDF
    cdf = np.cumsum(pdf)
    # cdf[-1] = 1.0  # 确保最后一个是1
    cdf /= cdf[-1]

    # 在[0,1]生成N个随机数
    u = np.random.rand(N)

    # 通过CDF反演找到所在区间
    indices = np.searchsorted(cdf, u)

    # 在每个bin内做线性插值
    bin_width = np.diff(x).mean()  # 假设等宽
    samples = x[indices] + (np.random.rand(N) - 0.5) * bin_width

    return samples

def logit_transform(x, a, b):
    a -= 1e-5
    b += 1e-5
    return np.log((x - a) / (b - x))

def logit_inverse(y, a, b):
    a -= 1e-5
    b += 1e-5
    ey = np.exp(y)
    return (a + b * ey) / (1 + ey)

def bounded_kde_transform(data, bounds):
    (xmin, xmax), (ymin, ymax) = bounds
    # 对每个维度做 logit 变换
    tx = logit_transform(data[:,0], xmin, xmax)
    ty = logit_transform(data[:,1], ymin, ymax)
    tdata = np.vstack([tx, ty])
    kde = gaussian_kde(tdata)
    return kde

def resample_bounded(kde, N, bounds):
    """
    从 bounded KDE 采样
    """
    (xmin, xmax), (ymin, ymax) = bounds
    t_samples = kde.resample(N).T
    xs = logit_inverse(t_samples[:,0], xmin, xmax)
    ys = logit_inverse(t_samples[:,1], ymin, ymax)
    return xs, ys

# >>>=============================================================================<<<

# >>>==============================  find voids  =================================<<<

def find_void(tracer_pos, boxsize=None, 
              exec_path="/home/suchen/applications/DIVE/DIVE", 
              dive_input="./tmp_tracer.dat",
              dive_output="./tmp_void.dat"):

    np.savetxt(dive_input, tracer_pos, fmt='%.3f')

    cmd = exec_path + " -i " + dive_input + " -o " + dive_output
    if boxsize is not None:
        cmd += " -u " + str(boxsize)
    print(cmd)
    os.system(cmd)
    print(f"rm {dive_input}")
    os.system(f"rm {dive_input}")

    void_info = np.loadtxt(dive_output)
    void_pos = void_info[:,:-1]
    void_radius = void_info[:,-1]

    print(f"rm {dive_output}")
    os.system(f"rm {dive_output}")

    return void_pos, void_radius