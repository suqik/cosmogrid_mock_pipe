'''
Utils usd in constructing foreground samples
'''

import numpy as np
from scipy.stats import gaussian_kde
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
    
def cut_shell_one_box(pos:np.ndarray, gid:np.ndarray, boxsize:float, shift:Union[tuple,list,np.ndarray], rmin:float, rmax:float) -> np.ndarray:
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

    Returns:
    -------
    local_pos: np.ndarray
        The coordinates after cutting out the shell.
    '''
    # local_pos = (pos + boxsize) % boxsize
    # local_pos += shift*boxsize
    
    # # cut = ((radial_dist(local_pos)>rmin)&(radial_dist(local_pos)<rmax))
    # cut = ((np.linalg.norm(local_pos,axis=1)>rmin)&(np.linalg.norm(local_pos,axis=1)<rmax))
    # local_pos = local_pos[cut]
    # del cut
    
    # return local_pos

    local_pos = (pos + boxsize) % boxsize
    local_pos += shift*boxsize
    
    cut = ((np.linalg.norm(local_pos,axis=1)>rmin)&(np.linalg.norm(local_pos,axis=1)<rmax))
    local_pos = local_pos[cut]
    local_gid = gid[cut]
    del cut
    
    return np.c_[local_pos, local_gid]

def search_cross_box(boxsize:float, radius:float) -> np.ndarray:
    '''
    Search for all possible box crossing points within a given radius.
    
    Parameters:
    ----------
    boxsize: float
        The size of the box edge.
    radius: float
        The radius of the crossing ring.
    
    Returns:
    -------
    unique_indice: np.ndarray
        An array of indices representing the possible box crossing points.
    '''
    nmax = int(np.ceil(radius/boxsize))
    search_indice_1d = np.arange(nmax)
    search_indice = np.array(np.meshgrid(search_indice_1d, search_indice_1d, search_indice_1d)).T.reshape(-1,3)
    nearest_dist_sq = np.sqrt(np.sum(search_indice**2, axis=1))
    farest_dist_sq  = np.sqrt(np.sum((search_indice+1)**2, axis=1))
    choice = ((nearest_dist_sq<radius/boxsize)&(farest_dist_sq>radius/boxsize))
    search_indice = search_indice[choice]

    search_indice = search_indice.astype(float) + 0.5
    signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
    expand_indice = (search_indice[:, None, :]*signs[None, :, :]).reshape(-1, 3)
    unique_indice = np.unique(expand_indice, axis=0)
    unique_indice = (np.floor(unique_indice-0.5)).astype(int)
    
    return unique_indice

def get_cross_box_indice(boxsize:float, chi_min:float, chi_max:float) -> np.ndarray:
    '''
    Get the indices of all possible box crossing points within a given range.

    Parameters:
    ----------
    boxsize: float
        The size of the box edge.
    chi_min: float
        The minimum radius of the crossing ring.
    chi_max: float
        The maximum radius of the crossing ring.

    Returns:
    -------
    indice_all: np.ndarray
        An array of indices representing the possible box crossing points.

    '''
    indice_min = search_cross_box(boxsize, chi_min)
    indice_max = search_cross_box(boxsize, chi_max)
    indice_all = np.unique(np.vstack((indice_min, indice_max)), axis=0)

    return indice_all

def make_lightcone_tiles(position:np.ndarray, boxsize:float, chi_min:float, chi_max:float, ctr:Union[tuple,list,np.ndarray,int]=[0,0,0]) -> np.ndarray:
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

    Returns:
    -------
    lightcone: np.ndarray
        The positions of particles in the lightcone.
    '''
    if type(ctr) is int:
        ctr = [ctr]*3
    shift_list = get_cross_box_indice(boxsize, chi_min, chi_max)
    pos_rectr = box_recenter(position, ctr, boxsize)
    gid = np.arange(len(position)) # Global ID of each tracer, start from 0
    
    lightcone = []
    for idx in range(len(shift_list)):
        tmp = cut_shell_one_box(pos_rectr, gid, boxsize, shift_list[idx], chi_min, chi_max)
        lightcone.append(tmp)
    lightcone = np.vstack(lightcone)

    return lightcone

def cat2shell(pos:np.ndarray, Nside:int, coord="cart"):
    '''
    Convert a catalog to a HEALPix shell.

    Parameters:
    ----------
    pos: np.ndarray
        The positions of the particles.
    Nside: int
        The HEALPix Nside parameter.
    coord: str
        The coordinate system of the positions. 
        Can be `cart`, `sph` or `lonlat`.
        Default is `cart`.

    Returns:
    -------
    shell: np.ndarray
        The HEALPix shell.
    '''
    if coord == "cart":
        if pos.shape[1] != 3:
            raise ValueError("Cartesian coordinate must have 3 components!")
        part_pix = hp.vec2pix(Nside, pos[:,0], pos[:,1], pos[:,2])
    if coord == "sph":
        if pos.shape[1] != 2:
            raise ValueError("Spherical coordinate must have 2 components!")
        part_pix = hp.ang2pix(Nside, pos[:,0], pos[:,1], lonlat=False)
    if coord == "lonlat":
        if pos.shape[1] != 2:
            raise ValueError("RADEC coordinate must have 2 components!")
        part_pix = hp.ang2pix(Nside, pos[:,0], pos[:,1], lonlat=True)
        
    shell = np.zeros(12*Nside**2).astype(np.int64)
    np.add.at(shell, part_pix, 1)

    return shell

# >>>=============================================================================<<<

# >>>============   Apply survey geometry and radial selection  ==================<<<

def apply_boss_geometry(galcone, geom_polygon, masks, galcone_ids=None):
    mask = geom_polygon.contains(galcone["ra"], galcone["dec"])
    galcone_boss = galcone[mask]
    galcone_boss["w"] = geom_polygon.weight(galcone_boss["ra"], galcone_boss["dec"])
    
    select = galcone_boss["w"] > 0
    galcone_boss = galcone_boss[select]

    if galcone_ids is not None:
        galcone_ids_out = galcone_ids[mask][select]
    else:
        galcone_ids_out = None

    for ipoly in range(len(masks)):
        mask = masks[ipoly].contains(galcone_boss["ra"], galcone_boss["dec"])
        tot_mask = mask if ipoly == 0 else tot_mask | mask
    galcone_boss = galcone_boss[~tot_mask]

    if galcone_ids is not None:
        galcone_ids_out = galcone_ids_out[~tot_mask]

    return galcone_boss, galcone_ids_out

def apply_boss_lowze2e3_trim(galcone, lowz_polygon, galcone_ids=None):
    trim = lowz_polygon.weight(galcone["ra"], galcone["dec"]) == 0
    galcone_trimmed = galcone[trim]
    if galcone_ids is not None:
        galcone_ids_out = galcone_ids[trim]
    else:
        galcone_ids_out = None
    return galcone_trimmed, galcone_ids_out

def apply_2dflens_geometry(galcone, mask_weight_maps, interp=True, galcone_ids=None):
    mask_map = mask_weight_maps[0]
    weight_map = mask_weight_maps[1]
    ### apply mask
    nside = hp.npix2nside(len(mask_map))

    galcone_pix = hp.ang2pix(nside, galcone["ra"], galcone["dec"], lonlat=True)
    select = mask_map[galcone_pix] > 0
    galcone_2dflens = galcone[select]
    
    if galcone_ids is not None:
        galcone_ids_out = galcone_ids[select]
    else:
        galcone_ids_out = None

    if interp:
        galcone_2dflens["w"] = hp.get_interp_val(weight_map, galcone_2dflens["ra"], galcone_2dflens["dec"], lonlat=True)
    else:
        galcone_2dflens["w"] = weight_map[galcone_pix]
    
    select = galcone_2dflens["w"] > 0
    galcone_2dflens = galcone_2dflens[select]

    if galcone_ids is not None:
        galcone_ids_out = galcone_ids_out[select]

    return galcone_2dflens, galcone_ids_out

def apply_nz_downsample(galcone, nofz_info, galcone_ids=None):
    zedges = nofz_info['zedges']
    shell_vol = nofz_info['shell_vol']
    nz_ref = nofz_info['nz_ref']

    z_mock = galcone["z"]
    Nz_mock, _ = np.histogram(z_mock, zedges)
    downsample_rate = nz_ref/Nz_mock*shell_vol

    downsample_rate = np.clip(downsample_rate, 0, 1)
    
    galcone_dsampled = []
    if galcone_ids is not None:
        galcone_ids_dsampled = []
    else:
        galcone_ids_dsampled = None
    number_in_bin = []
    for ibin in range(len(zedges)-1):
        zmin, zmax = zedges[ibin], zedges[ibin+1]
        in_bin = (z_mock >= zmin) & (z_mock < zmax)
        gal_in_bin = galcone[in_bin]

        mask = np.random.choice(np.arange(len(gal_in_bin)), size=int(downsample_rate[ibin]*len(gal_in_bin)), replace=False)
        galcone_dsampled.append(gal_in_bin[mask])
        number_in_bin.append(len(gal_in_bin[mask]))
        if galcone_ids is not None:
            galcone_ids_dsampled.append(galcone_ids[in_bin][mask])

    galcone_dsampled = np.concatenate(galcone_dsampled)

    if galcone_ids is not None:
        galcone_ids_dsampled = np.concatenate(galcone_ids_dsampled)

    return galcone_dsampled, galcone_ids_dsampled

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
    cdf[-1] = 1.0  # 确保最后一个是1

    # 在[0,1]生成N个随机数
    u = np.random.rand(N)

    # 通过CDF反演找到所在区间
    indices = np.searchsorted(cdf, u)

    # 在每个bin内做线性插值
    bin_width = np.diff(x).mean()  # 假设等宽
    samples = x[indices] + (np.random.rand(N) - 0.5) * bin_width

    return samples

def logit_transform(x, a, b):
    return np.log((x - a) / (b - x))

def logit_inverse(y, a, b):
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