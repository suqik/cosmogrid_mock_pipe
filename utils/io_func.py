'''
Basic I/O functions
'''

import numpy as np
import healpy as hp
import pyccl as ccl
import h5py
from astropy.io import fits
from typing import Union
from halotools.empirical_models import halo_mass_to_halo_radius, NFWProfile
from halotools.sim_manager import UserSuppliedHaloCatalog

# >>>========   Define constants   =========<<<

Gconst = 4.3009*1e-3 # Mpc/Msun*(km/s)^2
sol = 2.99792458*1e5 # km/s
rhoc0 = 3*100**2/(8*np.pi*Gconst)*sol

# >>>=======================================<<<

# >>>==================    Define used data types   ==================<<<
### PKDGrav3 FOF halo catalog
pkd_halo_dtype = np.dtype([
    ("rPot", ("i4", 3)),
    ("minPot", "f4"),
    ("rcen", ("f4", 3)),
    ("rcom", ("f4", 3)),
    ("vcom", ("f4", 3)), # this is a typo, should be vcom (velocity)
    ("angular", ("f4", 3)),
    ("inertia", ("f4", 6)),
    ("sigma", "f4"),
    ("rMax", "f4"),
    ("fMAss", "f4"),
    ("fEnvironDensity0", "f4"),
    ("fEnvironDensity1", "f4"),
    ("rHalf", "f4"),
    ("rvir", "f4"),
    ("profile", ('i4', 20)),
    ("virprofile", ('i4', 20))
])

### MAKE-SURVEY output catalog
make_survey_type = np.dtype([
    ("ra", "f4"),
    ("dec", "f4"),
    ("z", "f4"),
    ("w", "f4"),
    ("nref", "f4"),
    ("dummy", "f4"),
])

### DIVE input tracer catalog
dive_tracer_type = np.dtype([
    ("pos", ("f4", 3))
])

### DIVE output void catalog
dive_void_type = np.dtype([
    ("pos", ("f4", 3)),
    ("Rv", "f4")
])

### SWOT input background catalog (ascii)
bgal_type = np.dtype(
    [
        ("ra", "f4"), 
        ("dec", "f4"), 
        ("z", "f4"), 
        ("sigz", "f4"),
        ("g1", "f4"), 
        ("g2", "f4"), 
        ("w", "f4")
    ]
)

### matched foreground catalog
### Note for input of SWOT, the last line
### should be `sigma_z` rather than `weight`
### survey label: 0 for CMASSLOWZ, 1 for LOWZE2, 2 for LOWZE3, 3 for 2dFLenS
fgal_type = np.dtype(
    [
        ("ra", "f4"), 
        ("dec", "f4"), 
        ("z", "f4"), 
        ("w", "f4"),
        ("survey", "i4")
    ]
)

fvoid_type = np.dtype(
    [
        ("ra", "f4"), 
        ("dec", "f4"), 
        ("z", "f4"), 
        ("w", "f4"), 
        ("Rv", "f4"), 
        ("survey", "i4")
    ]
)

# >>>========================================================<<<

# >>>==================    I/O functions   ==================<<<
def get_cosmo_from_file(fname:str, otype="ccl") -> Union[dict, ccl.Cosmology]:
    '''
    Get cosmology from PKDGrav3 config. Can transform to ccl format

    Parameters:
    ----------
    fname: str
        Input ascii file name.
    otype: str
        Output type. Can be "dict" or "ccl".

    Returns:
    -------
    outputs: dict or ccl.Cosmology
        Cosmology dictionary.
    '''

    cosmo_par = {}
    with open(fname, "r") as f:
        for line in f.readlines():
            items = line.split(":")
            cosmo_par[items[0]] = float(items[1])
    
    if otype == "ccl":
        outputs = ccl.Cosmology(
            h=cosmo_par["H0"]/100, 
            Omega_b=cosmo_par["Ob"], 
            Omega_c=cosmo_par["O_cdm"], 
            A_s=cosmo_par["As"], 
            n_s=cosmo_par["ns"], 
            w0=cosmo_par["w0"], 
            wa=cosmo_par["wa"],
            m_nu=cosmo_par["m_nu"]*3
        )
    elif otype == "dict":
        outputs = cosmo_par
    else:
        raise NotImplementedError(f"Output type {otype} not implemented!")

    return outputs

def get_pkd_halo_attrs(fname:str, attrs:Union[str,list]=["pos", "mass"], Lbox:float=None, redshift:float=None) -> dict:
    '''
    Get PKDGrav3 halo attributes.
    Special attrs are position, velocity and Mass. Mass are defined as virial mass.
    
    Parameters:
    ----------
    fname: str
        Input pkd halo file name.
    attrs: List or str 
        Attributes to be extracted. Three external attributes are 
        "pos" (for position, in Mpc/h), "vel" (for velocity, in km/s)
        and "mass" (for virial mass, in Msun/h).
    Lbos: float or None
        Boxsize of the snapshot. Required when asking position, velocity
        or mass.
    rhoc0: float or None
        Critical density at redshift 0. Required when asking velocity or
        mass.
    redshift: float or None
        redshift of the snapshot. Required when asking velocity.
    
    Returns:
    -------
    outputs: dict
        A dictionary of asked halo attributes.
    '''
    halo = np.fromfile(fname, dtype=pkd_halo_dtype, count=-1, offset=0)
    if type(attrs) is not list:
        attrs = [attrs]

    # outputs = []
    outputs = {}
    for idx in range(len(attrs)):
        iattr = attrs[idx]
        if iattr == "pos":
            if Lbox is not None:
                int_fac = 1.0 / 0x80000000
                pos = Lbox * (halo["rPot"] * int_fac + halo["rcen"] + 0.5)
                # outputs.append(pos)
                outputs["pos"] = pos
            else:
                raise ValueError("Calculate position needs boxsize as input!")
        elif iattr == "mass":
            if Lbox is not None:
                mass = halo["fMAss"]*Lbox**3*rhoc0
                # outputs.append(mass)
                outputs["mass"] = mass
            else:
                raise ValueError("Calculate mass needs boxsize as input!")
        elif iattr == "vel":
            if Lbox is not None and redshift is not None:
                vel_fac = 100*Lbox*np.sqrt(3./(8*np.pi))*(1+redshift)
                vel = halo["vcom"]*vel_fac
                # outputs.append(vel)
                outputs["vel"] = vel
            else:
                raise ValueError("Calculate position needs boxsize and redshift as input!")
        elif iattr in halo.dtype.fields.keys():
            # outputs.append(halo[iattr])
            outputs[iattr] = halo[iattr]
        else:
            print(f"{iattr} does not support in halo catalog!")

    # if len(outputs) == 1:
    #     return outputs[0]
    # else:
    #     return outputs

    return outputs

def pkd_to_hod_type(pkd_infos:dict, cosmo:ccl.Cosmology, pmass:float, boxsize:float, redshift:float) -> UserSuppliedHaloCatalog:
    '''
    Transform the default PKDGrav3 halo format to HOD format.

    Parameters:
    ----------
    pkd_infos: dict
        PKDGrav3 halo attributes. Keys of the dictionary should be "pos", "vel" and "mass".
    cosmo: ccl.Cosmology
        Cosmology info from pyccl. Used to calculate concentration and radius.
    pmass: float
        Mass of the particle.
    boxsize: float
        Boxsize of the snapshot.
    redshift: float
        redshift of the snapshot.

    Returns:
    -------
    halo_cat: UserSuppliedHaloCatalog
        Halo catalog in HOD format.
    '''

    mdef = ccl.halos.MassDefVir
    conc_model = ccl.halos.ConcentrationDuffy08(mass_def=mdef)
    scale_fac = 1./(1 + redshift)
    hubble = cosmo.to_dict()["h"]

    halo_pos = pkd_infos["pos"]
    halo_vel = pkd_infos["vel"]
    halo_mass = pkd_infos["mass"]

    halo_radii = mdef.get_radius(cosmo, halo_mass/hubble, scale_fac)*hubble # Mpc/h, physical radius
    halo_concs = conc_model(cosmo, halo_mass/hubble, scale_fac) # concentration, Duffy08
    num_halo = len(halo_pos)

    halo_cat=UserSuppliedHaloCatalog(
        redshift=redshift,
        Lbox=boxsize,
        particle_mass=pmass,
        halo_upid=np.zeros(num_halo)-1,
        halo_x=halo_pos[:,0],
        halo_y=halo_pos[:,1],
        halo_z=halo_pos[:,2],
        halo_vx=halo_vel[:,0],
        halo_vy=halo_vel[:,1],
        halo_vz=halo_vel[:,2],
        halo_id=np.arange(num_halo),
        halo_rvir=halo_radii,
        halo_mvir=halo_mass,
        halo_nfw_conc=halo_concs,  ### concentration of NFW
        halo_hostid=np.arange(num_halo)
    )

    return halo_cat


def lcone_to_swot_type(lcone_pos:np.ndarray, cosmo:ccl.Cosmology, weight:np.ndarray=None):
    '''
    Transform the default lightcone format (3D Cartesian coordinates) to SWOT foreground format.

    Parameters:
    ----------
    lcone_pos: np.ndarray
        Positions of lightcone particles. Should be in Cartesian coordinates.
    cosmo: ccl.Cosmology
        Cosmology info from pyccl. Used to calculate redshifts.
    weight: np.ndarray or None
        Weights of the tracers containing observational systematics. In simulation
        this can be ignored. Default are unions.

    Returns:
    -------
    outputs: np.ndarray
        Positions and weights in SWOT format. Can be saved and read by SWOT.
    '''
    ntot = len(lcone_pos)
    tracer_ra, tracer_dec = hp.vec2ang(lcone_pos, lonlat=True)
    tracer_chi = np.linalg.norm(lcone_pos, axis=1)
    tracer_z = 1./ccl.scale_factor_of_chi(cosmo, tracer_chi/cosmo.to_dict()["h"]) - 1
    
    outputs = np.empty((ntot,), dtype=fgal_type)
    outputs['ra'] = tracer_ra
    outputs['dec'] = tracer_dec
    outputs['z'] = tracer_z
    if weight is not None:
        outputs['w'] = weight
    else:
        outputs['w'] = np.ones(ntot)
    
    return outputs

def load_raytracing_maps(fname:str, quantities:Union[str, list]=["gamma1", "gamma2"]) -> dict:
    '''
    Load raytracing maps in Dorian output formats.

    Parameters:
    ----------
    fname: str
        Input raytracing map file name.
    quantities: List or str 
        Quantities to be extracted. Can be "kappa", "omega_ray", "gamma1" 
        and "gamam2". Default are "gamma1" and "gamma2".

    Returns:
    -------
    outputs: dict
        Asked quantities. If `quantities` is a list, then return a dict of the
        asked quantities; otherwise return a dict with one key-value pair.
    '''

    with h5py.File(fname, "r") as f:
        A = np.array(f["Distortion_matrix"]["Raytraced"])

    if type(quantities) is not list:
        quantities = [quantities]
    
    outputs = {}
    for iquantity in quantities:
        if iquantity == "kappa":
            kappa = -(A[0][0] + A[1][1]) / 2
            outputs["kappa"] = kappa
        if iquantity == "omega_ray":
            omega_ray = (A[0][1] - A[1][0]) / 2
            outputs["omega_ray"] = omega_ray
        if iquantity == "gamma1":
            gamma1 = -(A[0][0] - A[1][1]) / 2
            outputs["gamma1"] = gamma1
        if iquantity == "gamam2":
            gamma2 = -(A[0][1] + A[1][0]) / 2
            outputs['gamma2'] = gamma2

    return outputs

def loadFitsMaps(name:str) -> list:
    """
    Load Fits-type file.

    Parameters
    ----------
    name : string
        name of the file
    
    Returns
    -------
    maps : list
        maps loaded in fits file
    """

    hdu = fits.open(name)
    nmaps = len(hdu) - 1

    maps = []
    for i in range(nmaps):
        tmp = (hdu[i+1].data["VALUE"]).flatten()
        maps.append(tmp)

    return maps

def saveFitsFullMap(name:str, full:Union[np.ndarray, list], comments:Union[str, list]=None, verbose:bool=True) -> None:
    """
    Copy from SALMO code (Lin 2020).

    Save a HEALPix map as a FITS file under a specific convention.

    Parameters
    ----------
    name : string
    full name of the file
    full : numpy array
    the HEALPix map to save, will be saved as a float32 array
    verbose : bool, optional
    print verbose message

    Returns
    -------
    No returns
    """

    if not isinstance(full, list):
        full = [full]

    if comments is None:
        comments = [f"Field{i}" for i in range(len(full))]
    elif not isinstance(comments, list):
        comments = [comments]

    HDU_List = []
    HDU_List.append(fits.PrimaryHDU())
    for imap in full:
        imap   = imap.astype(np.float32)
        nbRows = imap.size // 1024
        imap   = imap.reshape(nbRows, 1024)
        nside  = hp.npix2nside(imap.size)

        HDU_i = fits.BinTableHDU.from_columns([
            fits.Column(name='VALUE', format='1024E', unit='-       ', array=imap)
        ])

        hdr = HDU_i.header
        hdr.append(('COMMENT',  'HEALPIX pixelisation'),                                        bottom=True)
        hdr.append(('ORDERING', 'RING    ',    'Pixel ordering scheme'),                        bottom=True)
        hdr.append(('COORDSYS', 'C       ',    'Ecliptic, Galactic or Celestial (equatorial)'), bottom=True)
        hdr.append(('NSIDE',    nside,         'nside of the pixel'),                           bottom=True)
        hdr.append(('FIRSTPIX', 0,             'First pixel # (0 based)'),                      bottom=True)
        hdr.append(('LASTPIX',  12*nside**2-1, 'Last pixel # (0 based)'),                       bottom=True)
        hdr.append(('INDXSCHM', 'IMPLICIT',    'Indexing: IMPLICIT or EXPLICIT'),               bottom=True)

        HDU_List.append(HDU_i)

    fits.HDUList(HDU_List).writeto(name, overwrite=True)
    if verbose == True:
        print('Saved \"%s\"' % name)
    
    return None