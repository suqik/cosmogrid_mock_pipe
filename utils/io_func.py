'''
Basic I/O functions
'''

import os
import json
import numpy as np
import healpy as hp
import pyccl as ccl
import h5py
from astropy.io import fits
from typing import Union
from halotools.sim_manager import UserSuppliedHaloCatalog

# >>>========   Define constants   =========<<<

Gconst = 4.3009*1e-9 # Mpc/Msun*(km/s)^2
sol = 2.99792458*1e5 # km/s
rhoc0 = 3*100**2/(8*np.pi*Gconst) # (Msun/h) / (Mpc/h)^3

# >>>=======================================<<<

# >>>==================    Define used data types   ==================<<<
### PKDGrav3 FOF halo catalog
pkd_halo_dtype = np.dtype([
    ("rPot", ("i4", 3)),
    ("minPot", "f4"),
    ("rcen", ("f4", 3)),
    ("rcom", ("f4", 3)),
    ("vcom", ("f4", 3)), 
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

### SWOT input background catalog (ascii)
bgal_type = np.dtype(
    [
        ("ra", "f4"), 
        ("dec", "f4"), 
        ("z", "f4"),
        ("z_true", "f4"),
        ("sigz", "f4"),
        ("g1", "f4"), 
        ("g2", "f4"), 
        ("g1_pure", "f4"),
        ("g2_pure", "f4"),
        ("w", "f4"),
        ("tomo", "i4"),
        ("survey", "i4")
    ]
)

### matched foreground catalog
### survey label: 0 for LOWZ, 1 for LOWZE2, 2 for LOWZE3, 3 for 2dFLenS, 4 for CMASS
### gal_type: 1 for central, 0 for satellite
### GID: particle id in box, for estimating replication rate.
fgal_type = np.dtype(
    [
        ("ra", "f4"), 
        ("dec", "f4"), 
        ("z", "f4"),
        ("zrsd", "f4"), 
        ("w", "f4"),
        ("survey", "i4"),
        ("gal_type", "i4"),
        ("host_halo_mvir", "f4"),
        ("GID", "i4")
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
    Lbox: float or None
        Boxsize of the snapshot. Required when asking position, velocity
        or mass.
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
                # https://cosmo-gitlab.phys.ethz.ch/jafluri/pkdgrav3_intro/-/tree/main/example/pkdgrav3_output?ref_type=heads
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
                # https://cosmo-gitlab.phys.ethz.ch/jafluri/pkdgrav3_intro/-/tree/main/example/pkdgrav3_output?ref_type=heads
                vel_fac = 100*Lbox*np.sqrt(3./(8*np.pi))*(1+redshift)
                vel = halo["vcom"]*vel_fac
                # outputs.append(vel)
                outputs["vel"] = vel
            else:
                raise ValueError("Calculate position needs boxsize and redshift as input!")
        elif iattr == "rHalf":
            if Lbox is not None:
                outputs["rHalf"] = halo["rHalf"]*Lbox
            else:
                raise ValueError("Calculate rHalf needs boxsize as input!")
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
    halo_rhalf = pkd_infos["rHalf"]

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
        halo_rhalf=halo_rhalf, ### half mass radius in Mpc/h
        halo_hostid=np.arange(num_halo)
    )

    return halo_cat

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