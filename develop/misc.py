# >>>========   Define constants   =========<<<
PI = 3.1415926535897932384626433832795
Gconst = 4.3009*1e-9 # Mpc/Msun*(km/s)^2
sol = 2.99792458*1e5 # km/s
rhoc0 = 3*100**2/(8*PI*Gconst) # (Msun/h) / (Mpc/h)^3

# >>>=======================================<<<

import numpy as np
from astropy.io import fits

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


def make_nofz_info(nofz_info, survey_name, zedges, shell_vol, nz_ref):
    nofz_info[survey_name] = {}
    nofz_info[survey_name]['zedges'] = zedges
    nofz_info[survey_name]['shell_vol'] = shell_vol
    nofz_info[survey_name]['nz_ref'] = nz_ref

    return nofz_info