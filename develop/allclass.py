'''
Define the basic data class
'''

import numpy as np
from halotools.sim_manager import UserSuppliedHaloCatalog
from dataclasses import dataclass
import pymangle
from misc import *
from astropy.cosmology import Flatw0waCDM

@dataclass
class CosmoParams:
    Om: float = 0.31
    Ob: float = 0.05
    ns: float = 0.9667
    s8: float = 0.83
    H0: float = 67.74
    w0: float = -1.0
    wa: float = 0.0

    @classmethod
    def from_cosmogrid_yml(cls, path: str):
        with open(path, "r") as f:
            for line in f.readlines():
                items = line.split(":")
                if items[0] == "Om": Om = float(items[1])
                if items[0] == "s8": s8 = float(items[1])
                if items[0] == "H0": H0 = float(items[1])
                if items[0] == "Ob": Ob = float(items[1])
                if items[0] == "ns": ns = float(items[1])
                if items[0] == "w0": w0 = float(items[1])

        return cls(
            Om=Om,
            Ob=Ob,
            ns=ns,
            s8=s8,
            H0=H0,
            w0=w0
        )
    
    def to_astropy(self):
        return Flatw0waCDM(
            H0=self.H0,
            Om0=self.Om,
            Ob0=self.Ob,
            ns=self.ns,
            s8=self.s8,
            w0=self.w0,
            wa=self.wa
        )
    
@dataclass
class BaseCatalog:
    meta_data: dict = {}
    pos: np.ndarray = None
    vel: np.ndarray = None

class CosmoGridV1Halo(BaseCatalog):
    mass: np.ndarray = None
    rHalf: np.ndarray = None
    @classmethod
    def from_file(cls, path, boxsize, redshift, npart):
        meta_data = {
            'boxsize': boxsize,
            'redshift': redshift,
            'npart': npart
        }
        halo = np.fromfile(path, dtype=pkd_halo_dtype, count=-1, offset=0)
        ### load pos
        int_fac = 1.0 / 0x80000000
        pos = boxsize * (halo["rPot"] * int_fac + halo["rcen"] + 0.5)
        ### load mass
        mass = halo["fMAss"]*boxsize**3*rhoc0
        ### load vel
        vel_fac = 100*boxsize*np.sqrt(3./(8*PI))*(1+redshift)
        vel = halo["vcom"]*vel_fac
        ### load rHalf
        rHalf = halo["rHalf"]*boxsize

        return cls(
            pos = pos,
            vel = vel,
            mass = mass,
            rHalf = rHalf,
            meta_data = meta_data
        )
        
    def to_file(self, path):
        np.savez(path, meta_data=self.meta_data, pos=self.pos, vel=self.vel, mass=self.mass, rHalf=self.rHalf)

@dataclass
class HODPopulator:
    cosmo: CosmoParams
    hod_config: dict
