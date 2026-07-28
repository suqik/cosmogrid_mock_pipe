import numpy as np
from numpy.typing import ArrayLike
from astropy.table import Table
import healpy as hp

class SurveyData:
    def __init__(self, ra, dec, z, **kwargs):
        self._colnames = ['ra', 'dec', 'z']
        self.ra = np.atleast_1d(ra)
        self.dec = np.atleast_1d(dec)
        self.z = np.atleast_1d(z)
        for iattr, ival in kwargs.items():
            setattr(self, iattr, np.atleast_1d(ival))
            self._colnames.append(iattr)

    @property
    def colnames(self):
        return self._colnames

    @property
    def catsize(self):
        return len(self)
    
    def __len__(self):
        return len(self.ra)
    
    def __getitem__(self, query):
        if isinstance(query, str):
            if query not in self.colnames:
                raise KeyError(
                    f"Unknown column {query!r}. "
                    f"Available columns: {self.colnames}"
                )

            return getattr(self, query)
        
        selected = {
            name: getattr(self, name)[query]
            for name in self.colnames
        }

        return self.__class__(
            ra=selected.pop("ra"),
            dec=selected.pop("dec"),
            z=selected.pop("z"),
            **selected,
        )

    @classmethod
    def load_cosmogrid_galaxy(cls, fname):
        galcat = Table.read(fname)
        ra = galcat['ra']
        dec = galcat['dec']
        zrsd = galcat['zrsd']
        z = galcat['z']
        
        w = galcat['w']
        survey = galcat['survey']
        
        return cls(ra=ra, dec=dec, z=z, zrsd=zrsd, w=w, survey=survey)
    
    @classmethod
    def load_cosmogrid_void(cls, fname):
        voidcat = Table.read(fname)
        ra = voidcat['ra']
        dec = voidcat['dec']
        z = voidcat['z']
        w = voidcat['w']
        Rv = voidcat['Rv']
        survey = voidcat['survey']

        return cls(ra=ra, dec=dec, z=z, w=w, Rv=Rv, survey=survey)
    
    @classmethod
    def load_cosmogrid_rand(cls, fname):
        randcat = Table.read(fname)
        ra = randcat['RA']
        dec = randcat['DEC']
        z = randcat['Z']
        w = np.ones_like(ra)
        
        return cls(ra=ra, dec=dec, z=z, w=w)
    
    @classmethod
    def load_cosmogrid_shape(cls, fname):
        shapecat = Table.read(fname)
        ra = shapecat['ra']
        dec = shapecat['dec']
        z = shapecat['z']
        z_true = shapecat['z_true']

        g1 = shapecat['g1']
        g2 = shapecat['g2']
        g1_pure = shapecat['g1_pure']
        g2_pure = shapecat['g2_pure']

        #################################################################
        ### FIXME: just a patch. Should be removed after fixing the mocks
        # w = shapecat['w']
        w = np.ones_like(ra)
        #################################################################

        tomo = shapecat['tomo']
        survey = shapecat['survey']

        return cls(ra=ra, dec=dec, z=z, z_true=z_true, w=w,
                   g1=g1, g2=g2, 
                   g1_pure=g1_pure, 
                   g2_pure=g2_pure,
                   tomo=tomo, survey=survey)

    def apply_condition_cut(self, cond:str):
        ''' 
        apply condition cut. The condition should be a valid string.
        '''

        indexing = eval(cond, self.__dict__)
        return self[indexing]

    def to_astropy_table(self):
        table = Table()
        for iattr in self.colnames:
            table[iattr] = getattr(self, iattr)
        return table
    
    def to_healpix(self, nside):
        if not hp.isnsideok(nside):
            raise ValueError(f"{nside} is not a valid nside!")
        pix = hp.ang2pix(nside, self.ra, self.dec, lonlat=True)
        hpmap = np.zeros(hp.nside2npix(nside)).astype(np.int64)
        np.add.at(hpmap, pix, 1)

        return hpmap

    def match_to_reference(self, ref_cat:"SurveyData", nside=64, in_place=False):
        matched_indexing = self._get_matched_indexing(ref_cat, nside=nside)
        return self[matched_indexing]
        
    def _get_matched_indexing(self, ref_cat:"SurveyData", nside=64):
        ### make occupation maps for foreground and background catalogs
        target_shell = self.to_healpix(nside=nside)
        reference_shell = ref_cat.to_healpix(nside=nside)

        ### make intersection of two occupation maps
        target_shell = target_shell * reference_shell

        ### select galaxies in the intersection
        selected_pix = np.argwhere(target_shell != 0)
        target_pix = hp.ang2pix(nside, self.ra, self.dec, lonlat=True)
        matched_indexing = np.isin(target_pix, selected_pix)
        
        return matched_indexing