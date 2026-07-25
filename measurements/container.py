import numpy as np
from numpy.typing import ArrayLike
from astropy.table import Table
import healpy as hp

class SurveyData:
    def __init__(self, ra, dec, z, **kwargs):
        self._colnames = ['ra', 'dec', 'z']
        self.ra = ra
        self.dec = dec
        self.z = z
        for iattr, ival in kwargs.items():
            setattr(self, iattr, ival)
            self._colnames.append(iattr)

    @property
    def colnames(self):
        return self._colnames

    @property
    def catsize(self):
        return len(self.ra)

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
    
    def apply_indexing_cut(self, indexing: ArrayLike, in_place=False):
        ''' 
        apply indexing cut. 
        '''

        if in_place:
            for iattr in self.colnames:
                ival = getattr(self, iattr)
                setattr(self, iattr, ival[indexing])
            return None
        
        else:
            cutted_attrs = {}
            for iattr in self.colnames:
                ival = getattr(self, iattr)
                cutted_attrs[iattr] = ival[indexing]
            
            ra = cutted_attrs.pop("ra")
            dec = cutted_attrs.pop("dec")
            z = cutted_attrs.pop("z")

            return self.__class__(ra=ra, dec=dec, z=z, **cutted_attrs)

    def apply_condition_cut(self, cond:str, in_place=False):
        ''' 
        apply condition cut. The condition should be a valid string.
        '''

        indexing = eval(cond, self.__dict__)

        if in_place:
            self.apply_indexing_cut(indexing, in_place)
        else:
            return self.apply_indexing_cut(indexing, in_place)

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
        if in_place:
            self.apply_indexing_cut(matched_indexing, in_place=in_place)
        else:
            return self.apply_indexing_cut(matched_indexing, in_place=in_place)
        
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

    # def _compute_indices(self, data, x_edges, y_edges):
    #     ix = np.digitize(data[:, 0], x_edges) - 1
    #     iy = np.digitize(data[:, 1], y_edges) - 1
    #     return ix, iy
    
    # def _valid_mask(self, ix, iy, n_x, n_y):
    #     return (ix >= 0) & (ix < n_x) & (iy >= 0) & (iy < n_y)
    
    # def _get_matched_indexing(self, ref_cat:"SurveyData", nside_x=64, nside_y=64):
    #     ref_xy = np.vstack([ref_cat.ra, ref_cat.dec]).T
    #     match_xy = np.vstack([self.ra, self.dec]).T

    #     all_points = np.vstack([ref_xy, match_xy])
    #     xmin, ymin = np.min(all_points, axis=0)
    #     xmax, ymax = np.max(all_points, axis=0)

    #     x_edges = np.linspace(xmin, xmax, nside_x + 1)
    #     y_edges = np.linspace(ymin, ymax, nside_y + 1)

    #     ref_ix, ref_iy = self._compute_indices(ref_xy, x_edges, y_edges)
    #     match_ix, match_iy = self._compute_indices(match_xy, x_edges, y_edges)

    #     ref_mask = self._valid_mask(ref_ix, ref_iy, nside_x, nside_y)
    #     match_mask = self._valid_mask(match_ix, match_iy, nside_x, nside_y)

    #     ref_cells = set(zip(ref_ix[ref_mask], ref_iy[ref_mask]))
    #     match_cells = set(zip(match_ix[match_mask], match_iy[match_mask]))
    #     common_cells = ref_cells & match_cells

    #     match_selected = [i for i, (ix, iy) in enumerate(zip(match_ix, match_iy)) if (ix, iy) in common_cells]

    #     return match_selected
