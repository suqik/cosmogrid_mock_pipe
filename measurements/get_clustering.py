import numpy as np
from astropy.table import Table
from tqdm import trange
from utils.io_func import get_cosmo_from_file

def make_void_clustering_random(catalog, seed=1745):
    '''
    Make void random catalog for clustering measurements.
    First group the void catalog by redshifts and void size,
    then shuffle the catalog in each small group.
    '''

    # initial the random generator
    rng1 = np.random.default_rng(seed=seed)
    rng2 = np.random.default_rng(seed=2*seed)

    # Group the void catalog by redshifts and void size
    # zmin = catalog['z'].min()
    # zmax = catalog['z'].max()
    rmin = catalog['Rv'].min()
    rmax = catalog['Rv'].max()
    
    rbins = np.arange(rmin, rmax, 1)

    rand_cat = []
    for i in range(len(rbins)-1):
        slt = (catalog['Rv'] > rbins[i]) & (catalog['Rv'] < rbins[i+1])
        subcat = catalog[slt]
        shuffled_zidx = rng1.permutation(len(subcat))
        shuffled_ridx = rng2.permutation(len(subcat))
        subcat['z'] = subcat['z'][shuffled_zidx]
        subcat['Rv'] = subcat['Rv'][shuffled_ridx]
        rand_cat.append(subcat)

    rand_cat = np.concatenate(rand_cat)
    return rand_cat

if __name__ == '__main__':
    import sys
    import json

    vfile = sys.argv[1]
    vcat = np.load(vfile)


    cparnamebase = "/data3/suchen/CosmoGridV1/grid/cosmo_{:06d}/run_0/params.yml"
    cosmo_label = int(vfile.split('/')[-1].split('_')[1])

    rcut_min = 15.0
    rcut_max = 30.0

    cosmo_dict = get_cosmo_from_file(cparnamebase.format(cosmo_label), otype='dict')
    Om = cosmo_dict['Om']
    w0 = cosmo_dict['w0']

    vcat = Table(vcat)
    vcat.write("tmp/vcat.fits", overwrite=True)
    rand_cat = make_void_clustering_random(vcat)
    rand_cat = Table(rand_cat)
    rand_cat.write("tmp/rand_cat.fits", overwrite=True)

    print("{} {} {:.2f} {:.2f}".format(Om, w0, rcut_min, rcut_max))

    # with open("/data2/suchen/CosmoGrid/fix_HOD_suits/HOD/ngals.json", "r") as f:
    #     ngal_dict = json.load(f)

    # cparnamebase = "/data3/suchen/CosmoGridV1/grid/cosmo_{:06d}/run_0/params.yml"
    # cosmo_label = int(vfile.split('/')[-1].split('_')[1])
    # curr_ngal = ngal_dict[f'cosmo{cosmo_label:06d}']

    # rcut_min = 1./np.cbrt(curr_ngal*1e-4)
    # rcut_max = 2./np.cbrt(curr_ngal*1e-4)

    # cosmo_dict = get_cosmo_from_file(cparnamebase.format(cosmo_label), otype='dict')
    # Om = cosmo_dict['Om']
    # w0 = cosmo_dict['w0']

    # vcat = Table(vcat)
    # vcat.write("tmp/vcat.fits", overwrite=True)
    # rand_cat = make_void_clustering_random(vcat)
    # rand_cat = Table(rand_cat)
    # rand_cat.write("tmp/rand_cat.fits", overwrite=True)

    # print("{} {} {:.2f} {:.2f}".format(Om, w0, rcut_min, rcut_max))