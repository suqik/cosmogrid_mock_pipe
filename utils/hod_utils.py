'''
Utils to apply HOD. Totally from HODOR.
'''
import os
import sys
from typing import Union
from copy import copy
import numpy as np
from scipy.special import erf
import warnings

from halotools.empirical_models import HodModelFactory, HeavisideAssembias
from halotools.empirical_models import TrivialPhaseSpace, NFWPhaseSpace

from halotools.empirical_models import occupation_model_template, model_defaults
from halotools.custom_exceptions import HalotoolsError

# >>> ==================================      model class      ================================== <<<
class ModelClass():
  """ The class that populates the halo catalog with galaxies
  given the config file """

  def __init__(self, halo_files, halo_cat_list, \
          pipe_config, OmegaM=None):

    self.model = int(pipe_config.model)
    if self.model == 0:
      model_name = 'MW'
    elif self.model == 1:
      model_name = 'GP18'
    elif self.model == 2:
      model_name = 'MW_fic'
    elif self.model == 3:
      model_name = 'MW_fic2'
    elif self.model == 4:
      model_name = 'MW_ficAssemblyBias'    
    else:
      raise ValueError('Unrecognised model')
    self.parameters_names = pipe_config.model_params_names
    self.num_params = len(self.parameters_names)

    print(f'\nINFO: The used HOD model is {model_name} having {self.parameters_names} as parameters')
    # if len(self.parameters_names) != self.num_params:
    #   print("ERROR: Check the num_params and model_params_names in the config file")
    #   sys.exit(1)

    self.verbose   = pipe_config.verbose
    self.redshift  = pipe_config.redshift
    self.box_size  = pipe_config.Lbox
    self.init_seed = pipe_config.init_seed
    self.num_seeds = pipe_config.num_seeds

    print(f'\nINFO: The number of seeds to populate galaxies is {self.num_seeds}, starting from {self.init_seed}')
    self.z_space = pipe_config.z_space
    self.halo_files = halo_files
    print(f'\nINFO: The number of halo catalogs used to populate galaxies is {len(halo_files)}')

    self.Num_ptcl_requirement = pipe_config.Num_ptcl_requirement

    self.halo_cat_list = halo_cat_list
    self.model_instance = self.compute_model_instance()
    if self.z_space:
      self.Omega_m = OmegaM
      self.rsd_shift = self.compute_rsd_shift()

  def compute_model_instance(self):
    """ Compute model instance """
    print('STATUS: Constructing HOD model ...')
    if self.model == 0:
      cens_occ_model = MWCens(redshift=self.redshift)
      sats_occ_model = MWSats(redshift=self.redshift, cenocc_model=cens_occ_model, modulate_with_cenocc=True)
    elif self.model == 1:
      cens_occ_model = ContrerasCens(redshift=self.redshift)
      sats_occ_model = ContrerasSats(redshift=self.redshift)
    elif self.model == 2:
      cens_occ_model = MWCens_IC(redshift=self.redshift)
      sats_occ_model = MWSats(redshift=self.redshift, cenocc_model=cens_occ_model, modulate_with_cenocc=True)
    elif self.model == 3:
      cens_occ_model = MWCens_IC(redshift=self.redshift)
      sats_occ_model = MWSats2(redshift=self.redshift, cenocc_model=cens_occ_model, modulate_with_cenocc=True)
    elif self.model == 4:
      # We use 'rHalf' as the secondary property.
      # Change 'rHalf' to match your catalog column name exactly (e.g. 'halo_rhalf') if different.
      sec_key = 'halo_rhalf' 
      cens_occ_model = ABMWCens_IC(redshift=self.redshift, sec_haloprop_key=sec_key)
      sats_occ_model = ABMWSats(redshift=self.redshift, cenocc_model=cens_occ_model, 
                                modulate_with_cenocc=True, sec_haloprop_key=sec_key)

    cens_prof_model = TrivialPhaseSpace(redshift=self.redshift)
    sats_prof_model = NFWPhaseSpace(redshift=self.redshift)
    sats_occ_model._suppress_repeated_param_warning = True

    model_instance = HodModelFactory( \
      centrals_occupation=cens_occ_model, \
      centrals_profile=cens_prof_model, \
      satellites_occupation=sats_occ_model, \
      satellites_profile=sats_prof_model)
    return model_instance

  def compute_rsd_shift(self):
    """ Compute the RSD redshift """
    if self.z_space:
      Omega_l = 1.0 - self.Omega_m
      hubble = 100 * np.sqrt(self.Omega_m * (self.redshift + 1.0)**3 + Omega_l)
      rsd_shift = (self.redshift + 1.0) / hubble
    else:
      rsd_shift = 0
    return rsd_shift


  def populate_mock(self, param_vals:Union[np.ndarray,dict], ref_num_dens, indx=0, ifcheck=True):
    """ Populate the halo catalog with galaxies for different seeds """
    if self.verbose:
      print('\nSTATUS: Populate the halo catalog with galaxies...')

    if type(param_vals) != dict:
      for i in range(self.num_params):
        self.model_instance.param_dict[self.parameters_names[i]] = param_vals[i]
    else:
      for key, val in param_vals.items():
        self.model_instance.param_dict[key] = val

    # print(self.model_instance.param_dict)

    ### Function which tells you how many central and satellite galaxies were alocated to the FastPM halo catalog.
    #self.check_galaxy_types(self.init_seed)
    #exit()

    ### Actually populating the halo mocks with galaxies for different seeds.
    return_dict = {}
    for j, halo_cat in enumerate(self.halo_cat_list):
      for i in range(self.num_seeds):
        seedt, gal_samplet = self.intermediate_populate_mock(self.init_seed + i*1000 + j + indx, halo_cat)

        return_dict[os.path.basename(self.halo_files[j]) + str(seedt)] = gal_samplet

        if ifcheck:
          if (i == 0) and (j == 0):
            meas_num_dens = len(gal_samplet) / (self.box_size ** 3)
            if np.abs(meas_num_dens - ref_num_dens) > 0.10 * ref_num_dens: ### Check whether one catalog is 8 Sigma_1 larger than the reference. Sigma_1 is the error of one catalog. 
              return return_dict

    return return_dict


  def intermediate_populate_mock(self, seed, halo_cat):
    """ This is an intermediate step in populating the halo catalog with galaxies,
    required to parallelize the procedure for multiple seeds """
    if self.verbose:
      print('SubSTATUS: Populate the halo catalog with galaxies using {} as seed...'.format(seed))

    # check rhalf columns has been read in 
    # print("Halo Table Columns:", halo_cat.halo_table.colnames)

    self.model_instance.populate_mock(halo_cat, Num_ptcl_requirement=self.Num_ptcl_requirement, 
                                      seed=seed)
            
    gtable = self.model_instance.mock.galaxy_table
    idx = (gtable['gal_type'] == 'centrals')
    gtable['gal_type'] = idx.astype(int)

    ### TO DO for velocity dispersion:
    ### Must include an if, for the case when the vdisp is not used.
    ### Velocity Dispersion
    range_ = gtable['gal_type'] == 0
    # vdisp = self.model_instance.param_dict["vdisp"]
    vdisp = 1.0
    gtable['vz'][range_] = (gtable['vz'][range_] - gtable['halo_vz'][range_]) * vdisp + gtable['halo_vz'][range_]

    # # RSD shift along OZ axis.
    # z_rsd = gtable['z'] + gtable['vz'] * self.rsd_shift
    # z_rsd = (z_rsd + self.box_size) % self.box_size

    # gal_sample = np.rec.fromarrays([gtable['x'], gtable['y'], z_rsd, gtable['z'], gtable['gal_type']], dtype=[('x', 'f4'), ('y', 'f4'), ('z_rsd', 'f4'), ('z', 'f4'), ('gal_type', 'i4')])
    gal_sample = np.rec.fromarrays(
      [gtable['x'], gtable['y'], gtable['z'], gtable['vx'], gtable['vy'], gtable['vz'], gtable['gal_type'], 
       gtable['halo_mvir'], gtable['halo_rhalf'], gtable['halo_id']], 
      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('vx', 'f4'), ('vy', 'f4'), ('vz', 'f4'), ('gal_type', 'i4'),
            ('halo_mvir', 'f4'),('halo_rhalf', 'f4'),('halo_id', 'i4') ]
      )

    return seed, gal_sample


  def check_galaxy_types(self, seed):
    """ Function which tells you how many central and satellite galaxies were alocated to the FastPM halo catalog. """
    if self.verbose:
      print('SubSTATUS: Populate the halo catalog with galaxies using {} as seed...'.format(seed))
    
    for halo_cat in self.halo_cat_list:

      self.model_instance.populate_mock(halo_cat, Num_ptcl_requirement=self.Num_ptcl_requirement, seed=seed)
      gtable = self.model_instance.mock.galaxy_table

      idx = (gtable['gal_type'] == 'centrals')
      gtable['gal_type'] = idx.astype(int)

      unique, frequency = np.unique(gtable['gal_type'], return_counts=True)
      # print unique values array
      print("Unique Values:", unique)

      # print frequency array
      print("Frequency Values:", frequency)


  def return_galaxy_types(self, dict_of_gsamples):
    """ Function which tells you how many central and satellite galaxies were alocated to the FastPM halo catalog. """

    nsat = np.zeros(len(dict_of_gsamples.keys()))
    ncen = np.zeros(len(dict_of_gsamples.keys()))

    for j, key in enumerate(dict_of_gsamples.keys()):
      if self.verbose:
        print('INFO: >> Seed = {}...'.format(key))

      gal_type = dict_of_gsamples[key]["gal_type"]

      unique, frequency = np.unique(gal_type, return_counts=True)
      # print unique values array
      #print("Unique Values:", unique)
      nsat[j] = frequency[0]  

      if len(frequency) == 2:
        ncen[j] = frequency[1]
      # print frequency array
      #print("Frequency Values:", frequency)

    return np.mean(nsat), np.mean(ncen)

# >>> =========================================================================================== <<<

# >>> =====================================    HOD models   ===================================== <<<

# The models in this file are modulates of the Zheng07 model, see
# https://halotools.readthedocs.io/en/latest/_modules/halotools/empirical_models/occupation_models/zheng07_components.html

class MWCens(occupation_model_template.OccupationComponent):
  '''
    The central occupation model used by Martin White:
    <Ncen> = 0.5 * [ 1 + Erf( x / sqrt{2} / sigma ) ]
    where x = ln( Mhalo / Mcut )
  '''
  def __init__(self, threshold=model_defaults.default_luminosity_threshold, \
      prim_haloprop_key=model_defaults.prim_haloprop_key, redshift=0, \
      **kwargs):
    '''
      Examples
      --------
      cen_model = MWCens()
      cen_model = MWCens(threshold=-19.5)
      cen_model = MWCens(prim_haloprop_key='halo_m200b')
    '''
    upper_occupation_bound = 1.0
    super(MWCens, self).__init__(gal_type='centrals', \
        threshold=threshold, upper_occupation_bound=upper_occupation_bound, \
        prim_haloprop_key=prim_haloprop_key, **kwargs)
    self.redshift = redshift
    self.param_dict = self.get_default_parameters(self.threshold)
    self.list_of_haloprops_needed = ['halo_id','halo_rhalf']

  def mean_occupation(self, **kwargs):
    if 'table' in list(kwargs.keys()):
      mass = kwargs['table'][self.prim_haloprop_key]
    elif 'prim_haloprop' in list(kwargs.keys()):
      mass = np.atleast_1d(kwargs['prim_haloprop'])
    else:
      msg = ('\nYou must pass either a `table` or `prim_haloprop` argument \n'
          'to the `mean_occupation` function of the `MWCens` class.\n')
      raise HalotoolsError(msg)

    logM = np.log(mass)
    inv_sqrt2 = 0.7071067811865475244
    ln10 = 2.30258509299404568402

    mean_ncen = 0.5 * (1.0 + erf((logM - ln10 * self.param_dict['logMcut']) * \
        inv_sqrt2 / self.param_dict['sigma_logM']))
    return mean_ncen

  def get_default_parameters(self, threshold):
    '''
      Best-fit HOD parameters from Martin White
    '''
    param_dict = (
        {'logMcut': 11.5,
        'sigma_logM': 1.0}
        )
    return param_dict


class MWCens_IC(occupation_model_template.OccupationComponent):
  '''
    The central occupation model used by Martin White with incompleteness:
    <Ncen> = f_{ic} * 0.5 * [ 1 + Erf( x / sqrt{2} / sigma ) ] 
    where x = ln( Mhalo / Mcut )
  '''
  def __init__(self, threshold=model_defaults.default_luminosity_threshold, \
      prim_haloprop_key=model_defaults.prim_haloprop_key, redshift=0, \
      **kwargs):
    '''
      Examples
      --------
      cen_model = MWCens()
      cen_model = MWCens(threshold=-19.5)
      cen_model = MWCens(prim_haloprop_key='halo_m200b')
    '''
    upper_occupation_bound = 1.0
    super(MWCens_IC, self).__init__(gal_type='centrals', \
        threshold=threshold, upper_occupation_bound=upper_occupation_bound, \
        prim_haloprop_key=prim_haloprop_key, **kwargs)
    self.redshift = redshift
    self.param_dict = self.get_default_parameters(self.threshold)
    self.list_of_haloprops_needed = ['halo_id','halo_rhalf']

  def mean_occupation(self, **kwargs):
    if 'table' in list(kwargs.keys()):
      mass = kwargs['table'][self.prim_haloprop_key]
    elif 'prim_haloprop' in list(kwargs.keys()):
      mass = np.atleast_1d(kwargs['prim_haloprop'])
    else:
      msg = ('\nYou must pass either a `table` or `prim_haloprop` argument \n'
          'to the `mean_occupation` function of the `MWCens` class.\n')
      raise HalotoolsError(msg)

    logM = np.log(mass)
    inv_sqrt2 = 0.7071067811865475244
    ln10 = 2.30258509299404568402

    mean_ncen = self.param_dict['fic'] * 0.5 * (1.0 + erf((logM - ln10 * self.param_dict['logMcut']) * \
        inv_sqrt2 / self.param_dict['sigma_logM']))

    mean_ncen = np.maximum(0, mean_ncen)
    
    return mean_ncen

  def get_default_parameters(self, threshold):
    '''
      Best-fit HOD parameters from Martin White
    '''
    param_dict = (
        {'logMcut': 11.5,
        'sigma_logM': 1.0,
        'fic': 1.0}
        )
    return param_dict

class MWSats(occupation_model_template.OccupationComponent):
  '''
    The satellite occupation model used by Martin White:
    <Nsat> = [ ( Mhalo - k * Mcut ) / M1 ]^alpha
  '''
  def __init__(self, threshold=model_defaults.default_luminosity_threshold, \
      prim_haloprop_key=model_defaults.prim_haloprop_key, redshift=0, \
      modulate_with_cenocc=False, cenocc_model=None, **kwargs):
    upper_occupation_bound = float("inf")
    super(MWSats, self).__init__(gal_type='satellites', \
        threshold=threshold, upper_occupation_bound=upper_occupation_bound, \
        prim_haloprop_key=prim_haloprop_key, **kwargs)
    self.redshift = redshift
    self.param_dict = self.get_default_parameters(self.threshold)
    self.list_of_haloprops_needed = ['halo_id','halo_rhalf']

    if cenocc_model is None:
      cenocc_model = MWCens(prim_haloprop_key=prim_haloprop_key, \
          threshold=threshold)
    else:
      if modulate_with_cenocc is False:
        msg = ("You chose to input a `cenocc_model`, but you set the \n"
            "`modulate_with_cenocc` keyword to False, so your "
            "`cenocc_model` will have no impact on the model's behavior.\n"
            "Be sure this is what you intend before proceeding.\n")
        warnings.warn(msg)

    self.modulate_with_cenocc = modulate_with_cenocc
    if self.modulate_with_cenocc:
      try:
        assert isinstance(cenocc_model, \
            occupation_model_template.OccupationComponent)
      except AssertionError:
        msg = ('The input `cenocc_model` must be an instance of \n'
            '`OccupationComponent` or one of its sub-classes.\n')
        raise HalotoolsError(msg)

      self.central_occupation_model = cenocc_model
      self.param_dict.update(self.central_occupation_model.param_dict)

  def mean_occupation(self, **kwargs):
    if self.modulate_with_cenocc:
      for key, value in self.param_dict.items():
        if key in self.central_occupation_model.param_dict:
          self.central_occupation_model.param_dict[key] = value

    if 'table' in list(kwargs.keys()):
      mass = kwargs['table'][self.prim_haloprop_key]
    elif 'prim_haloprop' in list(kwargs.keys()):
      mass = np.atleast_1d(kwargs['prim_haloprop'])
    else:
      msg = ('\nYou must pass either a `table` or `prim_haloprop` argument \n'
          'to the `mean_occupation` function of the `MWSats` class.\n')
      raise HalotoolsError(msg)

    Mcut = 10.**self.param_dict['logMcut']
    M1 = 10.**self.param_dict['logM1']

    mean_nsat = np.zeros_like(mass)
    idx_nonzero = np.where(mass - self.param_dict['k'] * Mcut > 0)[0]
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", RuntimeWarning)
      mean_nsat[idx_nonzero] = \
          ((mass[idx_nonzero] - self.param_dict['k'] * Mcut) / \
          M1)**self.param_dict['alpha']
#    mean_nsat = ((mass - self.param_dict['k'] * Mcut) / \
#        M1)**self.param_dict['alpha']

    if self.modulate_with_cenocc:
      mean_ncen = self.central_occupation_model.mean_occupation(**kwargs)
      mean_nsat *= mean_ncen

    return mean_nsat

  def get_default_parameters(self, threshold):
    '''
      Best-fit HOD parameters from Martin White
    '''
    param_dict = (
        {'logM1': 13.75,
        'logMcut': 11.5,
        'k': 0.5,
        'alpha': 0.5}
        )
    return param_dict


class MWSats2(occupation_model_template.OccupationComponent):
  '''
    The satellite occupation model used by Martin White:
    <Nsat> = [ ( Mhalo - M0 ) / M1 ]^alpha
  '''
  def __init__(self, threshold=model_defaults.default_luminosity_threshold, \
      prim_haloprop_key=model_defaults.prim_haloprop_key, redshift=0, \
      modulate_with_cenocc=False, cenocc_model=None, **kwargs):
    upper_occupation_bound = float("inf")
    super(MWSats2, self).__init__(gal_type='satellites', \
        threshold=threshold, upper_occupation_bound=upper_occupation_bound, \
        prim_haloprop_key=prim_haloprop_key, **kwargs)
    self.redshift = redshift
    self.param_dict = self.get_default_parameters(self.threshold)
    self.list_of_haloprops_needed = ['halo_id','halo_rhalf']

    if cenocc_model is None:
      cenocc_model = MWCens(prim_haloprop_key=prim_haloprop_key, \
          threshold=threshold)
    else:
      if modulate_with_cenocc is False:
        msg = ("You chose to input a `cenocc_model`, but you set the \n"
            "`modulate_with_cenocc` keyword to False, so your "
            "`cenocc_model` will have no impact on the model's behavior.\n"
            "Be sure this is what you intend before proceeding.\n")
        warnings.warn(msg)

    self.modulate_with_cenocc = modulate_with_cenocc
    if self.modulate_with_cenocc:
      try:
        assert isinstance(cenocc_model, \
            occupation_model_template.OccupationComponent)
      except AssertionError:
        msg = ('The input `cenocc_model` must be an instance of \n'
            '`OccupationComponent` or one of its sub-classes.\n')
        raise HalotoolsError(msg)

      self.central_occupation_model = cenocc_model
      self.param_dict.update(self.central_occupation_model.param_dict)

  def mean_occupation(self, **kwargs):
    if self.modulate_with_cenocc:
      for key, value in self.param_dict.items():
        if key in self.central_occupation_model.param_dict:
          self.central_occupation_model.param_dict[key] = value

    if 'table' in list(kwargs.keys()):
      mass = kwargs['table'][self.prim_haloprop_key]
    elif 'prim_haloprop' in list(kwargs.keys()):
      mass = np.atleast_1d(kwargs['prim_haloprop'])
    else:
      msg = ('\nYou must pass either a `table` or `prim_haloprop` argument \n'
          'to the `mean_occupation` function of the `MWSats` class.\n')
      raise HalotoolsError(msg)

    # Mcut = 10.**self.param_dict['logMcut']
    M0 = 10.**self.param_dict['logM0']
    M1 = 10.**self.param_dict['logM1']

    mean_nsat = np.zeros_like(mass)
    idx_nonzero = np.where(mass - M0 > 0)[0]
    with warnings.catch_warnings():
      warnings.simplefilter("ignore", RuntimeWarning)
      mean_nsat[idx_nonzero] = \
          ((mass[idx_nonzero] - M0) / \
          M1)**self.param_dict['alpha']

    if self.modulate_with_cenocc:
      mean_ncen = self.central_occupation_model.mean_occupation(**kwargs)
      mean_nsat *= mean_ncen

    return mean_nsat

  def get_default_parameters(self, threshold):
    '''
      Best-fit HOD parameters from Martin White
    '''
    param_dict = (
        {'logM1': 13.75,
        'logM0': 11.45,
        'alpha': 0.5}
        )
    return param_dict


class ContrerasCens(occupation_model_template.OccupationComponent):
  '''
    The central occupation model from Contreras (1301.3497)
  '''
  def __init__(self, threshold=model_defaults.default_luminosity_threshold, \
      prim_haloprop_key=model_defaults.prim_haloprop_key, redshift=0, \
      **kwargs):
    '''
      Examples
      --------
      cen_model = ContrerasCens()
      cen_model = ContrerasCens(threshold=-19.5)
      cen_model = ContrerasCens(prim_haloprop_key='halo_m200b')
    '''
    upper_occupation_bound = 1.0
    super(ContrerasCens, self).__init__(gal_type='centrals', \
        threshold=threshold, upper_occupation_bound=upper_occupation_bound, \
        prim_haloprop_key=prim_haloprop_key, **kwargs)
    self.redshift = redshift
    self.param_dict = self.get_default_parameters(self.threshold)

  def mean_occupation(self, **kwargs):
    if 'table' in list(kwargs.keys()):
      mass = kwargs['table'][self.prim_haloprop_key]
    elif 'prim_haloprop' in list(kwargs.keys()):
      mass = np.atleast_1d(kwargs['prim_haloprop'])
    else:
      msg = ('\nYou must pass either a `table` or `prim_haloprop` argument \n'
          'to the `mean_occupation` function of the `ContrerasCens` class.\n')
      raise HalotoolsError(msg)

    logM = np.log10(mass)
    fac = np.zeros_like(mass)
    idx = (logM < self.param_dict['logMc'])

    fac[idx] = (logM[idx] - self.param_dict['logMc'])/self.param_dict['siga']
    fac[~idx] = (logM[~idx] - self.param_dict['logMc'])/self.param_dict['sigb']

    mean_ncen = self.param_dict['Fb'] * (1 - self.param_dict['Fa']) * \
        np.exp(-0.5 * fac**2) + self.param_dict['Fa'] * (1 + erf(fac))
    return mean_ncen

  def get_default_parameters(self, threshold):
    '''
      Best-fit HOD parameters
    '''
    param_dict = (
        {'Fa': 4.502e-3,
        'Fb': 0.255,
        'logMc': 11.332,
        'siga': 0.1,
        'sigb': 0.323}
        )
    return param_dict


class ContrerasSats(occupation_model_template.OccupationComponent):
  '''
    The satellite occupation model from Contreras (1301.3497)
  '''
  def __init__(self, threshold=model_defaults.default_luminosity_threshold, \
      prim_haloprop_key=model_defaults.prim_haloprop_key, redshift=0, \
      modulate_with_cenocc=False, cenocc_model=None, **kwargs):
    upper_occupation_bound = float("inf")
    super(ContrerasSats, self).__init__(gal_type='satellites', \
        threshold=threshold, upper_occupation_bound=upper_occupation_bound, \
        prim_haloprop_key=prim_haloprop_key, **kwargs)
    self.redshift = redshift
    self.param_dict = self.get_default_parameters(self.threshold)

    if cenocc_model is None:
      cenocc_model = ContrerasCens(prim_haloprop_key=prim_haloprop_key, \
          threshold=threshold)
    else:
      if modulate_with_cenocc is False:
        msg = ("You chose to input a `cenocc_model`, but you set the \n"
            "`modulate_with_cenocc` keyword to False, so your "
            "`cenocc_model` will have no impact on the model's behavior.\n"
            "Be sure this is what you intend before proceeding.\n")
        warnings.warn(msg)

    self.modulate_with_cenocc = modulate_with_cenocc
    if self.modulate_with_cenocc:
      try:
        assert isinstance(cenocc_model, \
            occupation_model_template.OccupationComponent)
      except AssertionError:
        msg = ('The input `cenocc_model` must be an instance of \n'
            '`OccupationComponent` or one of its sub-classes.\n')
        raise HalotoolsError(msg)

      self.central_occupation_model = cenocc_model
      self.param_dict.update(self.central_occupation_model.param_dict)

  def mean_occupation(self, **kwargs):
    if self.modulate_with_cenocc:
      for key, value in self.param_dict.items():
        if key in self.central_occupation_model.param_dict:
          self.central_occupation_model.param_dict[key] = value

    if 'table' in list(kwargs.keys()):
      mass = kwargs['table'][self.prim_haloprop_key]
    elif 'prim_haloprop' in list(kwargs.keys()):
      mass = np.atleast_1d(kwargs['prim_haloprop'])
    else:
      msg = ('\nYou must pass either a `table` or `prim_haloprop` argument \n'
          'to the `mean_occupation` function of the `ContrerasSats` class.\n')
      raise HalotoolsError(msg)

    Mmin = 10.**self.param_dict['logMmin']

    mean_nsat = np.zeros_like(mass)
    mean_nsat = self.param_dict['Fs'] * (1 + erf((np.log10(mass) - \
        self.param_dict['logMmin']) / self.param_dict['deltaM'])) * \
        (mass / Mmin)**self.param_dict['alpha']

    if self.modulate_with_cenocc:
      mean_ncen = self.central_occupation_model.mean_occupation(**kwargs)
      mean_nsat *= mean_ncen

    return mean_nsat

  def get_default_parameters(self, threshold):
    '''
      Best-fit HOD parameters
    '''
    param_dict = (
        {'Fs': 3.502e-3,
        'logMmin': 11.511,
        'deltaM': 0.156,
        'alpha': 0.73}
        )
    return param_dict


class ABMWCens_IC(HeavisideAssembias, MWCens_IC):
  '''
    MWCens_IC model decorated with Heaviside Assembly Bias.
    New parameter: 'mean_occupation_centrals_assembias_param1' (-1 to 1)
  '''
  def __init__(self, sec_haloprop_key, **kwargs):
    # Initialize the base Occupation model (MWCens) first
    MWCens_IC.__init__(self, **kwargs)
    
    # Initialize the Assembly Bias wrapper
    HeavisideAssembias.__init__(self,
      method_name_to_decorate='mean_occupation',
      sec_haloprop_key=sec_haloprop_key,
      assembias_param1_key='AB_cen',
      lower_assembias_bound= 0, 
      upper_assembias_bound= 1.0,
      **kwargs)

class ABMWSats(HeavisideAssembias, MWSats):
  '''
    MWSats model decorated with Heaviside Assembly Bias.
  '''
  def __init__(self, sec_haloprop_key, **kwargs):
    
    # Initialize the base Occupation model (MWSats) first
    MWSats.__init__(self, **kwargs)
    
    # Initialize the Assembly Bias wrapper
    HeavisideAssembias.__init__(self,
      method_name_to_decorate='mean_occupation',
      sec_haloprop_key=sec_haloprop_key,
      assembias_param1_key='AB_sat',
      lower_assembias_bound= 0, 
      upper_assembias_bound=np.inf,
      **kwargs)
