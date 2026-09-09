# FCFC_2PT config: projected 2PCF (w_p) of FastPM LOWZ-NGC mock, cosmo_000000
# Placeholders @DATA@ / @RAND@ / @OUT@ / @HOD@ filled by the job script.

##########################################
#  Specifications of the input catalogs  #
##########################################

CATALOG         = [ @DATA@, @RAND@ ]
CATALOG_LABEL   = [ D, R ]
CATALOG_TYPE    = [ 1, 1 ]
SELECTION       = [ ${survey} == 0, 1 == 1 ]
POSITION        = [ ${ra}, ${dec}, ${zrsd}, ${ra}, ${dec}, ${zrsd} ]
WEIGHT          = [ ${w}, ${w} ]
COORD_CONVERT   = T

##################################################
#  Fiducial cosmology for coordinate conversion  #
##################################################

OMEGA_M         = 0.200614
OMEGA_LAMBDA    = 0.799386

################################################################
#  Configurations for the 2-point correlation function (2PCF)  #
################################################################

BINNING_SCHEME  = 2
PAIR_COUNT      = [ DD, DR, RR ]
PAIR_COUNT_FILE = [ @OUT@/pc_dd_hod@HOD@.txt, @OUT@/pc_dr_hod@HOD@.txt, @OUT@/pc_rr.txt ]
CF_ESTIMATOR    = (DD - 2 * DR + RR) / RR
CF_OUTPUT_FILE  = @OUT@/cf_hod@HOD@.txt
PROJECTED_CF    = T
PROJECTED_FILE  = @OUT@/wp_hod@HOD@.txt

#############################
#  Definitions of the bins  #
#############################

SEP_BIN_FILE    = @OUT@/sep_bins.txt
PI_BIN_MIN      = 0
PI_BIN_MAX      = 100
PI_BIN_SIZE     = 2

####################
#  Other settings  #
####################

OUTPUT_FORMAT   = 1
OVERWRITE       = 1
VERBOSE         = T
