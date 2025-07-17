#!/bin/bash

WDIR=/home/suchen/Program/CosmoGrid

### Load halos and apply HOD
pixi run python $WDIR/src/make_fore_gal.py

# ### Apply mask to galaxy catalog
# MAKE_SURVEY=/home/suchen/applications/make_survey/make_survey
# ISNAP=$WDIR/catalogs/HOD/cosmo_000001_run_0_HOD_0_run_0_snap.txt
# OGAL=$WDIR/catalogs/HOD/cosmo_000001_run_0_HOD_0_run_0_lcone.txt
# $MAKE_SURVEY $WDIR/cfgs/make-survey/make_survey_cosmo_000001.cfg $ISNAP $OGAL

### make background shear catalog
pixi run python $WDIR/src/make_back_gal.py

### match foreground and background galaxies
pixi run python $WDIR/src/match_fore_back.py