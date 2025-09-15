import numpy as np
from loguru import logger

survey_part_dict = {
    "lowzcmass": 0,
    "lowze2" : 1,
    "lowze3" : 2
}

survey_name = "lowzcmass"
### bossdata void catalog
fname = "catalogs/bossdata_lowzcmasstot_void.npy"

logger.info("Load catalog")

vcat = np.load(fname)
select = vcat["survey"] == survey_part_dict[survey_name]
vcat = vcat[select]

logger.info("Save to file")

np.savetxt(f"catalogs/tmp_bossdata_{survey_name}_void.txt", np.c_[vcat["ra"], vcat["dec"], vcat["z"]], fmt="%.4f %.4f %.4f")

### bossdata void random catalog
fname = f"catalogs/bossdata_{survey_name}_void_rand2.npy"

logger.info("Load catalog")

vcat = np.load(fname)

logger.info("Save to file")

np.savetxt(f"catalogs/tmp_bossdata_{survey_name}_void_rand2.txt", np.c_[vcat["ra"], vcat["dec"], vcat["z"]], fmt="%.4f %.4f %.4f")