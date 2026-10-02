"""Generate a four-bin DES Y3-like shape catalog from Abacus maps."""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger

from handler import PipeConfig
from runner import AbacusRunner


DES_Y3_N_EFF = {
    "tomo1": 1.476,
    "tomo2": 1.479,
    "tomo3": 1.484,
    "tomo4": 1.461,
}
DES_Y3_TOMO_LABELS = {
    "tomo1": 1,
    "tomo2": 2,
    "tomo3": 3,
    "tomo4": 4,
}
ABACUS_SOURCE_REDSHIFTS = tuple(index * 0.05 for index in range(1, 41))

DEFAULT_SHEAR_MAP_FMT = (
    "/Volumes/Elements/Abacus_Mock/gamma{:d}_rt_z{:.2f}.fits"
)
DEFAULT_OUTPUT_PATH = Path(
    "/Users/suqikuai777/Dataspace/Abacus_Mock/"
    "abacus_run_0_des_y3_4tomos_uniform.fits"
)


def build_des_y3_runner(
        *, shear_map_fmt=DEFAULT_SHEAR_MAP_FMT,
        mask_path=None, nofz_dir=None,
        output_path=DEFAULT_OUTPUT_PATH,
        redshift_src_list=ABACUS_SOURCE_REDSHIFTS):
    """Build the Abacus runner for the DES Y3 four-bin source sample."""
    extras_dir = Path(__file__).resolve().parents[1] / "extras"
    if mask_path is None:
        mask_path = (
            extras_dir
            / "masks/des_y3_geom/des_y3_science_mask_nside4096.fits"
        )
    if nofz_dir is None:
        nofz_dir = extras_dir / "NOfZ/desy3_nofzs"
    nofz_dir = Path(nofz_dir)

    config = PipeConfig(
        Lbox=900.0,
        Npart=832,
        redshift=0.5125,
        sigma_e=0.261,
        seed_SN_ini=0,
        sigma_phz=0.01,
        seed_Phz_ini=26120,
        seed_pos=0,
    )
    nofz_paths = {
        f"tomo{tomo}": nofz_dir / (
            "pzwei_sources_abacushuge_desy3_"
            f"tom{tomo}_dz0pt05.dat"
        )
        for tomo in range(1, 5)
    }

    return AbacusRunner.build_shape_runner(
        config=config,
        shear_map_fmt=str(shear_map_fmt),
        back_mask_fnames_dict={"DES-Y3": str(mask_path)},
        back_nofz_fnames_dict=nofz_paths,
        back_survey_labels_dict={"DES-Y3": 0},
        back_ngals_dict=DES_Y3_N_EFF.copy(),
        tomo_labels_dict=DES_Y3_TOMO_LABELS.copy(),
        redshift_src_list=list(redshift_src_list),
        shear_ofmt=str(output_path),
        position_method="random",
    )


def main():
    runner = build_des_y3_runner()
    Path(runner.shear_ofmt).parent.mkdir(parents=True, exist_ok=True)
    logger.info("Generate DES Y3-like Abacus shape catalog ...")
    catalog = runner.gen_mock_shear(save=True)
    logger.info(
        f"Done: {len(catalog)} galaxies written to {runner.shear_ofmt}"
    )


if __name__ == "__main__":
    main()
