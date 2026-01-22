"""
Train nnUnet
"""

from pathlib import Path
from isles.utils import generate_datalist
from isles.nnunet.core import (
    NNUNetConfig,
    convert_datalist_to_nnunet,
    run_preprocessing,
    train,
)


def main():
    run_id = "run-020"
    modalities = ["cta", "cbf", "cbv", "mtt", "tmax"]

    data_root = Path("/home/renku/work/data-local")
    run_dir = data_root / f"runs/{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    generate_datalist(
        data_root=data_root,
        target_dir=run_dir,
        modalities=modalities,
        brain_mask=True,
        val_fold=0,
    )

    config = NNUNetConfig(
        datalist_path=run_dir / "datalist.json",
        data_root=data_root / "nnunet",
        intensity_windows={
            "cta": [0, 90],
            "cbf": [0, 35],
            "cbv": [0, 10],
            "mtt": [0, 20],
            "tmax": [0, 7],
        },
    )

    convert_datalist_to_nnunet(config, force=False)
    run_preprocessing(config)
    train(config=config, run_id=run_id, run_dir=run_dir)


if __name__ == "__main__":
    main()
