"""
Perform a sweep of parameter for sliding windows inference to
identify the cause of possible grid artifacts.
"""

from pathlib import Path
import json
from itertools import product

from isles.utils import patch_datalist
from isles.swin.config import SwinTrainConfig
from isles.swin.transforms import get_val_transforms
from isles.swin.training import get_dataloader
from isles.swin.evaluation import final_evaluation


def main():
    roi_size = [96, 64, 32]
    overlap = [0.1, 0.2, 0.4]
    blend_mode = ["constant", "gaussian"]

    data_root = Path("/home/renku/work/data-local")
    run_id = "run-016"
    run_dir = data_root / f"runs/{run_id}"
    sweep_dir = run_dir / "inference-sweep"
    checkpoint_path = run_dir / "checkpoints/best_model.pt"

    config = SwinTrainConfig.from_json(run_dir / "config.json")
    with open(run_dir / "datalist.json") as file:
        datalist = json.load(file)
    datalist = patch_datalist(datalist, process_guide=config.fg_guide)

    val_loader = get_dataloader(
        datalist=datalist,
        key="validation",
        transforms=get_val_transforms(config),
        batch_size=config.batch_size,
        cache_rate=0.0,
    )

    for r, o, b in product(roi_size, overlap, blend_mode):
        print(f"roi_size={r}, overlap={o}, blend_mode={b}")
        out_dir = sweep_dir / f"roi_{r}-overlap_{o}-blend_{b}"
        final_evaluation(
            checkpoint_path=checkpoint_path,
            val_loader=val_loader,
            config=config,
            out_dir=out_dir,
            roi_size=r,
            val_overlap_final=o,
            inferer_blend_mode=b,
        )
        params = {
            "roi_size": r,
            "val_overlap_final": o,
            "inferer_blend_mode": b,
        }
        with open(out_dir / "params.json", "w") as file:
            json.dump(params, indent=2)


if __name__ == "__main__":
    main()
