"""
Unified sliding window inference script.

Results are saved under:
    {run_dir}/inference-{checkpoint}/{out_subdir}/

where out_subdir encodes the inference parameters used.

Examples
--------
# Default conditions on last model (constant + gaussian, overlap 0.5, no logits):
python inference.py --run_ids run-021 run-022

# Both checkpoints in one sweep:
python inference.py --run_ids run-021 run-022 --checkpoint best last --save_logits

# Crop-margin mode (replaces inference_crop.py):
python inference.py --run_ids run-021 --checkpoint best --crop_margin 10 --save_logits

# Custom sweep:
python inference.py --run_ids run-021 --overlap 0.5 0.75 --blend_mode gaussian --checkpoint last
"""

import logging
import argparse
import json
from itertools import product
from pathlib import Path
from contextlib import contextmanager

from tqdm import tqdm

from isles.swin.config import SwinTrainConfig
from isles.swin.evaluation import final_evaluation
from isles.swin.training import get_dataloader
from isles.swin.transforms import get_val_transforms


@contextmanager
def suppress_logs(level: int = logging.INFO):
    logging.disable(level)
    try:
        yield
    finally:
        logging.disable(logging.NOTSET)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sliding window inference with configurable parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run_ids",
        nargs="+",
        required=True,
        help="Run IDs to evaluate (e.g. run-021 run-022).",
    )
    parser.add_argument(
        "--checkpoint",
        nargs="+",
        choices=["best", "last"],
        default=["last"],
        help="Checkpoint(s) to evaluate. Accepts both: --checkpoint best last.",
    )
    parser.add_argument(
        "--overlap",
        nargs="+",
        type=float,
        default=[0.5],
        help="Sliding window overlap(s).",
    )
    parser.add_argument(
        "--blend_mode",
        nargs="+",
        choices=["constant", "gaussian"],
        default=["constant", "gaussian"],
        help="Blend mode(s) for sliding window inference.",
    )
    parser.add_argument(
        "--crop_margin",
        type=int,
        default=None,
        help=(
            "If set, use cropped inference with this margin size. "
            "Forces blend_mode to constant and ignores --blend_mode."
        ),
    )
    parser.add_argument(
        "--save_logits",
        action="store_true",
        help="Save raw logits alongside predictions.",
    )
    parser.add_argument(
        "--data_root",
        type=Path,
        default=Path("/home/renku/work/data-local"),
        help="Root directory containing runs/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # crop_margin forces constant blend; sweeping blend_mode is meaningless in that case
    blend_modes = ["constant"] if args.crop_margin is not None else args.blend_mode

    combinations = list(product(args.run_ids, args.checkpoint, args.overlap, blend_modes))

    for run_id, ckpt, o, b in tqdm(combinations, desc="Evaluating runs"):
        run_dir = args.data_root / f"runs/{run_id}"

        config = SwinTrainConfig.from_json(run_dir / "config.json")
        with open(run_dir / "datalist.json") as file:
            datalist = json.load(file)

        val_loader = get_dataloader(
            datalist=datalist,
            key="validation",
            transforms=get_val_transforms(config),
            batch_size=config.batch_size,
            cache_rate=0.0,
        )

        checkpoint_file = f"{ckpt}_model.pt"
        checkpoint_path = run_dir / "checkpoints" / checkpoint_file
        out_dir = run_dir / f"inference-{ckpt}"

        if args.crop_margin is not None:
            out_dir = out_dir / f"overlap_{o}-crop_{args.crop_margin}"
        else:
            out_dir = out_dir / f"overlap_{o}-blend_{b}"

        params = {
            "checkpoint": checkpoint_file,
            "val_overlap_final": o,
            "inferer_blend_mode": b,
            "inferer_crop_margin": args.crop_margin,
        }
        final_evaluation(
            checkpoint_path=checkpoint_path,
            val_loader=val_loader,
            config=config,
            out_dir=out_dir,
            save_logits=args.save_logits,
            val_overlap_final=o,
            inferer_blend_mode=b,
            inferer_crop_margin=args.crop_margin,
        )
        with open(out_dir / "params.json", "w") as file:
            json.dump(params, file, indent=2)


if __name__ == "__main__":
    with suppress_logs():
        main()
