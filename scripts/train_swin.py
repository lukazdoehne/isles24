"""
Train multi encoder Swin-UNETR
"""

from pathlib import Path
import re
import argparse
from dataclasses import asdict
import wandb
from isles.swin.config import SwinTrainConfig
from isles.swin.model import get_model
from isles.swin.training import train_swin, get_swin_dataloaders
from isles.swin.evaluation import final_evaluation
from isles.utils import generate_datalist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train multi encoder Swin-UNETR")
    parser.add_argument("--run-id", required=True, type=str)
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["cta", "cbf"],
        type=str,
        choices=["cta", "cbf", "cbv", "mtt", "tmax"],
    )
    parser.add_argument(
        "--model",
        default="MultiEncoderSwinUNETR",
        type=str,
        choices=["BaseSwinUNETR", "MultiEncoderSwinUNETR"],
    )
    parser.add_argument(
        "--crop-mode",
        default="label_classes",
        type=str,
        choices=["label_classes", "spatial"],
    )
    parser.add_argument("--max-epochs", default=300, type=int)
    parser.add_argument("--learning-rate", default=1e-4, type=float)
    parser.add_argument("--evaluation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = SwinTrainConfig(
        model=args.model,
        max_epochs=args.max_epochs,
        modalities=args.modalities,
        target_spacing=(1.0, 1.0, 1.0),
        roi_size=(64, 64, 64),
        learning_rate=args.learning_rate,
        crop_mode=args.crop_mode,
        crop_ratios=(1, 1),
        include_background=False,
        intensity_windows={
            "cta": [0, 90],
            "cbf": [0, 35],
            "cbv": [0, 10],
            "mtt": [0, 20],
            "tmax": [0, 7],
        },
        batch_size=1,
        val_interval=10,
        inspect_training=True,
        inspect_interval=25,
    )

    data_root = Path("/home/renku/work/data-local")
    pretrained_path = (
        data_root / "pretrained/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt"
    )
    run_dir = data_root / f"runs/{args.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    datalist = generate_datalist(
        data_root=data_root,
        target_dir=run_dir,
        modalities=config.modalities,
        brain_mask=True,
        val_fold=0,
    )

    wandb.init(
        project="ISLES",
        name=args.run_id,
        dir=run_dir,
        config={
            **asdict(config),
            "loss": "DiceCELoss",
            "optimizer": "AdamW",
            "scheduler": "WarmupCosineSchedule",
        },
        save_code=True,
    )
    artifact = wandb.Artifact("datalist", type="datalist")
    artifact.add_file(run_dir / "datalist.json", name="datalist.json")
    wandb.log_artifact(artifact)

    train_loader, val_loader = get_swin_dataloaders(datalist, config)
    model = get_model(config)
    model.load_pretrained_encoders(pretrained_path)

    train_swin(
        model=model,
        config=config,
        run_dir=run_dir,
        train_loader=train_loader,
        val_loader=val_loader,
        case_id_fn=lambda p: re.search(r"sub-stroke\d+", p).group(),
    )

    if args.evaluation:
        checkpoint_path = run_dir / "checkpoints/best_model.pt"
        eval_dir = run_dir / "evaluation"
        final_evaluation(
            checkpoint_path=checkpoint_path,
            val_loader=val_loader,
            config=config,
            out_dir=eval_dir,
            save_logits=True,
        )

        wandb.save(f"{eval_dir}/**/*", base_path=run_dir)
    wandb.finish()


if __name__ == "__main__":
    main()
