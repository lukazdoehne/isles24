"""
Pipeline to reproduce Ren, T. et al. How We Won the ISLES'24 Challenge by Preprocessing.

References
----------
.. 1. Ren, T. et al. How We Won the ISLES'24 Challenge by Preprocessing.
      Preprint at https://doi.org/10.48550/arXiv.2505.18424 (2025).

"""

import json
import os
import re
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections.abc import Sequence

from tqdm import tqdm
import nibabel as nib
import numpy as np
import torch
from monai.transforms import CropForeground
from skimage.exposure import equalize_hist
import wandb

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


# ======================================================================================
# Configuration
# ======================================================================================


@dataclass
class NNUNetConfig:
    """Configuration for nnU-Net training pipeline.

    Parameters
    ----------
    datalist_path : Path
        Path to MONAI-style datalist.json.
    data_root : Path
        Data root for the various nnU-Net folders.
    dataset_id : int
        nnU-Net dataset ID.
    dataset_name : str
        Dataset name.
    labels : dict[str, int]
        Label name to value mapping.
    planner : str
        nnU-Net planner class name.
    configuration : str
        nnU-Net configuration (2d, 3d_fullres, etc.).
    plans_name : str
        Plans identifier.
    wandb_project : str
        Weights & Biases project name.
    intensity_windows : dict[str, Sequence[float]] | None
        Per-modality intensity windows: {"modality": [min, max]}.
        If None, no intensity windowing is applied.
    histogram_equalization : bool
        Whether to apply histogram equalization.

    Attributes
    ----------
    num_folds : int
        Number of cross-validation folds. Initialized from datalist.
    val_fold : int | None
        Validation fold. Initialized from datalist. None if there's no validation
        fold in the datalist.
    modalities : list[str]
        Modalities used for the model. This is also the channel order.
    nnunet_raw : Path
        Base path for nnU-Net raw data.
    nnunet_preprocessed : Path
        Base path for nnU-Net preprocessed data.
    nnunet_results : Path
        Base path for nnU-Net results.

    """

    datalist_path: Path
    data_root: Path

    dataset_id: int = 100
    dataset_name: str = "ISLES2024"
    labels: dict[str, int] = field(
        default_factory=lambda: {"background": 0, "lesion": 1}
    )

    planner: str = "nnUNetPlannerResEncL"
    configuration: str = "3d_fullres"
    plans_name: str = "nnUNetResEncUNetLPlans"

    wandb_project: str = "ISLES"

    intensity_windows: dict[str, Sequence[float]] | None = None
    histogram_equalization: bool = False

    num_folds: int = field(init=False)
    val_fold: int | None = field(init=False, default=None)
    modalities: list[str] = field(init=False)

    nnunet_raw: Path = field(init=False)
    nnunet_preprocessed: Path = field(init=False)
    nnunet_results: Path = field(init=False)

    def __post_init__(self) -> None:
        self.datalist_path = Path(self.datalist_path)
        self.data_root = Path(self.data_root)
        self.nnunet_raw = self.data_root / "raw"
        self.nnunet_preprocessed = self.data_root / "preprocessed"
        self.nnunet_results = self.data_root / "results"

        with open(self.datalist_path, "r") as file:
            datalist = json.load(file)
        cases = datalist["training"] + datalist["validation"]
        self.num_folds = max([max([case["fold"] for case in cases]) + 1])
        self.modalities = datalist["modalities"]
        try:
            self.val_fold = datalist["validation"][0]
        except IndexError:
            print("No validation fold defined in datalist.")

    @property
    def dataset_dir(self) -> Path:
        return self.nnunet_raw / f"Dataset{self.dataset_id:03d}_{self.dataset_name}"

    @property
    def images_tr(self) -> Path:
        return self.dataset_dir / "imagesTr"

    @property
    def labels_tr(self) -> Path:
        return self.dataset_dir / "labelsTr"

    def set_environment(self) -> None:
        """Set nnU-Net environment variables."""
        os.environ["nnUNet_raw"] = str(self.nnunet_raw)
        os.environ["nnUNet_preprocessed"] = str(self.nnunet_preprocessed)
        os.environ["nnUNet_results"] = str(self.nnunet_results)

    def to_json(self, path: str | Path) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)


# ======================================================================================
# Custom Trainer
# ======================================================================================


def create_wandb_trainer_class(
    wandb_project: str,
    run_id: str,
    run_dir: Path,
    config: NNUNetConfig,
) -> type[nnUNetTrainer]:
    """
    Create a W&B-enabled nnUNetTrainer subclass.

    Parameters
    ----------
    wandb_project : str
        W&B project name.
    wandb_name : str
        W&B run name.
    run_dir : Path
        Directory with run data.
    config : NNUNetConfig
        Config class for NNUnet.

    Returns
    -------
    type
        nnUNetTrainer subclass with W&B logging.
    """

    class WandbnnUNetTrainer(nnUNetTrainer):
        """nnUNetTrainer with W&B logging."""

        # Initialize wandb run and log config and artifacts
        def on_train_start(self) -> None:
            super().on_train_start()
            wandb.init(
                project=wandb_project,
                name=run_id,
                dir=run_dir,
                config={
                    "architecture": "ResEncUNet",
                    "fold": self.fold,
                    "patch_size": list(self.configuration_manager.patch_size),
                    "batch_size": self.configuration_manager.batch_size,
                    "plans": self.plans_manager.plans_name,
                    **asdict(config),
                },
                save_code=True,
            )
            artifact = wandb.Artifact("datalist", type="datalist")
            artifact.add_file(run_dir / "datalist.json", name="datalist.json")
            wandb.log_artifact(artifact)
            config.to_json(run_dir / "config.json")
            wandb.save(run_dir / "config.json", base_path=run_dir)

        def on_epoch_end(self) -> None:
            super().on_epoch_end()
            logs = self.logger.my_fantastic_logging
            metrics = {"epoch": self.current_epoch}
            if logs["train_losses"]:
                metrics["train/loss"] = logs["train_losses"][-1]
            if logs["val_losses"]:
                metrics["val/loss"] = logs["val_losses"][-1]
            if logs["ema_fg_dice"]:
                metrics["val/dice"] = logs["ema_fg_dice"][-1]
            if (
                logs.get("dice_per_class_or_region")
                and logs["dice_per_class_or_region"]
            ):
                for i, score in enumerate(logs["dice_per_class_or_region"][-1]):
                    metrics[f"val/dice_class_{i}"] = score
            metrics["learning_rate"] = self.optimizer.param_groups[0]["lr"]
            wandb.log(metrics)

        def on_train_end(self) -> None:
            super().on_train_end()
            wandb.finish()

    return WandbnnUNetTrainer


# ======================================================================================
# Dataset Conversion
# ======================================================================================


def extract_case_id(filepath: str) -> str:
    """Extract case ID (e.g., 'sub-stroke0001') from filepath."""
    match = re.search(r"(sub-stroke\d+)", filepath)
    return match.group(1)


def compute_crop_slices(brain_mask: np.ndarray) -> tuple[slice, ...]:
    """Compute crop slices from brain mask."""
    crop_fg = CropForeground(select_fn=lambda x: x > 0.5)
    box_start, box_end = crop_fg.compute_bounding_box(brain_mask)
    return tuple(slice(s, e) for s, e in zip(box_start, box_end))


def preprocess_image(
    image: np.ndarray,
    crop_slices: tuple[slice, ...],
    intensity_window: tuple[float, float] | None = None,
    histogram_equalization: bool = False,
) -> np.ndarray:
    """Apply intensity preprocessing to image data."""

    image = image[crop_slices]

    if intensity_window is not None:
        lo, hi = intensity_window
        image = np.clip(image, lo, hi)
        image = (image - lo) / (hi - lo)

    if histogram_equalization:
        mask = image > 0
        if mask.any():
            image[mask] = equalize_hist(image[mask])

    return image


def convert_datalist_to_nnunet(config: NNUNetConfig, force: bool = False) -> None:
    """
    Convert MONAI datalist.json to nnU-Net raw format.

    Parameters
    ----------
    config : NNUNetConfig
        Pipeline configuration.
    force : bool
        If True, remove existing dataset and reconvert.
    """
    if config.dataset_dir.exists():
        if force:
            shutil.rmtree(config.dataset_dir)
        else:
            print(
                f"Dataset exists at {config.dataset_dir}. Use force=True to reconvert."
            )
            return

    config.images_tr.mkdir(parents=True, exist_ok=True)
    config.labels_tr.mkdir(parents=True, exist_ok=True)

    with open(config.datalist_path) as f:
        datalist = json.load(f)

    cases = datalist["training"] + datalist["validation"]
    case_folds: dict[str, int] = {}

    for case in tqdm(cases, "Processing cases"):
        case_id = extract_case_id(case["image"][0])
        case_folds[case_id] = case["fold"]

        brain_mask = nib.load(case["brain_mask"]).get_fdata()
        crop_slices = compute_crop_slices(brain_mask)

        for ch_idx, img_path in enumerate(case["image"]):
            modality = config.modalities[ch_idx]
            img = nib.load(img_path)
            img_data = img.get_fdata()

            # Apply brain mask
            img_data[brain_mask == 0] = 0

            img_data = preprocess_image(
                img_data,
                crop_slices,
                intensity_window=config.intensity_windows.get(modality),
                histogram_equalization=config.histogram_equalization,
            )
            out_path = config.images_tr / f"{case_id}_{ch_idx:04d}.nii.gz"
            nib.save(nib.Nifti1Image(img_data, img.affine, img.header), out_path)

        # Crop labels
        label_img = nib.load(case["label"])
        label_data = label_img.get_fdata()[crop_slices].astype(np.uint8)
        nib.save(
            nib.Nifti1Image(label_data, label_img.affine, label_img.header),
            config.labels_tr / f"{case_id}.nii.gz",
        )

    # Create dataset.json
    dataset_json = {
        "channel_names": {
            str(i): modality for i, modality in enumerate(config.modalities)
        },
        "labels": config.labels,
        "numTraining": len(cases),
        "file_ending": ".nii.gz",
    }
    with open(config.dataset_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)

    # splits_final.json
    splits = []
    for fold in range(config.num_folds):
        val_cases = [cid for cid, f in case_folds.items() if f == fold]
        train_cases = [cid for cid, f in case_folds.items() if f != fold]
        splits.append({"train": train_cases, "val": val_cases})

    with open(config.dataset_dir / "splits_final.json", "w") as f:
        json.dump(splits, f, indent=4)

    print(f"Done: {config.dataset_dir}")


# ======================================================================================
# Preprocessing
# ======================================================================================


def run_preprocessing(config: NNUNetConfig) -> None:
    """Run nnU-Net planning and preprocessing."""

    from nnunetv2.experiment_planning.plan_and_preprocess_api import (
        extract_fingerprints,
        plan_experiments,
        preprocess,
    )

    print("Extracting fingerprints...")
    extract_fingerprints(dataset_ids=[config.dataset_id])

    print(f"Planning with {config.planner}...")
    plan_experiments(
        dataset_ids=[config.dataset_id],
        experiment_planner_class_name=config.planner,
    )

    print(f"Preprocessing {config.configuration}...")
    preprocess(
        dataset_ids=[config.dataset_id],
        plans_identifier=config.plans_name,
        configurations=[config.configuration],
        num_processes=[8],
    )

    print("Done.")


# ======================================================================================
# Training
# ======================================================================================


def train(
    config: NNUNetConfig,
    run_id: str,
    run_dir: Path,
) -> None:
    """
    Train nnU-Net with W&B logging.

    Parameters
    ----------
    config : NNUNetConfig
        Pipeline configuration.
    run_id : str
        W&B run name.
    run_dir : Path
        Directory with the run files.

    """
    config.set_environment()

    from nnunetv2.run import run_training as run_training_mod

    trainer_class = create_wandb_trainer_class(
        wandb_project=config.wandb_project,
        run_id=run_id,
        run_dir=run_dir,
        config=config,
    )

    # Patch the class lookup
    original_find = run_training_mod.recursive_find_python_class

    def patched_find(folder: str, trainer_name: str, current_module: str) -> type:
        if trainer_name == "WandbnnUNetTrainer":
            return trainer_class
        return original_find(folder, trainer_name, current_module)

    run_training_mod.recursive_find_python_class = patched_find

    try:
        run_training_mod.run_training(
            dataset_name_or_id=config.dataset_id,
            configuration=config.configuration,
            fold=config.val_fold,
            trainer_class_name="WandbnnUNetTrainer",
            plans_identifier=config.plans_name,
            pretrained_weights=None,
            num_gpus=1,
            use_compressed_data=False,
            export_validation_probabilities=False,
            continue_training=False,
            only_run_validation=False,
            disable_checkpointing=False,
            device=torch.device("cuda"),
        )
    finally:
        run_training_mod.recursive_find_python_class = original_find
