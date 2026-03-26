"""
Code for multi encoder Swin-UNETR
"""

from pathlib import Path
import csv
from collections.abc import Callable

from tqdm import tqdm
import wandb

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import nibabel as nib

from monai.inferers import SlidingWindowInferer
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.data import DataLoader
from monai.optimizers import WarmupCosineSchedule

from isles.io import get_dataloader
from isles.swin.config import SwinTrainConfig
from isles.swin.model import MultiEncoderSwinUNETR
from isles.swin.checkpoint import save_checkpoint
from isles.swin.transforms import (
    get_train_transforms,
    get_val_transforms,
)


def _center_of_mass_normalized(arr: np.ndarray) -> np.ndarray:
    """Compute center of mass of a 3D binary or probability array in normalized
    coordinates [0, 1]^3.

    Parameters
    ----------
    arr : np.ndarray
        3D array of shape ``(H, W, D)``.

    Returns
    -------
    np.ndarray
        Normalized CoM of shape ``(3,)``, or ``[nan, nan, nan]`` if the array is empty.
    """
    total = arr.sum()
    if total == 0:
        return np.full(arr.ndim, np.nan)
    indices = np.indices(arr.shape)
    com = np.array([(arr * indices[i]).sum() / total for i in range(arr.ndim)])
    return com / (np.array(arr.shape) - 1)


class TrainingInspector:
    """Accumulates spatial bias statistics and saves inspection data during training.

    Maintains a running foreground probability heatmap across all training patches
    and logs per-patch center-of-mass statistics to a CSV. Optionally saves NIfTI
    files for visual inspection at specified intervals.

    Parameters
    ----------
    out_dir : Path
        Root directory for all inspection outputs.
    roi_size : tuple[int, ...]
        Spatial size of training patches, used to initialise the heatmap.
    case_id_fn : Callable[[str], str] | None
        Function mapping a file path string to a case identifier string.
        If None, the filename stem is used.
    save_patches : bool
        Whether to save NIfTI patch files on each ``save`` call. Default True.
    """

    def __init__(
        self,
        out_dir: Path,
        roi_size: tuple[int, ...],
        case_id_fn: Callable[[str], str] | None = None,
        save_patches: bool = True,
    ) -> None:
        self.out_dir = out_dir
        self.save_patches = save_patches
        self.case_id_fn = case_id_fn or (lambda p: Path(p).stem.split(".")[0])
        self.roi_size = roi_size

        self.heatmap_dir = self.out_dir / "heatmaps"
        self.heatmap_dir.mkdir(exist_ok=True, parents=True)

        self._com_csv_path = out_dir / "com_stats.csv"
        self._com_header_written = self._com_csv_path.exists()

        self._reset_heatmaps()

    def _reset_heatmaps(self) -> None:
        """Reset accumulated heatmaps"""
        self.heatmap_gt_mask = np.zeros(self.roi_size, dtype=np.float64)
        self.heatmap_pred_mask = np.zeros(self.roi_size, dtype=np.float64)
        self.heatmap_pred_prob = np.zeros(self.roi_size, dtype=np.float64)
        self.n_patches = 0

    def update(self, batch: dict, logits: torch.Tensor) -> None:
        """Accumulate heatmap and center of mass (CoM) statistics for one training batch.

        Should be called after the forward pass on every batch, before
        the backward pass or optimizer step.

        Parameters
        ----------
        batch : dict
            Training batch dict with ``"image"`` and ``"label"`` MetaTensors.
        logits : torch.Tensor
            Raw model logits of shape ``(B, num_classes, H, W, D)``, on any device.
        """
        probs_fg = torch.softmax(logits.float(), dim=1)[:, 1].detach().cpu().numpy()
        self.heatmap_pred_prob += probs_fg.sum(axis=0)
        self.n_patches += probs_fg.shape[0]

        filenames = batch["image"].meta["filename_or_obj"]
        labels = batch["label"]
        patch_counters: dict[str, int] = {}
        rows = []

        for i in range(probs_fg.shape[0]):
            filename = (
                filenames[i] if isinstance(filenames, (list, tuple)) else filenames
            )
            case_id = self.case_id_fn(str(filename))

            patch_idx = patch_counters.get(case_id, 0)
            patch_counters[case_id] = patch_idx + 1

            # Calculate ground truth and prediction center of mass
            gt_label = labels[i, 0].float().cpu().numpy()
            gt_com = _center_of_mass_normalized(gt_label)
            pred_mask = (probs_fg[i] > 0.5).astype(np.float32)
            pred_com = _center_of_mass_normalized(pred_mask)

            # Calculate patch dice score
            intersection = (gt_label * pred_mask).sum()
            denom = gt_label.sum() + pred_mask.sum()
            patch_dice = float(2 * intersection / denom) if denom > 0 else float("nan")

            # Update accumulated mask heatmaps
            self.heatmap_gt_mask += gt_label
            self.heatmap_pred_mask += pred_mask

            rows.append(
                {
                    "case_id": case_id,
                    "patch_idx": patch_idx,
                    "gt_empty": np.isnan(gt_com[0]),
                    "pred_empty": np.isnan(pred_com[0]),
                    "gt_com_x": gt_com[0],
                    "gt_com_y": gt_com[1],
                    "gt_com_z": gt_com[2],
                    "pred_com_x": pred_com[0],
                    "pred_com_y": pred_com[1],
                    "pred_com_z": pred_com[2],
                    "gt_fg_voxels": int(gt_label.sum()),
                    "pred_fg_voxels": int(pred_mask.sum()),
                    "patch_dice": patch_dice,
                }
            )

        with open(self._com_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            if not self._com_header_written:
                writer.writeheader()
                self._com_header_written = True
            writer.writerows(rows)

    @torch.no_grad()
    def save(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        epoch: int,
        config: SwinTrainConfig,
    ) -> None:
        """Save heatmap and optionally NIfTI patches for the current epoch.

        The model is restored to its prior training/eval state after the call.

        Parameters
        ----------
        model : torch.nn.Module
            Segmentation network.
        train_loader : DataLoader
            Training dataloader. The first batch is used for patch saving.
        epoch : int
            Current epoch number (1-indexed), used for directory naming.
        config : SwinTrainConfig
            Training configuration.
        """

        # Heatmap: save current interval's accumulation, then reset
        def save_heatmap(heatmap: np.ndarray, filename: str) -> None:
            """Save heatmap to self.heatmap_dir/filename"""
            heatmap_mean = (heatmap / self.n_patches).astype(np.float32)
            np.save(self.heatmap_dir / filename, heatmap_mean)

        if self.n_patches > 0:
            save_heatmap(
                self.heatmap_gt_mask, f"heatmap_gt_mask_epoch_{epoch:04d}.npy"
            )
            save_heatmap(
                self.heatmap_pred_mask, f"heatmap_pred_mask_epoch_{epoch:04d}.npy"
            )
            save_heatmap(
                self.heatmap_pred_prob, f"heatmap_pred_prob_epoch_{epoch:04d}.npy"
            )
            self._reset_heatmaps()

        if not self.save_patches:
            return

        was_training = model.training
        model.eval()
        device = torch.device(config.device)

        batch = next(iter(train_loader))
        images: torch.Tensor = batch["image"]
        labels: torch.Tensor = batch["label"]

        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=config.amp):
            logits = model(images.to(device))

        logits = logits.float().cpu()
        epoch_dir = self.out_dir / f"patches/epoch_{epoch:04d}"
        epoch_dir.mkdir(parents=True, exist_ok=True)

        filenames = images.meta["filename_or_obj"]
        affines = images.meta["affine"]
        patch_counters: dict[str, int] = {}

        for i in range(images.shape[0]):
            filename = (
                filenames[i] if isinstance(filenames, (list, tuple)) else filenames
            )
            case_id = self.case_id_fn(str(filename))

            patch_idx = patch_counters.get(case_id, 0)
            patch_counters[case_id] = patch_idx + 1

            prefix = epoch_dir / f"{case_id}_{patch_idx:02d}"
            affine_np: np.ndarray = affines[i].numpy()

            nib.save(
                nib.Nifti1Image(
                    images[i].float().numpy().transpose(1, 2, 3, 0), affine_np
                ),
                f"{prefix}_image.nii.gz",
            )
            nib.save(
                nib.Nifti1Image(
                    labels[i].numpy().squeeze(0).astype(np.uint8), affine_np
                ),
                f"{prefix}_gt.nii.gz",
            )
            nib.save(
                nib.Nifti1Image(logits[i].numpy().transpose(1, 2, 3, 0), affine_np),
                f"{prefix}_logits.nii.gz",
            )
            nib.save(
                nib.Nifti1Image(
                    logits[i].numpy().argmax(axis=0).astype(np.uint8), affine_np
                ),
                f"{prefix}_pred.nii.gz",
            )

        model.train(was_training)


def _train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: SwinTrainConfig,
    inspector: TrainingInspector | None = None,
) -> float:
    """
    Run one training epoch.

    Returns
    -------
    float
        Mean loss for the epoch.
    """
    model.train()
    epoch_loss = 0.0
    device = torch.device(config.device)

    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.max_epochs}")
    for batch in train_pbar:
        image = batch["image"].to(device)
        label = batch["label"].to(device)

        optimizer.zero_grad()

        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=config.amp):
            logits = model(image)
            loss = loss_fn(logits, label)

        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()

        if inspector is not None:
            inspector.update({"image": image, "label": label}, logits)

        epoch_loss += loss.item()
        train_pbar.set_postfix(loss=f"{loss.item():.4f}")

    return epoch_loss / len(train_loader)


@torch.no_grad()
def _validate_epoch(
    model: torch.nn.Module,
    val_loader: DataLoader,
    loss_fn: torch.nn.Module,
    dice_metric: DiceMetric,
    inferer: SlidingWindowInferer,
    config: SwinTrainConfig,
) -> tuple[float, float]:
    """
    Run one validation epoch.

    Returns
    -------
    tuple[float, float]
        (mean_dice, mean_loss)
    """
    model.eval()
    val_loss = 0.0
    device = torch.device(config.device)

    for batch in tqdm(val_loader, desc="Validating", leave=False):
        image = batch["image"].to(device)
        label = batch["label"].to(device)

        with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=config.amp):
            logits = inferer(image, model)
            loss = loss_fn(logits, label)

        pred = logits.argmax(dim=1, keepdim=True)
        dice_metric(y_pred=pred, y=label)
        val_loss += loss.item()

    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()

    return mean_dice, val_loss / len(val_loader)


def train_swin(
    model: MultiEncoderSwinUNETR,
    config: SwinTrainConfig,
    run_dir: Path | str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    upload_checkpoints: bool = True,
    case_id_fn: Callable[[str], str] | None = None,
) -> None:
    """
    Train a binary segmentation model.

    Parameters
    ----------
    model : MultiEncoderSwinUNETR
        Segmentation network with out_channels=1.
    config : TrainConfig
        Training configuration.
    run_dir : Path | str
        Directory where to save artifacts.
    train_loader : DataLoader
        Training data loader.
    val_loader : DataLoader
        Validation data loader.
    upload_checkpoints : bool
        Whether to upload the best checkpoint to Weights and Biases.
        Default is True.
    case_id_fn : Callable[[str], str] | None
        Function used to extract case name from the file names during training
        inspection, only performed when config.inspect_training is True.
        If None, uses the filename stem. Default is None.
    """
    if isinstance(run_dir, str):
        run_dir = Path(run_dir)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Dump config to JSON and log to WandB
    config_path = run_dir / "config.json"
    config.to_json(config_path)
    wandb.save(config_path, base_path=run_dir)

    device = torch.device(config.device)
    model = model.to(device)

    # === Define optimizer, scheduler, losses, etc. ===
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = WarmupCosineSchedule(
        optimizer=optimizer,
        t_total=config.max_epochs,
        warmup_steps=int(config.max_epochs * config.warmup_ratio),
        warmup_multiplier=0.1,
    )
    inferer = SlidingWindowInferer(
        roi_size=config.roi_size,
        sw_batch_size=config.inferer_batch_size,
        overlap=config.val_overlap,
        mode="gaussian",
    )
    inspector: TrainingInspector | None = None
    if config.inspect_training:
        inspector = TrainingInspector(
            out_dir=run_dir / "training-inspection",
            roi_size=tuple(config.roi_size),
            case_id_fn=case_id_fn,
        )

    loss_fn = get_loss_function(config)
    dice_metric = get_dice_metric(config)

    best_dice = -1.0

    for epoch in range(config.max_epochs):
        # === Training ===
        train_loss = _train_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epoch=epoch,
            config=config,
            inspector=inspector,
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        metrics = {
            "train/loss": train_loss,
            "train/lr": current_lr,
            "epoch": epoch + 1,
        }

        # === Optional inspection of training patches ===
        if inspector is not None and (
            (epoch + 1) % config.inspect_interval == 0
            or epoch == config.max_epochs
            or epoch == 0
        ):
            inspector.save(
                model=model,
                train_loader=train_loader,
                epoch=epoch + 1,
                config=config,
            )

        # === Validation ===
        if (
            (epoch + 1) % config.val_interval == 0
            or epoch == config.max_epochs
            or epoch == 0
        ):
            dice, val_loss = _validate_epoch(
                model=model,
                val_loader=val_loader,
                loss_fn=loss_fn,
                dice_metric=dice_metric,
                inferer=inferer,
                config=config,
            )
            torch.cuda.empty_cache()

            metrics["val/loss"] = val_loss
            metrics["val/dice"] = dice
            print(
                f"Epoch {metrics['epoch']}: "
                f"train_loss={metrics['train/loss']:.4f}, "
                f"val_loss={metrics['val/loss']:.4f}, "
                f"dice={metrics['val/dice']:.4f}"
            )

            # === Checkpointing ===
            is_best = dice > best_dice
            if is_best:
                best_dice = dice
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                current_dice=dice,
                best_dice=best_dice,
                is_best=is_best,
                config=config,
            )

        wandb.log(metrics)

    # Upload final and best model to WandB
    if upload_checkpoints:
        wandb.save(checkpoint_dir / "best_model.pt", base_path=run_dir)
        wandb.save(checkpoint_dir / "last_model.pt", base_path=run_dir)


def get_swin_dataloaders(datalist: dict, config: SwinTrainConfig) -> tuple[DataLoader]:
    """Get dataloader for training multi-encoder Swin-UNETR.

    Parameters
    ----------
    datalist : dict
        Datalist as dictionary. It should have the keys "training" and "validation".
    config : SwinTrainConfig
        Configuration for training multi-encoder Swin-UNETR

    Returns
    -------
    (train_loader, val_loader): tuple[DataLoaders]
        Training, validation dataloaders.
    """
    train_loader = get_dataloader(
        datalist=datalist,
        key="training",
        transforms=get_train_transforms(config),
        batch_size=config.batch_size,
    )

    val_loader = get_dataloader(
        datalist=datalist,
        key="validation",
        transforms=get_val_transforms(config),
        batch_size=config.batch_size,
    )

    return train_loader, val_loader


def get_loss_function(config: SwinTrainConfig) -> DiceCELoss:
    """
    Get loss function for softmax segmentation.

    This function could be extended to support different kinds of losses.
    Loss function expects:
        - predictions: [B, C, H, W, D] raw logits
        - labels: [B, 1, H, W, D] integer class indices
    """
    return DiceCELoss(
        include_background=config.include_background,
        to_onehot_y=True,
        softmax=True,
        squared_pred=False,
        smooth_nr=1e-5,
        smooth_dr=1e-5,
    )


def get_dice_metric(config: SwinTrainConfig) -> DiceMetric:
    """
    Dice metric excluding background class.

    Reports mean Dice across foreground classes only.
    """
    return DiceMetric(
        include_background=config.include_background,
        reduction="mean",
        num_classes=config.num_classes,
    )
