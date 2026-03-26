"""
Configuration classes for training and using Multi-encoder Swin-UNETR
"""

import json
from pathlib import Path
from typing import Literal
from collections.abc import Sequence
from dataclasses import dataclass, asdict


@dataclass
class SwinTrainConfig:
    """
    Configuration for multi-encoder Swin-UNETR training.

    Parameters
    ----------
    modalities : Sequence[str]
        Input modality names (e.g., ["ncct", "cta"]).
    target_spacing : Sequence[float]
        Target voxel spacing in mm.
    intensity_windows : dict[str, Sequence[float]] | None
        Per-modality intensity windows: {"modality": [min, max]}.
        If None, no intensity windowing is applied.
    sanitize_modalities : Sequence[str] | None
        Modalities to be sanitized for inf/nan values. These values will be set to 0.
    model : Literal["BaseSwinUNETR", "MultiEncoderSwinUNETR"]
        Model architecture.
    feature_size : int
        Swin-UNETR embedding dimension.
    fusion_kernel_size : int
        Kernel size for multi-encoder fusion convolution.
    num_classes : int
        Number of classes including background (2 for binary segmentation).
    roi_size : Sequence[int]
        Patch size for training and sliding window inference.
    batch_size : int
        Training batch size.
    num_crops_per_image : int
        Number of patches sampled per image.
    crop_mode : Literal["label_classes", "spatial"]
        Crop mode, either label-guided or fully random spatial sampling.
    crop_ratios : Sequence[float] | None
        Per-class sampling ratios for RandCropByLabelClassesd, used when crop_mode
        is ``"label_classes"``. None for equal sampling.
    max_epochs : int
        Total training epochs.
    learning_rate : float
        Initial learning rate for AdamW.
    weight_decay : float
        Weight decay for AdamW.
    warmup_ratio : float
        Fraction of training for learning rate warmup.
    amp : bool
        Enable automatic mixed precision.
    include_background : bool
        Include background class in Dice loss/metric.
    inspect_patches: bool
        If True, run inference on patches from training dataloader and saves them to
        run_dir / "training-inspection".
    inspect_interval: int
        Run training patch inspection every N epochs, if active.
    val_interval : int
        Validate every N epochs.
    val_overlap : float
        Sliding window overlap during training validation.
    val_overlap_final : float
        Sliding window overlap for final evaluation.
    inferer_batch_size : int
        Batch size for sliding window inference.
    inferer_blend_mode : str
        Blend mode for sliding window inference. Can be "constant" or "gaussian".
    inferer_crop_margin : int | None
        Size of the margin to crop from patched during inference. This is only
        used during final evaluation. If not None, overrides `inferer_blend_mode`
        and `val_overlap_final` if it's too small. If None, no cropping is done.
    tta_flips : bool
        Whether to perform test time augmentation by volume flips during
        final inference.
    device : str
        Device for training ("cuda", "cpu", etc.).
    """

    # Required
    modalities: Sequence[str]

    # Data preprocessing
    target_spacing: Sequence[float] = (1.0, 1.0, 1.0)
    intensity_windows: dict[str, Sequence[float]] | None = None
    sanitize_modalities: Sequence[str] | None = None

    # Model architecture
    model: Literal["BaseSwinUNETR", "MultiEncoderSwinUNETR"] = "MultiEncoderSwinUNETR"
    feature_size: int = 48
    fusion_kernel_size: int = 1
    num_classes: int = 2

    # Training / inference patches
    roi_size: Sequence[int] = (64, 64, 64)
    batch_size: int = 1
    num_crops_per_image: int = 4
    crop_mode: Literal["label_classes", "spatial"] = "label_classes"
    crop_ratios: Sequence[float] | None = None

    # Training
    max_epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_ratio: float = 0.05
    amp: bool = True
    include_background: bool = False
    inspect_training: bool = False
    inspect_interval: int = 25

    # Validation
    val_interval: int = 5
    val_overlap: float = 0.2
    val_overlap_final: float = 0.5
    inferer_batch_size: int = 4
    inferer_blend_mode: str = "constant"
    inferer_crop_margin: int | None = None
    tta_flips: bool = False

    # Device
    device: str = "cuda"

    def __post_init__(self) -> None:
        """Ensure that parameter values are not incompatible"""

        # Update parameters when crop_margin is specified
        if self.inferer_crop_margin is not None:
            min_overlap = 2 * self.inferer_crop_margin / min(self.roi_size)
            self.inferer_blend_mode = "constant"
            if self.val_overlap_final < min_overlap:
                raise UserWarning(
                    f"Supplied overlap {self.overlap} too small for correct predictions. "
                    f"Updated to {min_overlap}"
                )
                self.val_overlap_final = min_overlap

    def to_json(self, path: str | Path) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "SwinTrainConfig":
        """Load configuration from JSON file."""
        with open(path) as f:
            return cls(**json.load(f))
