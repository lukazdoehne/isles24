"""
Code for image processing and transforms
"""

from pathlib import Path
from typing import Literal
from collections.abc import Sequence, Mapping
import numpy as np
from numpy.typing import DTypeLike
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    CropForegroundd,
    RandFlipd,
    RandRotate90d,
    EnsureTyped,
    MapTransform,
    ScaleIntensityRange,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandCropByLabelClassesd,
    CastToTyped,
    SpatialPadd,
    DeleteItemsd,
    Activationsd,
    AsDiscreted,
    Invertd,
    SaveImaged,
    MaskIntensityd,
    RandSpatialCropSamplesd,
    MultiSampleTrait,
    Randomizable,
)
from monai.utils import convert_to_dst_type
from monai.data import DataLoader

from isles.swin.config import SwinTrainConfig


def get_train_transforms(config: SwinTrainConfig):
    """
    Build training transforms.

    Parameters
    ----------
    config : SwinTrainConfig
        Configuration dataclass for training multi-encoder Swin-UNETR.

    Returns
    -------
    Compose
        MONAI composed transforms.
    """

    transforms = [
        LoadImaged(keys=["image", "label", "brain_mask"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label", "brain_mask"]),
        Orientationd(keys=["image", "label", "brain_mask"], axcodes="RAS", labels=None),
        MaskIntensityd(keys=["image"], mask_key="brain_mask"),
        CropForegroundd(
            keys=["image", "label"],
            source_key="brain_mask",
            select_fn=lambda x: x > 0,
            margin=0,
        ),
        DeleteItemsd(keys=["brain_mask"]),
        PerChannelScaleIntensityd(
            keys=["image"],
            modalities=config.modalities,
            windows=config.intensity_windows,
            b_min=0.0,
            b_max=1.0,
            clip=True,
            sanitize_modalities=config.sanitize_modalities,
        ),
    ]

    if config.target_spacing is not None:
        transforms.append(
            Spacingd(
                keys=["image", "label"],
                pixdim=config.target_spacing,
                mode=("bilinear", "nearest"),
            )
        )

    transforms.extend(
        [
            CastToTyped(keys=["image", "label"], dtype=[torch.float32, torch.uint8]),
            EnsureTyped(keys=["image", "label"], track_meta=True),
            SpatialPadd(keys=["image", "label"], spatial_size=config.roi_size),
            RandCropByModed(
                keys=["image", "label"],
                label_key="label",
                spatial_size=config.roi_size,
                num_samples=config.num_crops_per_image,
                mode=config.crop_mode,
                num_classes=config.num_classes,
                ratios=config.crop_ratios,
            ),
            RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.1),
            RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.1),
        ]
    )

    return Compose(transforms)


def get_val_transforms(config: SwinTrainConfig):
    """
    Build validation transforms (no augmentation).

    Parameters
    ----------
    config : SwinTrainConfig
        Configuration dataclass for training multi-encoder Swin-UNETR.

    Returns
    -------
    Compose
        MONAI composed transforms.
    """

    transforms = [
        LoadImaged(keys=["image", "label", "brain_mask"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label", "brain_mask"]),
        Orientationd(keys=["image", "label", "brain_mask"], axcodes="RAS", labels=None),
        MaskIntensityd(keys=["image"], mask_key="brain_mask"),
        CropForegroundd(
            keys=["image", "label"],
            source_key="brain_mask",
            select_fn=lambda x: x > 0,
            margin=0,
        ),
        DeleteItemsd(keys=["brain_mask"]),
        PerChannelScaleIntensityd(
            keys=["image"],
            modalities=config.modalities,
            windows=config.intensity_windows,
            b_min=0.0,
            b_max=1.0,
            clip=True,
            sanitize_modalities=config.sanitize_modalities,
        ),
    ]

    if config.target_spacing is not None:
        transforms.append(
            Spacingd(
                keys=["image", "label"],
                pixdim=config.target_spacing,
                mode=("bilinear", "nearest"),
            )
        )

    transforms.extend(
        [
            CastToTyped(keys=["image", "label"], dtype=[torch.float32, torch.uint8]),
            EnsureTyped(keys=["image", "label"], track_meta=True),
        ]
    )

    return Compose(transforms)


class PerChannelScaleIntensityd(MapTransform):
    """
    Rescale images and apply different intensity windowing per channel.

    Parameters
    ----------
    keys : str
        Key for the multichannel image.
    modalities : Sequence[str]
        Order of channel modalities in the ["image"] key.
    windows : Mapping[str, Sequence[float]] | None
        Intensity windows for each channel, e.g. {"cta": (a_min, a_max)}. If None,
        no windowing is performed, and the whole data range is used instead.
        Default is None.
    b_min : float
        Output minimum.
    b_max : float
        Output maximum.
    clip : bool
        Whether to clip values outside range.
    dtype : DTypeLike
        Output data type, if None, same as input image. defaults to float32.
    sanitize_modalities : Sequence[str] | None
        Modalities for which NaN and Inf values are replaced with 0.0 before
        scaling. If None, no sanitization is performed. Default is None.
    """

    def __init__(
        self,
        keys: str,
        modalities: Sequence[str],
        windows: Mapping[str, Sequence[float]] | None = None,
        b_min: float = 0.0,
        b_max: float = 1.0,
        clip: bool = True,
        dtype: DTypeLike = np.float32,
        sanitize_modalities: Sequence[str] | None = None,
    ):
        super().__init__(keys)
        self.modalities = modalities
        self.windows = windows
        self.b_min = b_min
        self.b_max = b_max
        self.clip = clip
        self.dtype = dtype
        self.sanitize_modalities = (
            set(sanitize_modalities) if sanitize_modalities else set()
        )

    def __call__(self, data):
        d = dict(data)

        for key in self.keys:
            image: np.ndarray = d[key]

            scaled_channels = []
            for c, modality in enumerate(self.modalities):
                channel = image[c : c + 1]

                # Sanitize NaN/Inf before any scaling
                if modality in self.sanitize_modalities:
                    channel = np.nan_to_num(channel, nan=0.0, posinf=0.0, neginf=0.0)

                if self.windows:
                    window = (
                        self.windows[modality]
                        if self.windows[modality]
                        else (None, None)
                    )
                else:
                    window = (None, None)

                a_min = window[0] if window[0] is not None else channel.min()
                a_max = window[1] if window[1] is not None else channel.max()

                scaling = ScaleIntensityRange(
                    a_min=a_min,
                    a_max=a_max,
                    b_min=self.b_min,
                    b_max=self.b_max,
                    clip=self.clip,
                    dtype=self.dtype,
                )

                scaled_channels.append(scaling(channel))

            ret = torch.cat(scaled_channels, dim=0)
            ret = convert_to_dst_type(ret, image, dtype=self.dtype)[0]
            d[key] = ret

        return d


class RandCropByModed(Randomizable, MapTransform, MultiSampleTrait):
    """Randomly crop patches using either label-guided or fully random spatial sampling.

    Parameters
    ----------
    keys : Sequence[str]
        Keys to crop, typically ``["image", "label"]``.
    label_key : str
        Key for the label map, used only in ``"label_classes"`` mode.
    spatial_size : Sequence[int]
        Spatial size of each cropped patch.
    num_samples : int
        Number of patches to sample per image.
    mode : Literal["label_classes", "spatial"]
        Crop strategy. ``"label_classes"`` uses ``RandCropByLabelClassesd``,
        ``"spatial"`` uses ``RandSpatialCropSamplesd`` with no label guidance.
    num_classes : int
        Number of classes. Used only in ``"label_classes"`` mode.
    ratios : Sequence[float] | None
        Per-class sampling ratios. Used only in ``"label_classes"`` mode.
    """

    def __init__(
        self,
        keys: Sequence[str],
        label_key: str,
        spatial_size: Sequence[int],
        num_samples: int,
        mode: Literal["label_classes", "spatial"],
        num_classes: int = 2,
        ratios: Sequence[float] | None = None,
    ) -> None:
        MapTransform.__init__(self, keys)
        if mode == "label_classes":
            self._transform = RandCropByLabelClassesd(
                keys=keys,
                label_key=label_key,
                spatial_size=spatial_size,
                num_classes=num_classes,
                ratios=ratios,
                num_samples=num_samples,
                warn=False,
            )
        elif mode == "spatial":
            self._transform = RandSpatialCropSamplesd(
                keys=keys,
                roi_size=spatial_size,
                num_samples=num_samples,
                random_size=False,
            )
        else:
            raise ValueError(f"Unknown crop mode: {mode!r}")

    def randomize(self, data: dict | None = None) -> None:
        self._transform.randomize(data)

    def __call__(self, data: dict) -> list[dict]:
        return self._transform(data)


def get_post_transforms(val_loader: DataLoader, out_dir: Path | None = None) -> Compose:
    """
    Build post processing transforms for prediction at original spacing.

    Parameters
    ----------
    val_loader : DataLoader
        Validation dataloader. This is necessary to read the correct transforms to
        invert.
    out_dir : Path | None
        Directory where to save predictions. If None, do not save predictions to disk.
        Default is None

    Returns
    -------
    Compose
        MONAI composed transforms.
    """

    val_transforms = val_loader.dataset.transform

    transforms = [
        Activationsd(keys="pred", softmax=True),
        Invertd(
            keys="pred",
            transform=val_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        AsDiscreted(keys="pred", argmax=True),
    ]

    if out_dir is not None:
        transforms.append(
            SaveImaged(
                keys="pred",
                meta_keys="pred_meta_dict",
                output_dir=out_dir,
                output_postfix="pred",
                resample=False,
                separate_folder=False,
                dtype=np.uint8,
            )
        )

    return Compose(transforms)


def get_logit_post_transforms(
    val_loader: DataLoader,
    out_dir: Path,
) -> Compose:
    """Build post-processing transforms that save raw logits at original spacing.

    Unlike ``get_post_transforms``, this skips ``Activationsd`` and
    ``AsDiscreted`` so that the saved volumes contain raw logit values
    (float32) rather than discrete class labels.

    Parameters
    ----------
    val_loader : DataLoader
        Validation dataloader, used to retrieve the forward transforms for
        inversion.
    out_dir : Path
        Directory where logit volumes will be saved.

    Returns
    -------
    Compose
        MONAI composed transforms.
    """
    val_transforms = val_loader.dataset.transform

    return Compose(
        [
            Invertd(
                keys="pred",
                transform=val_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            SaveImaged(
                keys="pred",
                meta_keys="pred_meta_dict",
                output_dir=out_dir,
                output_postfix="logits",
                resample=False,
                separate_folder=False,
                dtype=np.float32,
            ),
        ]
    )
