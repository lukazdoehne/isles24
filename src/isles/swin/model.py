"""
Model code for multi-encoder Swin-UNETR
"""

import copy
from pathlib import Path
from collections.abc import Sequence
from itertools import product

import torch
import torch.nn as nn

from monai.networks.nets.swin_unetr import SwinUNETR, filter_swinunetr
from monai.networks.blocks import UnetrBasicBlock
from monai.networks.utils import copy_model_state
from monai.inferers import SlidingWindowInferer

from isles.swin.config import SwinTrainConfig
from isles.swin.checkpoint import Checkpoint


def get_model(config: SwinTrainConfig) -> nn.Module:
    """Dispact the correct module, already instantiated."""

    model_dict = {
        "BaseSwinUNETR": BaseSwinUNETR,
        "MultiEncoderSwinUNETR": MultiEncoderSwinUNETR,
    }
    model_class = model_dict[config.model]

    return model_class.from_config(config)


class BaseSwinUNETR(SwinUNETR):
    """Base Swin-UNETR extended with convenience methods for compatibility with
    MultiEncoderSwinUNETR

    Parameters
    ----------
    **kwargs
        Arguments passed to parent SwinUNETR.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def load_pretrained_encoders(self, weights_path: str) -> None:
        """Load SSL pretrained weights into all encoders."""
        ssl_weights = torch.load(weights_path, weights_only=False)["state_dict"]

        _, loaded, not_loaded = copy_model_state(
            self, ssl_weights, filter_func=filter_swinunetr
        )
        print(f"Encoder: loaded {len(loaded)} keys, skipped {len(not_loaded)} keys")

    @classmethod
    def from_config(cls, config: SwinTrainConfig) -> "BaseSwinUNETR":
        """Create model from config."""
        return cls(
            in_channels=len(config.modalities),
            out_channels=config.num_classes,
            feature_size=config.feature_size,
        )


class MultiEncoderSwinUNETR(SwinUNETR):
    """Swin-UNETR with multi encoders.

    Based on <https://arxiv.org/abs/2201.01266>", inherits all the decoder architecture
    from monai.networks.nets.SwinUNETR and replaces the single encoder with per-modality
    encoder and channel fusion.

    The model is for the moment hardcoded to produce binary segmentation restuls.

    Parameters
    ----------
    modalities : list[str]
        List of modality names (e.g., ["CTA", "CBF"]).
    feature_size : int
        Base feature dimension.
    fusion_kernel_size : int
        Kernel size for fusion convolutions.
    **kwargs
        Additional arguments passed to parent SwinUNETR.
    """

    def __init__(
        self,
        modalities: list[str],
        num_classes: int = 2,
        feature_size: int = 48,
        fusion_kernel_size: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            in_channels=1,
            feature_size=feature_size,
            out_channels=num_classes,
            **kwargs,
        )

        self.modalities = modalities
        num_modalities = len(modalities)

        self.swin_encoders = nn.ModuleDict(
            {modality: copy.deepcopy(self.swinViT) for modality in modalities}
        )
        del self.swinViT

        self.fusion_layers = nn.ModuleList(
            [
                nn.Conv3d(
                    in_channels=feature_size * mult * num_modalities,
                    out_channels=feature_size * mult,
                    kernel_size=fusion_kernel_size,
                    padding=fusion_kernel_size // 2,
                )
                for mult in [1, 2, 4, 8, 16]
            ]
        )

        # Replace encoder1 to handle multi-channel input
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=kwargs.get("spatial_dims", 3),
            in_channels=num_modalities,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=kwargs.get("norm_name", "instance"),
            res_block=True,
        )

    def load_pretrained_encoders(self, weights_path: str) -> None:
        """Load SSL pretrained weights into all encoders."""
        ssl_weights = torch.load(weights_path, weights_only=False)["state_dict"]

        for modality, encoder in self.swin_encoders.items():
            wrapper = nn.Module()
            wrapper.swinViT = encoder

            _, loaded, _ = copy_model_state(
                wrapper, ssl_weights, filter_func=filter_swinunetr
            )
            print(f"Encoder [{modality}]: loaded {len(loaded)} keys")

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        assert x_in.shape[1] == len(self.modalities)

        # Run each modality through its encoder
        all_hidden_states = [
            self.swin_encoders[modality](x_in[:, i : i + 1, ...], self.normalize)
            for i, modality in enumerate(self.modalities)
        ]

        # Fuse at each scale
        fused_hidden_states = [
            self.fusion_layers[s](torch.cat([hs[s] for hs in all_hidden_states], dim=1))
            for s in range(5)
        ]

        # Decoder (reusing parent's blocks)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(fused_hidden_states[0])
        enc2 = self.encoder3(fused_hidden_states[1])
        enc3 = self.encoder4(fused_hidden_states[2])
        dec4 = self.encoder10(fused_hidden_states[4])

        dec3 = self.decoder5(dec4, fused_hidden_states[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)

        logits = self.out(out)
        return logits

    @classmethod
    def from_config(cls, config: SwinTrainConfig) -> "MultiEncoderSwinUNETR":
        """Create model from config."""
        return cls(
            modalities=config.modalities,
            num_classes=config.num_classes,
            feature_size=config.feature_size,
            fusion_kernel_size=config.fusion_kernel_size,
        )


class SwinUNETRPredictor:
    """Wrapper for Swin-UNETR inference with sliding window.

    Handles device management, sliding window inference, and post-processing
    for both single and multi-encoder Swin-UNETR models.

    Parameters
    ----------
    model : MultiEncoderSwinUNETR | BaseSwinUNETR
        Swin-UNETR model for inference
    roi_size : Sequence[int]
        Size of sliding window ROI
    overlap : float, default=0.2
        Overlap ratio between windows (0-1)
    sw_batch_size : int, default=2
        Batch size for sliding window inference
    sw_blend_mode : str, default="gaussian"
        Blending mode for overlapping predictions
    tta_flips : bool, default=True
        Wheter to perform test time augmentation with volume flips
    amp : bool, default=True
        Whether to use automatic mixed precision
    """

    def __init__(
        self,
        model: MultiEncoderSwinUNETR | BaseSwinUNETR,
        roi_size: Sequence[int],
        overlap: float = 0.2,
        sw_batch_size: int = 2,
        sw_blend_mode: str = "gaussian",
        tta_flips: bool = True,
        amp: bool = True,
    ):
        self.model = model
        self.amp = amp
        self.tta_flips = tta_flips
        self.inferer = SlidingWindowInferer(
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            mode=sw_blend_mode,
        )

    @classmethod
    def from_config(
        cls,
        model: MultiEncoderSwinUNETR | BaseSwinUNETR,
        config: SwinTrainConfig,
        final: bool = False,
        **config_overrides,
    ) -> "SwinUNETRPredictor":
        """Create predictor from training config.

        Parameters
        ----------
        model : MultiEncoderSwinUNETR | BaseSwinUNETR
            Initialized model
        config : SwinTrainConfig
            Training configuration
        final : bool, default=False
            Whether to use final evaluation settings
        **config_overrides
            Keyword arguments to override in config

        Returns
        -------
        SwinUNETRPredictor
            Predictor instance
        """

        # Override config attributes
        for key, value in config_overrides.items():
            setattr(config, key, value)

        overlap = config.val_overlap_final if final else config.val_overlap

        return cls(
            model=model,
            roi_size=config.roi_size,
            overlap=overlap,
            sw_batch_size=config.inferer_batch_size,
            sw_blend_mode=config.inferer_blend_mode,
            tta_flips=config.tta_flips,
            amp=config.amp,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        device: str | torch.device = "cpu",
        final: bool = False,
        **config_overrides,
    ) -> "SwinUNETRPredictor":
        """Create predictor from checkpoint.

        Parameters
        ----------
        checkpoint_path : Path | str
            Path to checkpoint file
        device : str | torch.device, default="cpu"
            Device to load model on
        final : bool, default=False
            Whether to use final evaluation settings
        **config_overrides
            Keyword arguments to override in config

        Returns
        -------
        SwinUNETRPredictor
            Predictor instance
        """
        checkpoint = Checkpoint.load(checkpoint_path)
        config = SwinTrainConfig(**checkpoint.config)

        model = get_model(config)
        model.load_state_dict(checkpoint.model_state_dict)
        model = model.to(device)
        return cls.from_config(model, config, final, **config_overrides)

    @property
    def device(self) -> torch.device:
        """Get model's current device.

        Returns
        -------
        torch.device
            Device where model parameters reside
        """
        return next(self.model.parameters()).device

    def to(self, device: torch.device | str) -> "SwinUNETRPredictor":
        """Move model to device.

        Parameters
        ----------
        device : torch.device | str
            Target device

        Returns
        -------
        SwinUNETRPredictor
            Self for method chaining
        """
        self.model = self.model.to(device)
        return self

    @torch.no_grad()
    def predict_logits(self, image: torch.Tensor) -> torch.Tensor:
        """Predict raw logits using sliding window inference.

        When ``tta_flips`` is enabled, the input is mirrored along all
        combinations of spatial axes (8 passes for 3D), each result is
        flipped back, and the logits are averaged.

        Parameters
        ----------
        image : torch.Tensor
            Input image tensor of shape ``(B, C, H, W, D)``.

        Returns
        -------
        torch.Tensor
            Raw logits on the original input device.
        """
        self.model.eval()
        input_device = image.device
        image = image.to(self.device)

        if self.tta_flips:
            logits = self._predict_logits_tta(image)
        else:
            logits = self._predict_logits_single(image)

        return logits.to(input_device)

    @torch.no_grad()
    def predict_probs(self, image: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities.

        Parameters
        ----------
        image : torch.Tensor
            Input image tensor

        Returns
        -------
        torch.Tensor
            Class probabilities (softmax of logits)
        """
        logits = self.predict_logits(image)
        return torch.softmax(logits, dim=1)

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """Predict discrete class labels.

        Parameters
        ----------
        image : torch.Tensor
            Input image tensor

        Returns
        -------
        torch.Tensor
            Predicted class labels (argmax of logits)
        """
        logits = self.predict_logits(image)
        return logits.argmax(dim=1, keepdim=True)

    def _predict_logits_single(self, image: torch.Tensor) -> torch.Tensor:
        """Single-pass sliding window inference.

        Parameters
        ----------
        image : torch.Tensor
            Input on ``self.device``.

        Returns
        -------
        torch.Tensor
            Raw logits.
        """
        with torch.amp.autocast(
            device_type=self.device.type, dtype=torch.bfloat16, enabled=self.amp
        ):
            return self.inferer(image, self.model)

    @torch.no_grad()
    def _predict_logits_tta(self, image: torch.Tensor) -> torch.Tensor:
        """Mirroring TTA: average logits over all 2^3 flip combinations.

        Parameters
        ----------
        image : torch.Tensor
            Input on ``self.device``, shape ``(B, C, H, W, D)``.

        Returns
        -------
        torch.Tensor
            Averaged logits over 8 mirrored views.
        """

        spatial_dims = [2, 3, 4]
        logits_sum: torch.Tensor | None = None

        for flip_flags in product([False, True], repeat=len(spatial_dims)):
            axes = [d for d, flip in zip(spatial_dims, flip_flags) if flip]

            aug = torch.flip(image, dims=axes) if axes else image
            pred = self._predict_logits_single(aug)
            if axes:
                pred = torch.flip(pred, dims=axes)

            if logits_sum is None:
                logits_sum = pred
            else:
                logits_sum = logits_sum + pred

        return logits_sum / (2 ** len(spatial_dims))
