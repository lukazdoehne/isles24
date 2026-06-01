"""
Model code for multi-encoder Swin-UNETR
"""
import sys
sys.stderr.write(">>> MODEL.PY IS LOADED FROM: " + __file__ + "\n")
sys.stderr.flush()

import copy
from pathlib import Path
from collections.abc import Sequence
from itertools import product
from functools import partial

import torch
import torch.nn as nn

from monai.networks.nets.swin_unetr import SwinUNETR, filter_swinunetr
from monai.networks.blocks import UnetrBasicBlock
from monai.networks.utils import copy_model_state
from monai.inferers import sliding_window_inference

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
        tabular_embedding_dim: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(
            in_channels=1,
            feature_size=feature_size,
            out_channels=num_classes,
            **kwargs,
        )

        self.modalities = modalities
        self.tabular_embedding_dim = tabular_embedding_dim
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
        if tabular_embedding_dim > 0:
            # Project tabular embedding to bottleneck channel size
            self.tabular_proj = nn.Sequential(
                nn.Linear(tabular_embedding_dim, feature_size * 16),
                #nn.GELU(),
                #nn.Linear(feature_size * 16, feature_size * 16),
            )
            # encoder10 consumes the concatenated [image_features, tabular_features]
            self.encoder10 = UnetrBasicBlock(
                spatial_dims=kwargs.get("spatial_dims", 3),
                in_channels=feature_size * 32, #768*2
                out_channels=feature_size * 16, #768
                kernel_size=3,
                stride=1,
                norm_name=kwargs.get("norm_name", "instance"),
                res_block=True,
            )
            # Keep concatenating tabular context at each decoder scale.
            self.tabular_decoder_channels = [feature_size * 8, feature_size * 4, feature_size * 2, feature_size]
            self.tabular_decoder_proj = nn.ModuleList(
                [nn.Linear(tabular_embedding_dim, ch) for ch in self.tabular_decoder_channels]
            )
            self.tabular_decoder_fuse = nn.ModuleList(
                [nn.Conv3d(in_channels=ch * 2, out_channels=ch, kernel_size=1) for ch in self.tabular_decoder_channels]
            )
        else:
            self.tabular_proj = None
            self.tabular_decoder_channels = []
            self.tabular_decoder_proj = nn.ModuleList()
            self.tabular_decoder_fuse = nn.ModuleList()

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

    def forward(
        self, x_in: torch.Tensor, tabular_embedding: torch.Tensor | None = None
    ) -> torch.Tensor:
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
        #dec4 = self.encoder10(fused_hidden_states[4])
        bottleneck = fused_hidden_states[4]
        tabular_features = None
        if self.tabular_proj is not None and tabular_embedding is not None:
            tabular_features = self.tabular_proj(tabular_embedding.float())
            #Shifting
            #tabular_features = tabular_shift.view(tabular_shift.shape[0], -1, 1, 1, 1)
            #bottleneck = bottleneck + tabular_features
            #Concatenating
            # Sliding-window inference may call forward with window batches that
            # differ from the tabular batch dimension (often B=1 vs B>1).
            if tabular_features.shape[0] != bottleneck.shape[0]:
                if tabular_features.shape[0] == 1:
                    tabular_features = tabular_features.expand(bottleneck.shape[0], -1)
                else:
                    raise RuntimeError(
                        "Tabular and image batch sizes are incompatible for "
                        f"concatenation: tabular={tabular_features.shape[0]}, "
                        f"image={bottleneck.shape[0]}"
                    )

            tabular_features = tabular_features.view(
                bottleneck.shape[0], -1, 1, 1, 1
            ).expand(-1, -1, bottleneck.shape[2], bottleneck.shape[3], bottleneck.shape[4])
            bottleneck = torch.cat([bottleneck, tabular_features], dim=1)

        dec4 = self.encoder10(bottleneck)

        decoder_concat_shapes: dict[str, torch.Size] = {}

        def concat_tabular_at_scale(
            feature_map: torch.Tensor, scale_idx: int, stage_name: str
        ) -> torch.Tensor:
            if tabular_embedding is None or self.tabular_proj is None:
                return feature_map

            tab_scale = self.tabular_decoder_proj[scale_idx](tabular_embedding.float())
            if tab_scale.shape[0] != feature_map.shape[0]:
                if tab_scale.shape[0] == 1:
                    tab_scale = tab_scale.expand(feature_map.shape[0], -1)
                else:
                    raise RuntimeError(
                        "Tabular and image batch sizes are incompatible for "
                        f"concatenation: tabular={tab_scale.shape[0]}, "
                        f"image={feature_map.shape[0]}"
                    )

            tab_scale = tab_scale.view(feature_map.shape[0], -1, 1, 1, 1).expand(
                -1, -1, feature_map.shape[2], feature_map.shape[3], feature_map.shape[4]
            )
            fused = torch.cat([feature_map, tab_scale], dim=1)
            decoder_concat_shapes[stage_name] = fused.shape
            return self.tabular_decoder_fuse[scale_idx](fused)

        dec3 = self.decoder5(dec4, fused_hidden_states[3])
        dec3 = concat_tabular_at_scale(dec3, 0, "dec3")
        dec2 = self.decoder4(dec3, enc3)
        dec2 = concat_tabular_at_scale(dec2, 1, "dec2")
        dec1 = self.decoder3(dec2, enc2)
        dec1 = concat_tabular_at_scale(dec1, 2, "dec1")
        dec0 = self.decoder2(dec1, enc1)
        dec0 = concat_tabular_at_scale(dec0, 3, "dec0")
        out = self.decoder1(dec0, enc0)

        if not hasattr(self, "_shapes_printed"):
            if tabular_features is not None:
                print(f"tabular_features: {tabular_features.shape}")
            print(f"x_in  : {x_in.shape}")
            print(f"enc0  : {enc0.shape}")
            print(f"enc1  : {enc1.shape}")
            print(f"enc2  : {enc2.shape}")
            print(f"enc3  : {enc3.shape}")
            print(f"bottleneck: {bottleneck.shape}")
            print(f"dec4  : {dec4.shape}")
            if decoder_concat_shapes:
                print(f"dec3_cat: {decoder_concat_shapes.get('dec3')}")
                print(f"dec2_cat: {decoder_concat_shapes.get('dec2')}")
                print(f"dec1_cat: {decoder_concat_shapes.get('dec1')}")
                print(f"dec0_cat: {decoder_concat_shapes.get('dec0')}")
            print(f"dec3  : {dec3.shape}")
            print(f"dec2  : {dec2.shape}")
            print(f"dec1  : {dec1.shape}")
            print(f"dec0  : {dec0.shape}")
            print(f"out   : {out.shape}")
            self._shapes_printed = True

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
            tabular_embedding_dim=config.tabular_embedding_dim,
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
    crop_margin : int | None, default=None
        Margin in voxels to exclude from predictions. This calculates
        a custom weight map and overrides `sw_blend_mode` to `'constant'`.
        `overlap` should be larger than `2 * crop_margin / overlap` to
        ensure non-empty interior predictions; if smaller than that,
        it will be overridden. If None, no custom weights are calculated,
        and blending weights are controlled by sw_blend_mode.
    tta_flips : bool, default=False
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
        crop_margin: int | None = None,
        tta_flips: bool = False,
        amp: bool = True,
    ):
        self.model = model
        self.amp = amp
        self.tta_flips = tta_flips

        if crop_margin is not None:
            # ROI size should be passed as tuple to generate_weight_map
            weight_map, overlap = generate_weight_map(
                roi_size=(roi_size, roi_size, roi_size),
                margin=crop_margin,
                overlap=overlap,
            )
            sw_blend_mode = "constant"
        else:
            weight_map = None

        self.inferer = partial(
            sliding_window_inference,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            overlap=overlap,
            mode=sw_blend_mode,
            roi_weight_map=weight_map,
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
            crop_margin=config.inferer_crop_margin,
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
            return self.inferer(inputs=image, predictor=self.model)

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


def generate_weight_map(
    roi_size: tuple[int, ...],
    margin: int,
    overlap: float,
) -> tuple[torch.Tensor, float]:
    """Create a binary weight map zeroing a border margin, with overlap validation.

    Parameters
    ----------
    roi_size : tuple[int, ...]
        Spatial size of the ROI window.
    margin : int
        Number of voxels to zero out from each border per dimension.
    overlap : float
        Sliding window overlap fraction in [0, 1), as passed to the inferer.
        If overlap is smaller than the minimum overlap required to avoid
        empty predictions in the interior of the volume, it will be set to
        this minimum value `2 * margin / roi_size`.

    Returns
    -------
    torch.Tensor
        Shape (roi_size) weight map with 0s in the margin and 1s inside.
    float
        The updated overlap, same as the supplied one if sufficiently large.

    Raises
    ------
    UserWarning
        When overlap is automatically updated to ensure correct prediction.
    """
    min_overlap_per_dim = [2 * margin / s for s in roi_size]
    min_overlap = max(min_overlap_per_dim)
    if overlap < min_overlap:
        raise UserWarning(
            f"Supplied overlap {overlap} too small for correct predictions. "
            f"Updated to {min_overlap}"
        )
        overlap = min_overlap

    weights = torch.ones(roi_size, dtype=torch.float32)
    for dim, size in enumerate(roi_size):
        idx = [slice(None)] * len(roi_size)

        idx[dim] = slice(0, margin)
        weights[tuple(idx)] = 0.0

        idx[dim] = slice(size - margin, size)
        weights[tuple(idx)] = 0.0

    return weights, overlap
