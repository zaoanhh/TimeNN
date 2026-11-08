"""
LMS-ResNet: Hierarchical Scattering with ResNet1D

Architecture: Input -> [LMSBlock x N] -> Concat(s_out) -> ResNet1D -> Pool -> MLP -> Output

Based on Andén & Mallat (2014) Deep Scattering Spectrum Theory.
"""

from dataclasses import dataclass
from typing import Literal, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from timenn.models.lms_net import LMSBlock, FilterConfig
from timenn.models.resnet1d import ResNet, BasicBlock


@dataclass
class LMSResNetConfig:
    """
    Configuration for LMS-ResNet model with multi-stage support.

    Supports two modes:
    1. Single block: use individual block parameters
    2. Multi-block: Use stage_configs for stacked blocks
    """

    # === Single-block configuration ===
    filter_type: Literal["morlet", "chirplet", "morse"] = "morlet"
    scale_type: Literal["mel", "linear"] = "linear"
    num_scales: int = 4
    min_sigma: float = 0.05
    f_min: float = 0.01
    f_max: float = 0.4
    kernel_length: int = 64
    intra_out_per_group: int = 8
    inter_out: int = 128
    blur_stride: int = 2
    blur_kernel_size: int = 5
    blur_min_sigma: float = 0.8
    use_log: bool = True
    mixing_activation: Literal["gelu", "relu", "none"] = "gelu"

    # === Multi-stage mode ===
    # stage_configs overrides individual block params when provided
    stage_configs: Optional[List[dict]] = None

    # === ResNet1D parameters ===
    resnet_layers: Tuple[int, int, int, int] = (2, 2, 2, 2)
    resnet_base_filters: int = 64
    resnet_dropout: float = 0.0

    # === Head parameters ===
    mlp_hidden: int = 256
    dropout: float = 0.1

    # === Loss coefficient ===
    coverage_lambda: float = 0.1


class LMSResNet(nn.Module):
    """
    LMS-ResNet: Hierarchical Scattering with ResNet1D.

    Multi-stage architecture:
        Input -> [LMSBlock x N] -> Concat(s_out) -> ResNet1D -> Pool -> MLP

    Each LMSBlock produces:
        - u_next: High-frequency stream for next block
        - s_out: Invariant features (collected and concatenated)

    Final features = Concat(s_out_1, s_out_2, ..., s_out_N) in channel dimension.
    ResNet then processes these multi-scale invariant features.
    """

    def __init__(
        self,
        c_in: int,
        input_length: int,
        num_classes: int,
        config: Optional[LMSResNetConfig] = None,
    ):
        super().__init__()
        self.c_in = c_in
        self.input_length = input_length
        self.num_classes = num_classes

        if config is None:
            config = LMSResNetConfig()

        self.config = config
        self.use_multi_stage = config.stage_configs is not None

        if self.use_multi_stage:
            # === Multi-stage mode ===
            self._build_multi_stage(c_in, config)
        else:
            # === Single-block configuration ===
            self._build_single_stage(c_in, config)

    def _build_single_stage(self, c_in: int, config: LMSResNetConfig):
        """Build a single LMSBlock."""
        filter_config = FilterConfig(
            filter_type=config.filter_type,
            scale_type=config.scale_type,
            num_scales=config.num_scales,
            min_sigma=config.min_sigma,
            f_min=config.f_min,
            f_max=config.f_max,
            kernel_length=config.kernel_length,
        )

        self.lms_blocks = nn.ModuleList(
            [
                LMSBlock(
                    c_in=c_in,
                    filter_config=filter_config,
                    intra_out_per_group=config.intra_out_per_group,
                    inter_out=config.inter_out,
                    blur_stride=config.blur_stride,
                    blur_kernel_size=config.blur_kernel_size,
                    blur_min_sigma=config.blur_min_sigma,
                    use_log=config.use_log,
                    mixing_activation=config.mixing_activation,
                )
            ]
        )

        self.inter_outs = [config.inter_out]
        self.blur_strides = [config.blur_stride]

        # ResNet input channels = inter_out of single block
        self._build_resnet(config.inter_out, config)

    def _build_multi_stage(self, c_in: int, config: LMSResNetConfig):
        """Build multiple stacked LMSBlocks with s_out concatenation."""
        self.lms_blocks = nn.ModuleList()
        self.inter_outs = []
        self.blur_strides = []

        current_channels = c_in

        for stage_cfg in config.stage_configs:
            filter_config = FilterConfig(
                filter_type=stage_cfg.get("filter_type", config.filter_type),
                scale_type=stage_cfg.get("scale_type", config.scale_type),
                num_scales=stage_cfg.get("num_scales", config.num_scales),
                min_sigma=stage_cfg.get("min_sigma", config.min_sigma),
                f_min=stage_cfg.get("f_min", config.f_min),
                f_max=stage_cfg.get("f_max", config.f_max),
                kernel_length=stage_cfg.get("kernel_length", config.kernel_length),
                use_joint_mixing=stage_cfg.get("use_joint_mixing", False),
                freq_kernel_size=stage_cfg.get("freq_kernel_size", 3),
                time_kernel_size=stage_cfg.get("time_kernel_size", 15),
                reduce_scales=stage_cfg.get("reduce_scales", True),
                pool_type=stage_cfg.get("pool_type", "learnable"),
                mixing_groups=stage_cfg.get("mixing_groups", None),
            )

            block = LMSBlock(
                c_in=current_channels,
                filter_config=filter_config,
                intra_out_per_group=stage_cfg.get(
                    "intra_out_per_group", config.intra_out_per_group
                ),
                inter_out=stage_cfg.get("inter_out", config.inter_out),
                blur_stride=stage_cfg.get("blur_stride", config.blur_stride),
                blur_kernel_size=stage_cfg.get(
                    "blur_kernel_size", config.blur_kernel_size
                ),
                blur_min_sigma=stage_cfg.get("blur_min_sigma", config.blur_min_sigma),
                use_log=stage_cfg.get("use_log", config.use_log),
                mixing_activation=stage_cfg.get(
                    "mixing_activation", config.mixing_activation
                ),
            )

            self.lms_blocks.append(block)
            current_channels = stage_cfg.get("inter_out", config.inter_out)
            self.inter_outs.append(current_channels)
            self.blur_strides.append(stage_cfg.get("blur_stride", config.blur_stride))

        # ResNet input channels = sum of all inter_outs (concatenated s_out)
        total_channels = sum(self.inter_outs)
        self._build_resnet(total_channels, config)

    def _build_resnet(self, in_channels: int, config: LMSResNetConfig):
        """Build ResNet1D backbone."""
        self.resnet = ResNet(
            block=BasicBlock,
            layers=config.resnet_layers,
            in_channels=in_channels,
            num_classes=self.num_classes,
            base_filters=config.resnet_base_filters,
            dropout=config.resnet_dropout,
            for_lmsnet=False,
        )
        self.norm = nn.BatchNorm1d(in_channels)

        final_filters = config.resnet_base_filters * (
            2 ** (len(config.resnet_layers) - 1)
        )
        self.resnet_fc_features = final_filters * BasicBlock.expansion

        self.resnet.fc = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(self.resnet_fc_features, config.mlp_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden, self.num_classes),
        )

    def get_total_coverage_loss(self) -> torch.Tensor:
        """Aggregate coverage loss from all LMSBlock filter banks."""
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for block in self.lms_blocks:
            total_loss = total_loss + block.filter_bank.get_coverage_loss()
        return total_loss

    def _align_features(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Align s_out features to same time length via interpolation."""
        if len(features) <= 1:
            return features

        lengths = [f.shape[-1] for f in features]
        target_length = max(lengths)

        aligned = []
        for feat, length in zip(features, lengths):
            if length == target_length:
                aligned.append(feat)
            else:
                aligned.append(
                    F.interpolate(
                        feat, size=target_length, mode="linear", align_corners=True
                    )
                )
        return aligned

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: LMSBlock stack -> Concat(s_out) -> ResNet -> MLP.

        Returns: Logits [B, num_classes]
        """
        # === Collect s_out from all LMSBlocks ===
        s_out_list = []
        u_current = x

        for block in self.lms_blocks:
            u_current, s_out = block(u_current)
            s_out_list.append(s_out)

        # === Concatenate s_out in channel dimension ===
        s_out_aligned = self._align_features(s_out_list)
        s_out_cat = torch.cat(s_out_aligned, dim=1)  # [B, sum(inter_outs), L_max]

        # === ResNet processing ===
        out = self.norm(s_out_cat)
        out = self.resnet.extract_features(out)

        return self.head(out)


def build_lms_resnet(
    c_in: int,
    input_length: int,
    num_classes: int,
    # Single block params
    filter_type: Literal["morlet", "chirplet", "morse"] = "morlet",
    scale_type: Literal["mel", "linear"] = "linear",
    num_scales: int = 4,
    # ResNet params
    resnet_layers: Tuple[int, int, int, int] = (2, 2, 2, 2),
    resnet_base_filters: int = 64,
    # Head params
    mlp_hidden: int = 256,
    dropout: float = 0.1,
    **kwargs,
) -> LMSResNet:
    """
    Convenience function to build LMS-ResNet with single block config.

    Args:
        c_in: Number of input channels
        input_length: Input sequence length
        num_classes: Number of output classes
        filter_type: Type of physical filter
        scale_type: Scale spacing
        num_scales: Number of scales per filter bank
        resnet_layers: ResNet layer configuration
        resnet_base_filters: Base filters for ResNet
        mlp_hidden: Hidden dimension of MLP head
        dropout: Dropout rate
        **kwargs: Additional arguments passed to LMSResNetConfig

    Returns:
        Configured LMSResNet instance
    """
    config = LMSResNetConfig(
        filter_type=filter_type,
        scale_type=scale_type,
        num_scales=num_scales,
        resnet_layers=resnet_layers,
        resnet_base_filters=resnet_base_filters,
        mlp_hidden=mlp_hidden,
        dropout=dropout,
        **kwargs,
    )

    return LMSResNet(
        c_in=c_in,
        input_length=input_length,
        num_classes=num_classes,
        config=config,
    )
