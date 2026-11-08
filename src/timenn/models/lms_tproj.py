"""
Transformer variant with LMSBlock replacing the standard input projection.
"""

import torch
import torch.nn as nn

from timenn.models.lms_net import FilterConfig, LMSBlock
from timenn.models.transformer import PositionalEncoding


class LMSTProjTransformer(nn.Module):
    """
    LMS-Transformer with LMSBlock tokenization.

    Architecture:
        Input -> LMSBlock -> 1x1 projection to d_model -> PositionalEncoding
        -> TransformerEncoder stack -> Pool -> MLP -> Logits
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        transform_config: dict,
        lmsblock_config: dict,
        mlp_hidden: int = 256,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = transform_config["d_model"]

        filter_config = FilterConfig(
            filter_type=lmsblock_config["filter_type"],
            scale_type=lmsblock_config["scale_type"],
            num_scales=lmsblock_config["num_scales"],
            min_sigma=lmsblock_config["min_sigma"],
            f_min=lmsblock_config["f_min"],
            f_max=lmsblock_config["f_max"],
            kernel_length=lmsblock_config["kernel_length"],
            use_joint_mixing=lmsblock_config["use_joint_mixing"],
            freq_kernel_size=lmsblock_config["freq_kernel_size"],
            time_kernel_size=lmsblock_config["time_kernel_size"],
            reduce_scales=lmsblock_config["reduce_scales"],
            pool_type=lmsblock_config["pool_type"],
            mixing_groups=lmsblock_config["mixing_groups"],
        )

        self.input_projection = LMSBlock(
            c_in=in_channels,
            filter_config=filter_config,
            intra_out_per_group=lmsblock_config["intra_out_per_group"],
            inter_out=lmsblock_config["inter_out"],
            blur_stride=lmsblock_config["blur_stride"],
            blur_kernel_size=lmsblock_config["blur_kernel_size"],
            blur_min_sigma=lmsblock_config["blur_min_sigma"],
            use_log=lmsblock_config["use_log"],
            mixing_activation=lmsblock_config["mixing_activation"],
            is_last_block=True,
        )

        if lmsblock_config["inter_out"] == self.d_model:
            self.channel_projection = nn.Identity()
        else:
            self.channel_projection = nn.Conv1d(
                lmsblock_config["inter_out"], self.d_model, kernel_size=1
            )

        self.pos_encoder = PositionalEncoding(
            self.d_model, max_len=transform_config["max_len"]
        )
        self.transformer_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=transform_config["nhead"],
                    dim_feedforward=transform_config["dim_feedforward"],
                    dropout=transform_config["dropout"],
                    batch_first=True,
                )
                for _ in range(transform_config["num_layers"])
            ]
        )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(self.d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(mlp_hidden, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        _, x = self.input_projection(x)
        x = self.channel_projection(x)
        x = self.pos_encoder(x)
        x = x.permute(0, 2, 1)

        for layer in self.transformer_layers:
            x = layer(x)

        return x

    def get_total_coverage_loss(self) -> torch.Tensor:
        return self.input_projection.filter_bank.get_coverage_loss()

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._encode(x)
        x = x.permute(0, 2, 1)
        return self.avgpool(x).squeeze(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        return self.head(features)
