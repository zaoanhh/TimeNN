"""
Transformer variant with LMSBlock replacing the first encoder layer.
"""

import torch
import torch.nn as nn

from timenn.models.lms_net import FilterConfig, LMSBlock
from timenn.models.transformer import PositionalEncoding


class LMSFirstEncoderLayer(nn.Module):
    """
    Adapter that makes an LMSBlock behave like a transformer front layer.

    The block may change sequence length through blur pooling. Positional encoding is
    therefore applied after this layer in the parent model.
    """

    def __init__(self, d_model: int, lmsblock_config: dict, dropout: float):
        super().__init__()

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

        self.lmsblock = LMSBlock(
            c_in=d_model,
            filter_config=filter_config,
            intra_out_per_group=lmsblock_config["intra_out_per_group"],
            inter_out=lmsblock_config["inter_out"],
            blur_stride=lmsblock_config["blur_stride"],
            blur_kernel_size=lmsblock_config["blur_kernel_size"],
            blur_min_sigma=lmsblock_config["blur_min_sigma"],
            use_log=lmsblock_config["use_log"],
            mixing_activation=lmsblock_config["mixing_activation"],
            use_residual=True,
            is_last_block=True,
        )

        if lmsblock_config["inter_out"] == d_model:
            self.channel_projection = nn.Identity()
        else:
            self.channel_projection = nn.Conv1d(
                lmsblock_config["inter_out"], d_model, kernel_size=1
            )

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        _, x = self.lmsblock(x)
        x = self.channel_projection(x)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        return self.norm(x)


class LMSTEnc1Transformer(nn.Module):
    """
    Transformer with LMSBlock replacing the first encoder layer.

    Architecture:
        Input -> Conv patch embed -> LMS first encoder -> PositionalEncoding
        -> Remaining TransformerEncoder layers -> Pool -> MLP -> Logits
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        patch_size: int,
        stride: int,
        transform_config: dict,
        lmsblock_config: dict,
        mlp_hidden: int = 256,
        head_dropout: float = 0.1,
    ):
        super().__init__()

        num_layers = transform_config["num_layers"]
        if num_layers < 1:
            raise ValueError("LMSTEnc1Transformer requires transform.num_layers >= 1")

        self.d_model = transform_config["d_model"]

        self.input_projection = nn.Conv1d(
            in_channels, self.d_model, kernel_size=patch_size, stride=stride
        )
        self.first_encoder = LMSFirstEncoderLayer(
            d_model=self.d_model,
            lmsblock_config=lmsblock_config,
            dropout=transform_config["dropout"],
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
                for _ in range(num_layers - 1)
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
        x = self.input_projection(x)
        x = x.permute(0, 2, 1)
        x = self.first_encoder(x)
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)
        x = x.permute(0, 2, 1)

        for layer in self.transformer_layers:
            x = layer(x)

        return x

    def get_total_coverage_loss(self) -> torch.Tensor:
        return self.first_encoder.lmsblock.filter_bank.get_coverage_loss()

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self._encode(x)
        x = x.permute(0, 2, 1)
        return self.avgpool(x).squeeze(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        return self.head(features)
