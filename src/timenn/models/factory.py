"""
Unified Model Factory

This module provides a unified interface for building various models.
It encapsulates all model builders and provides a simple factory function.

Usage:
    from timenn.models.factory import build_model, get_available_models

    model = build_model("lmsnet", cfg.model, c_in=4, input_length=2000, num_classes=10)

    models = get_available_models()
"""

from typing import Any

import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from timenn.models.lms_net import LMSNet, FilterConfig
from timenn.models.resnet1d import ResNet, BasicBlock
from timenn.models.simple_cnn import SimpleCNNClassifier, SimpleCNNConfig
from timenn.models.transformer import TimeSeriesTransformer
from timenn.models.lms_tproj import LMSTProjTransformer
from timenn.models.lms_tenc1 import LMSTEnc1Transformer
from timenn.models.lms_cnn import LMSCNN, LMSCNNConfig
from timenn.models.lms_resnet import LMSResNet, LMSResNetConfig


_MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "lmsnet": {
        "builder": "build_lmsnet",
    },
    "resnet1d": {
        "builder": "build_resnet1d",
    },
    "simple_cnn": {
        "builder": "build_simple_cnn",
    },
    "transformer": {
        "builder": "build_transformer",
    },
    "lms_tproj": {
        "builder": "build_lms_tproj",
    },
    "lms_tenc1": {
        "builder": "build_lms_tenc1",
    },
    "lms_cnn": {
        "builder": "build_lms_cnn",
    },
    "lms_resnet": {
        "builder": "build_lms_resnet",
    },
}


def _parse_stage_configs(
    config: DictConfig,
) -> list[tuple[FilterConfig, int, int, int, int, float]]:
    """Parse stage_configs from config into FilterConfig tuples."""
    stage_configs = []
    for stage_cfg in config.stage_configs:
        filter_cfg = stage_cfg.filter_config
        filter_config = FilterConfig(
            filter_type=filter_cfg.filter_type,
            scale_type=filter_cfg.scale_type,
            num_scales=filter_cfg.num_scales,
            min_sigma=filter_cfg.min_sigma,
            f_min=filter_cfg.f_min,
            f_max=filter_cfg.f_max,
            kernel_length=filter_cfg.kernel_length,
            use_joint_mixing=filter_cfg.use_joint_mixing,
            freq_kernel_size=filter_cfg.freq_kernel_size,
            time_kernel_size=filter_cfg.time_kernel_size,
            reduce_scales=filter_cfg.reduce_scales,
            pool_type=filter_cfg.pool_type,
            mixing_groups=filter_cfg.mixing_groups,
        )
        stage_configs.append(
            (
                filter_config,
                stage_cfg.intra_out_per_group,
                stage_cfg.inter_out,
                stage_cfg.blur_stride,
                stage_cfg.blur_kernel_size,
                stage_cfg.blur_min_sigma,
            )
        )
    return stage_configs


def build_lmsnet(
    config: DictConfig, c_in: int, input_length: int, num_classes: int
) -> LMSNet:
    """Build LMSNet model from config."""
    stage_configs = _parse_stage_configs(config)

    return LMSNet(
        c_in=c_in,
        input_length=input_length,
        num_classes=num_classes,
        stage_configs=stage_configs,
        use_log=config.use_log,
        mixing_activation=config.mixing_activation,
        mlp_hidden=config.mlp_hidden,
        dropout=config.dropout,
        use_std=config.use_std,
        align_time=config.align_time,
        blur_use_padding=config.blur_use_padding,
    )


def build_resnet1d(
    config: DictConfig, c_in: int, input_length: int, num_classes: int
) -> ResNet:
    """Build ResNet1D model from config."""
    layers = list(config.layers)

    return ResNet(
        block=BasicBlock,
        layers=layers,
        in_channels=c_in,
        num_classes=num_classes,
        base_filters=config.base_filters,
        dropout=config.dropout,
        for_lmsnet=config.for_lmsnet,
        groups=config.groups,
        width_per_group=config.width_per_group,
        zero_init_residual=config.zero_init_residual,
        mlp_hidden=config.get("mlp_hidden", None),
    )


def build_simple_cnn(
    config: DictConfig, c_in: int, input_length: int, num_classes: int
) -> SimpleCNNClassifier:
    """Build Simple CNN model from config."""
    cnn_layers = tuple(config.cnn_layers)

    model_config = SimpleCNNConfig(
        cnn_layers=cnn_layers,
        cnn_kernel_size=config.cnn_kernel_size,
        cnn_stride=config.cnn_stride,
        cnn_dropout=config.cnn_dropout,
        mlp_hidden=config.mlp_hidden,
        dropout=config.dropout,
    )

    return SimpleCNNClassifier(
        c_in=c_in,
        input_length=input_length,
        num_classes=num_classes,
        config=model_config,
    )


def build_transformer(
    config: DictConfig, c_in: int, input_length: int, num_classes: int
) -> TimeSeriesTransformer:
    """Build Transformer model from config."""
    return TimeSeriesTransformer(
        in_channels=c_in,
        num_classes=num_classes,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        max_len=config.max_len,
        patch_size=config.patch_size,
        stride=config.stride,
    )


def build_lms_cnn(
    config: DictConfig, c_in: int, input_length: int, num_classes: int
) -> LMSCNN:
    """Build LMS-CNN model from config."""
    stage_configs = []
    for stage_cfg in config.stage_configs:
        stage_dict = {
            "filter_type": stage_cfg.filter_type,
            "scale_type": stage_cfg.scale_type,
            "num_scales": stage_cfg.num_scales,
            "min_sigma": stage_cfg.min_sigma,
            "f_min": stage_cfg.f_min,
            "f_max": stage_cfg.f_max,
            "kernel_length": stage_cfg.kernel_length,
            "intra_out_per_group": stage_cfg.intra_out_per_group,
            "inter_out": stage_cfg.inter_out,
            "blur_stride": stage_cfg.blur_stride,
            "blur_kernel_size": stage_cfg.blur_kernel_size,
            "blur_min_sigma": stage_cfg.blur_min_sigma,
            "use_log": stage_cfg.use_log,
            "mixing_activation": stage_cfg.mixing_activation,
            "use_joint_mixing": stage_cfg.use_joint_mixing,
            "freq_kernel_size": stage_cfg.freq_kernel_size,
            "time_kernel_size": stage_cfg.time_kernel_size,
            "reduce_scales": stage_cfg.reduce_scales,
            "pool_type": stage_cfg.pool_type,
            "mixing_groups": stage_cfg.mixing_groups,
        }
        stage_configs.append(stage_dict)

    model_config = LMSCNNConfig(
        stage_configs=stage_configs,
        cnn_layers=tuple(config.cnn_layers),
        cnn_kernel_size=config.cnn_kernel_size,
        cnn_stride=config.cnn_stride,
        cnn_dropout=config.cnn_dropout,
        mlp_hidden=config.mlp_hidden,
        dropout=config.dropout,
        norm_type=OmegaConf.select(config, "norm_type", default="batch"),
    )

    return LMSCNN(
        c_in=c_in,
        input_length=input_length,
        num_classes=num_classes,
        config=model_config,
        use_duration=config.use_duration,
    )


def build_lms_tproj(
    config: DictConfig, c_in: int, input_length: int, num_classes: int
) -> LMSTProjTransformer:
    """Build LMS-Transformer with LMSBlock input projection."""
    del input_length
    transform_config = {
        "d_model": config.transform.d_model,
        "nhead": config.transform.nhead,
        "num_layers": config.transform.num_layers,
        "dim_feedforward": config.transform.dim_feedforward,
        "dropout": config.transform.dropout,
        "max_len": config.transform.max_len,
    }
    lmsblock_config = {
        "filter_type": config.lmsblock.filter_type,
        "scale_type": config.lmsblock.scale_type,
        "num_scales": config.lmsblock.num_scales,
        "min_sigma": config.lmsblock.min_sigma,
        "f_min": config.lmsblock.f_min,
        "f_max": config.lmsblock.f_max,
        "kernel_length": config.lmsblock.kernel_length,
        "use_joint_mixing": config.lmsblock.use_joint_mixing,
        "freq_kernel_size": config.lmsblock.freq_kernel_size,
        "time_kernel_size": config.lmsblock.time_kernel_size,
        "reduce_scales": config.lmsblock.reduce_scales,
        "pool_type": config.lmsblock.pool_type,
        "mixing_groups": config.lmsblock.mixing_groups,
        "intra_out_per_group": config.lmsblock.intra_out_per_group,
        "inter_out": config.lmsblock.inter_out,
        "blur_stride": config.lmsblock.blur_stride,
        "blur_kernel_size": config.lmsblock.blur_kernel_size,
        "blur_min_sigma": config.lmsblock.blur_min_sigma,
        "use_log": config.lmsblock.use_log,
        "mixing_activation": config.lmsblock.mixing_activation,
    }
    return LMSTProjTransformer(
        in_channels=c_in,
        num_classes=num_classes,
        transform_config=transform_config,
        lmsblock_config=lmsblock_config,
        mlp_hidden=config.mlp_hidden,
        head_dropout=config.dropout,
    )


def build_lms_tenc1(
    config: DictConfig, c_in: int, input_length: int, num_classes: int
) -> LMSTEnc1Transformer:
    """Build LMS-Transformer with LMSBlock replacing the first encoder."""
    del input_length
    transform_config = {
        "d_model": config.transform.d_model,
        "nhead": config.transform.nhead,
        "num_layers": config.transform.num_layers,
        "dim_feedforward": config.transform.dim_feedforward,
        "dropout": config.transform.dropout,
        "max_len": config.transform.max_len,
    }
    lmsblock_config = {
        "filter_type": config.lmsblock.filter_type,
        "scale_type": config.lmsblock.scale_type,
        "num_scales": config.lmsblock.num_scales,
        "min_sigma": config.lmsblock.min_sigma,
        "f_min": config.lmsblock.f_min,
        "f_max": config.lmsblock.f_max,
        "kernel_length": config.lmsblock.kernel_length,
        "use_joint_mixing": config.lmsblock.use_joint_mixing,
        "freq_kernel_size": config.lmsblock.freq_kernel_size,
        "time_kernel_size": config.lmsblock.time_kernel_size,
        "reduce_scales": config.lmsblock.reduce_scales,
        "pool_type": config.lmsblock.pool_type,
        "mixing_groups": config.lmsblock.mixing_groups,
        "intra_out_per_group": config.lmsblock.intra_out_per_group,
        "inter_out": config.lmsblock.inter_out,
        "blur_stride": config.lmsblock.blur_stride,
        "blur_kernel_size": config.lmsblock.blur_kernel_size,
        "blur_min_sigma": config.lmsblock.blur_min_sigma,
        "use_log": config.lmsblock.use_log,
        "mixing_activation": config.lmsblock.mixing_activation,
    }
    return LMSTEnc1Transformer(
        in_channels=c_in,
        num_classes=num_classes,
        patch_size=config.patch_size,
        stride=config.stride,
        transform_config=transform_config,
        lmsblock_config=lmsblock_config,
        mlp_hidden=config.mlp_hidden,
        head_dropout=config.dropout,
    )


def build_lms_resnet(
    config: DictConfig, c_in: int, input_length: int, num_classes: int
) -> LMSResNet:
    """Build LMS-ResNet model from config."""
    stage_configs = []
    for stage_cfg in config.stage_configs:
        stage_dict = {
            "filter_type": stage_cfg.filter_type,
            "scale_type": stage_cfg.scale_type,
            "num_scales": stage_cfg.num_scales,
            "min_sigma": stage_cfg.min_sigma,
            "f_min": stage_cfg.f_min,
            "f_max": stage_cfg.f_max,
            "kernel_length": stage_cfg.kernel_length,
            "intra_out_per_group": stage_cfg.intra_out_per_group,
            "inter_out": stage_cfg.inter_out,
            "blur_stride": stage_cfg.blur_stride,
            "blur_kernel_size": stage_cfg.blur_kernel_size,
            "blur_min_sigma": stage_cfg.blur_min_sigma,
            "use_log": stage_cfg.use_log,
            "mixing_activation": stage_cfg.mixing_activation,
            "use_joint_mixing": stage_cfg.use_joint_mixing,
            "freq_kernel_size": stage_cfg.freq_kernel_size,
            "time_kernel_size": stage_cfg.time_kernel_size,
            "reduce_scales": stage_cfg.reduce_scales,
            "pool_type": stage_cfg.pool_type,
            "mixing_groups": stage_cfg.mixing_groups,
        }
        stage_configs.append(stage_dict)

    model_config = LMSResNetConfig(
        stage_configs=stage_configs,
        resnet_layers=tuple(config.resnet_layers),
        resnet_base_filters=config.resnet_base_filters,
        resnet_dropout=config.resnet_dropout,
        mlp_hidden=config.mlp_hidden,
        dropout=config.dropout,
    )

    return LMSResNet(
        c_in=c_in,
        input_length=input_length,
        num_classes=num_classes,
        config=model_config,
    )


def build_model(
    model_name: str,
    config: DictConfig,
    c_in: int,
    input_length: int,
    num_classes: int,
) -> nn.Module:
    """
    Build a model based on the model name and configuration.

        Args:
            model_name: Name of the model to build.
            Available options: "lmsnet", "resnet1d", "simple_cnn",
            "transformer", "lms_tproj", "lms_tenc1", "lms_cnn", "lms_resnet"
        config: Configuration DictConfig object (parsed from Hydra YAML).
            Must contain all required keys for the specified model.
        c_in: Number of input channels.
        input_length: Input sequence length.
        num_classes: Number of output classes.

    Returns:
        Configured model instance

    Raises:
        ValueError: If model_name is not recognized
        AttributeError: If required config keys are missing
    """
    if model_name not in _MODEL_REGISTRY:
        available = list(_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model: '{model_name}'. Available models: {available}"
        )

    builder_name = _MODEL_REGISTRY[model_name]["builder"]

    if builder_name == "build_lmsnet":
        return build_lmsnet(config, c_in, input_length, num_classes)
    elif builder_name == "build_resnet1d":
        return build_resnet1d(config, c_in, input_length, num_classes)
    elif builder_name == "build_simple_cnn":
        return build_simple_cnn(config, c_in, input_length, num_classes)
    elif builder_name == "build_transformer":
        return build_transformer(config, c_in, input_length, num_classes)
    elif builder_name == "build_lms_tproj":
        return build_lms_tproj(config, c_in, input_length, num_classes)
    elif builder_name == "build_lms_tenc1":
        return build_lms_tenc1(config, c_in, input_length, num_classes)
    elif builder_name == "build_lms_cnn":
        return build_lms_cnn(config, c_in, input_length, num_classes)
    elif builder_name == "build_lms_resnet":
        return build_lms_resnet(config, c_in, input_length, num_classes)

    raise ValueError(f"Model builder not implemented for: {model_name}")


def get_available_models() -> list[str]:
    """Return list of available model names."""
    return list(_MODEL_REGISTRY.keys())


def get_model_info(model_name: str) -> dict[str, Any]:
    """Get detailed information about a model."""
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: '{model_name}'. Available: {get_available_models()}"
        )

    entry = _MODEL_REGISTRY[model_name]
    return {
        "name": model_name,
        "builder": entry["builder"],
    }
