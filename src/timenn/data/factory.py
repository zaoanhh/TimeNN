"""Dataset factory used by the training and evaluation scripts.

The factory maps a short dataset name (as used in ``configs/``) to the
corresponding builder function and extracts only the configuration keys that
the builder understands.

Adding a dataset
----------------
1. Implement ``build_<name>_dataset`` in a new module under ``timenn.data``.
2. Register the builder in ``_DATASET_REGISTRY`` below.
3. Add ``dataset/<name>.yaml`` under ``configs/lms_cnn_capture24/dataset``.
"""

from __future__ import annotations

from typing import Any, Callable, TypeAlias

from omegaconf import DictConfig
from torch.utils.data import Dataset

from timenn.data.capture24 import create_capture24_datasets
from timenn.data.medley import build_medley_dataset
from timenn.data.speechcommands import build_speechcommands_dataset
from timenn.data.timit import create_timit_datasets
from timenn.data.widar import build_widar_dataset
from timenn.data.xrf55_rfid import build_xrf55_dataset as build_xrf55_rfid_dataset
from timenn.data.xrf55_wifi import build_xrf55_dataset as build_xrf55_wifi_dataset

# ``(train, val, test, num_classes)``
DatasetBundle: TypeAlias = tuple[Dataset, Dataset, Dataset, int]
Builder: TypeAlias = Callable[..., Any]


def _build_widar(cfg: DictConfig) -> DatasetBundle:
    return build_widar_dataset(
        root_dir=cfg.root_dir,
        subcarrier_ids=cfg.subcarrier_ids,
        antenna_ids=cfg.antenna_ids,
        target_sample_rate=cfg.target_sample_rate,
        split_args=cfg.split_args,
        bandpass_freqs=cfg.bandpass_freqs,
        exclude_labels=cfg.exclude_labels,
        rx_device_ids=cfg.rx_device_ids,
    )


def _build_xrf55_wifi(cfg: DictConfig) -> DatasetBundle:
    return build_xrf55_wifi_dataset(
        root_dir=cfg.root_dir,
        split_args=cfg.split_args,
        scenes=cfg.scenes,
        include_types=cfg.include_types,
        subcarrier_ids=cfg.subcarrier_ids,
        target_sample_rate=cfg.target_sample_rate,
        demean=cfg.demean,
    )


def _build_xrf55_rfid(cfg: DictConfig) -> DatasetBundle:
    return build_xrf55_rfid_dataset(
        root_dir=cfg.root_dir,
        split_args=cfg.split_args,
        scenes=cfg.scenes,
        include_types=cfg.include_types,
        exclude_actions=cfg.exclude_actions,
        exclude_repetitions=cfg.exclude_repetitions,
        target_sample_rate=cfg.target_sample_rate,
    )


def _build_capture24(cfg: DictConfig) -> DatasetBundle:
    train_ds, val_ds, test_ds = create_capture24_datasets(
        datadir=cfg.datadir,
        label_method=cfg.label_method,
        val_size=cfg.val_size,
        train_radio=cfg.train_radio,
        seq_length=cfg.seq_length,
        augment_cfg=cfg.augment,
        labels_dict=cfg.labels,
    )
    return train_ds, val_ds, test_ds, train_ds.num_classes


def _build_speechcommands(cfg: DictConfig) -> DatasetBundle:
    return build_speechcommands_dataset(
        root_dir=cfg.root_dir,
        target_sample_rate=cfg.target_sample_rate,
        original_sample_rate=cfg.original_sample_rate,
        target_length=cfg.target_length,
        exclude_labels=cfg.exclude_labels,
        include_labels=cfg.include_labels,
    )


def _build_timit(cfg: DictConfig) -> DatasetBundle:
    train_ds, val_ds, test_ds = create_timit_datasets(
        root_dir=cfg.root_dir,
        window_size=cfg.window_size,
        use_all=cfg.use_all,
        sample_rate=cfg.sample_rate,
        exclude_sa=cfg.exclude_sa,
    )
    return train_ds, val_ds, test_ds, train_ds.get_num_classes()


def _build_medley(cfg: DictConfig) -> DatasetBundle:
    return build_medley_dataset(
        data_dir=cfg.root_dir,
        target_sample_rate=cfg.target_sample_rate,
        original_sample_rate=cfg.original_sample_rate,
        target_length=cfg.target_length,
    )


# Registry: dataset name -> builder.  The builder receives the composed
# ``cfg.dataset`` node and returns ``(train, val, test, num_classes)``.
_DATASET_REGISTRY: dict[str, Builder] = {
    "widar": _build_widar,
    "xrf55_wifi": _build_xrf55_wifi,
    "xrf55_rfid": _build_xrf55_rfid,
    "capture24": _build_capture24,
    "speechcommands": _build_speechcommands,
    "timit": _build_timit,
    "medley": _build_medley,
}


def build_dataset(dataset_name: str, config: DictConfig) -> DatasetBundle:
    """Build ``(train, val, test, num_classes)`` for ``dataset_name``.

    Parameters
    ----------
    dataset_name:
        One of the keys in :data:`_DATASET_REGISTRY`.
    config:
        The composed Hydra ``cfg.dataset`` node.
    """
    try:
        builder = _DATASET_REGISTRY[dataset_name]
    except KeyError as exc:
        available = ", ".join(sorted(_DATASET_REGISTRY))
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. Available: {available}"
        ) from exc
    return builder(config)


def get_available_datasets() -> list[str]:
    """Return the sorted list of registered dataset names."""
    return sorted(_DATASET_REGISTRY)


__all__ = ["DatasetBundle", "build_dataset", "get_available_datasets"]
