"""Preprocessing utilities for Capture-24 wrist accelerometer data."""

from pathlib import Path

import numpy as np
import pandas as pd

INPUT_CHANNELS = ("x", "y", "z", "VM")
CLASS_NAMES = ("light", "moderate-vigorous", "sedentary", "sleep")
CLASS_TO_ID = {label: idx for idx, label in enumerate(CLASS_NAMES)}


def load_annotation_dictionary(path: str | Path) -> pd.DataFrame:
    """Load the Capture-24 annotation-to-movement-label dictionary."""
    return pd.read_csv(path, index_col="annotation", dtype="string")


def load_participant(
    path: str | Path,
    annotation_dictionary: pd.DataFrame,
    annotation_column: str = "label:Walmsley2020",
) -> pd.DataFrame:
    """Load one participant, map annotations, and calculate vector magnitude."""
    data = pd.read_csv(
        path,
        index_col="time",
        parse_dates=["time"],
        dtype={"x": "f4", "y": "f4", "z": "f4", "annotation": "string"},
    )

    data["label"] = (
        annotation_dictionary[annotation_column]
        .reindex(data["annotation"])
        .to_numpy()
    )

    data["VM"] = np.sqrt(
        data["x"] ** 2 + data["y"] ** 2 + data["z"] ** 2
    )
    return data


def extract_windows(
    data: pd.DataFrame,
    window: str = "10s",
    samples_per_window: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract non-overlapping valid windows and their majority labels.

    With 100-Hz data and a 10-s window, each model input has shape (1000, 4),
    corresponding to x, y, z, and vector magnitude.
    """
    windows = []
    labels = []

    for _, frame in data.resample(window, origin="start"):
        if len(frame) != samples_per_window or frame.isna().any().any():
            continue

        label = frame["label"].mode(dropna=False)
        if len(label) != 1:
            continue

        windows.append(frame[list(INPUT_CHANNELS)].to_numpy(dtype=np.float32))
        labels.append(str(label.iloc[0]))

    return np.asarray(windows, dtype=np.float32), np.asarray(labels)


def encode_labels(labels: np.ndarray) -> np.ndarray:
    """Encode movement-behaviour labels using one consistent class order."""
    unknown = sorted(set(labels) - set(CLASS_TO_ID))
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")
    return np.asarray([CLASS_TO_ID[label] for label in labels], dtype=np.int64)
