from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import sobel, threshold_otsu
from skimage.segmentation import watershed

from methods_comparison._constants import MEASUREMENT_BOX

InputKind = Literal["aolp", "aolp_grad", "dolp"]


@dataclass(frozen=True)
class WatershedParams:
    min_area: int
    max_area: int
    min_distance: int
    measurement_box: tuple[int, int, int, int] = field(default_factory=lambda: MEASUREMENT_BOX)
    input_kind: InputKind = "aolp"


def _preprocess(arr: np.ndarray, kind: InputKind) -> np.ndarray:
    if kind == "aolp_grad":
        g = sobel(arr.astype(np.float32))
        g = (g - g.min()) / (g.max() - g.min() + 1e-9)
        return (g * 255).astype(np.uint8)
    # "aolp" and "dolp" use the raw single-channel array as-is
    return arr


def predict_count(image_path: Path, params: WatershedParams) -> int:
    arr = np.asarray(Image.open(image_path).convert("L"))
    arr = _preprocess(arr, params.input_kind)
    thr = threshold_otsu(arr)
    binary = arr > thr
    distance = np.asarray(ndi.distance_transform_edt(binary))
    coords = peak_local_max(distance, min_distance=params.min_distance, labels=binary)
    markers = np.zeros(distance.shape, dtype=int)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    labels = watershed(-distance, markers, mask=binary)
    y0, x0, y1, x1 = params.measurement_box
    box_labels = labels[y0:y1, x0:x1]
    unique = np.unique(box_labels)
    unique = unique[unique != 0]
    count = 0
    for lbl in unique:
        area = int((box_labels == lbl).sum())
        if params.min_area <= area <= params.max_area:
            count += 1
    return count
