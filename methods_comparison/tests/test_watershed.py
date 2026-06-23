import numpy as np
from PIL import Image
from pathlib import Path
from methods_comparison.watershed import predict_count, WatershedParams


def _synthetic_image_with_n_blobs(n: int, size: int = 256) -> np.ndarray:
    img = np.zeros((size, size), dtype=np.uint8)
    rng = np.random.default_rng(0)
    coords = rng.integers(20, size - 20, size=(n, 2))
    for y, x in coords:
        img[y - 5:y + 5, x - 5:x + 5] = 255
    return img


def test_watershed_counts_well_separated_blobs(tmp_path: Path):
    img = _synthetic_image_with_n_blobs(15)
    p = tmp_path / "blobs.png"
    Image.fromarray(img).save(p)
    n = predict_count(p, WatershedParams(min_area=20, max_area=500, min_distance=8,
                                         measurement_box=(0, 0, 256, 256)))
    assert 13 <= n <= 17  # leichte Toleranz fuer Watershed-Kanteneffekte
