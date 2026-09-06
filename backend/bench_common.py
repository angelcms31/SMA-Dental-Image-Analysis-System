"""
bench_common.py
----------------
Small helpers shared by the v2 benchmark scripts (compare_three_way.py,
ablate_esma_v2.py) so that image listing, the development/hold-out split
and per-image preparation are defined in exactly ONE place.

The split is deterministic: the sorted file list is shuffled with
random.Random(seed) (same idiom as compare_full_images.py), the first
`dev_n` files form the DEVELOPMENT set (used for ablations / choosing
v2 defaults) and the rest form the HOLD-OUT set (used for reported
results). Nothing tuned on the dev set is ever selected using hold-out
numbers.
"""

import glob
import os
import random

import cv2

from sma_algorithms import autocrop_black_borders, compute_histogram_prob

IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg")


def list_images(images_dir):
    paths = []
    for ext in IMAGE_EXTS:
        paths.extend(glob.glob(os.path.join(images_dir, ext)))
    return sorted(paths)


def dev_holdout_split(paths, seed=42, dev_n=40):
    """Deterministic (dev, holdout) split of a path list."""
    paths = list(paths)
    random.Random(seed).shuffle(paths)
    return paths[:dev_n], paths[dev_n:]


def select_subset(paths, subset, seed=42, dev_n=40, max_images=None):
    """subset: 'dev', 'holdout' or 'all' (shuffled by seed like compare_full_images.py)."""
    dev, hold = dev_holdout_split(paths, seed=seed, dev_n=dev_n)
    if subset == "dev":
        chosen = dev
    elif subset == "holdout":
        chosen = hold
    elif subset == "all":
        chosen = dev + hold
    else:
        raise ValueError("subset must be 'dev', 'holdout' or 'all'")
    if max_images:
        chosen = chosen[:max_images]
    return chosen


def load_gray_autocropped(path):
    """Grayscale read + border autocrop, exactly as compare_full_images.py does."""
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return autocrop_black_borders(image)


def prepare_image(path):
    """Returns (image, prob) or (None, None) if unreadable."""
    image = load_gray_autocropped(path)
    if image is None:
        return None, None
    return image, compute_histogram_prob(image)
