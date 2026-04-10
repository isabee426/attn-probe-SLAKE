"""Dataset loaders for medical VQA — reuses faithscan_vqarad data loading."""

from faithscan_vqarad.data.vqarad_dataset import (
    Example, get_image, iter_examples_by_split, load_vqarad,
)
from faithscan_vqarad.data.multi_dataset import (
    load_multi_dataset, load_slake, load_pathvqa,
)
from faithscan_vqarad.data.vindrcxr_dataset import load_vindrcxr

__all__ = [
    "Example", "get_image", "iter_examples_by_split",
    "load_vqarad", "load_multi_dataset", "load_slake", "load_pathvqa",
    "load_vindrcxr",
]
