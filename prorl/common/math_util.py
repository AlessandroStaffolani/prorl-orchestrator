from typing import Optional

import numpy as np


def normalize_scalar(value: float, max_val: float, min_val: float = 0, a: float = 0, b: float = 1) -> float:
    return (b - a) * ((value - min_val) / (max_val - min_val)) + a


def truncated_normal_distribution(
        mean: float,
        std: float,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        size: Optional[int] = None,
        random_state: Optional[np.random.RandomState] = None
) -> np.ndarray:
    random = random_state if random_state is not None else np.random
    return np.clip(random.normal(mean, std, size), min_value, max_value)


def l1_distance(x, y):
    return np.abs(x-y)
