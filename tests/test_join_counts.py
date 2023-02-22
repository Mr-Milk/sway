import pytest

from sway import Weights, join_counts
from spatialtis_core import points_neighbors
import numpy as np

points = np.random.randn(10, 2)
labels = [i for i in range(len(points))]
neighbors = points_neighbors(points, labels, k=4)
exp_arr = np.random.randn(3, 10) > 0

source = dict(zip(labels, neighbors))
w = Weights(neighbors, labels)


def test_join_count():
    r = join_counts(exp_arr, w)
    print(r)
