from collections import OrderedDict
import pytest

from sway import Weights
from spatialtis_core import points_neighbors
import numpy as np
from libpysal.weights import W

points = np.random.randn(10, 2)
labels = [i for i in range(len(points))]
neighbors = points_neighbors(points, labels, k=4)
print(neighbors)

source = OrderedDict(zip(labels, neighbors))
ref_w = W(source)
w = Weights(neighbors, labels)


@pytest.mark.parametrize("transform", ["B", "R"])
def test_neighbors_dict(transform):
    w.transform = transform
    ref_w.transform = transform
    test_neighbors = w.neighbors
    for k, v in ref_w.neighbors.items():
        assert set(test_neighbors[k]) == set(v)
    w.id2i
    w.id_order


@pytest.mark.parametrize("transform", ["B", "R"])
def test_weights_computed_attr(transform):
    w.transform = transform
    ref_w.transform = transform
    assert w.s0 == ref_w.s0
    assert w.s1 == ref_w.s1
    assert w.s2 == ref_w.s2


@pytest.mark.parametrize("transform", ["B", "R"])
def test_weights_sparse(transform):
    w.transform = transform
    ref_w.transform = transform
    w_sparse = w.sparse
    ref_sparse = ref_w.sparse
    assert (w_sparse.indptr == ref_sparse.indptr).all()
    assert (w_sparse.indices == ref_sparse.indices).all()
    assert (w_sparse.data == ref_sparse.data).all()
