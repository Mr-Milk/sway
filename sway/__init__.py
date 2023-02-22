import numpy as np
from scipy.sparse import csr_matrix

from .sway import SpatialWeights
from .sway import _py_join_counts


class Weights(SpatialWeights):

    @property
    def sparse(self) -> csr_matrix:
        """Return the sparse matrix of spatial weights

        ... note::
            The order of row is not guarantee

        Returns
        -------
        :class: scipy.csr_matrix

        """
        data, col_index, indptr = self._to_sparse()
        return csr_matrix((data, col_index, indptr))


def join_counts(
        exp: np.ndarray,
        w: Weights,
        permutations: int = 1000,
        alpha: float = 0.05,
        early_stop: bool = True,
):
    """

    Parameters
    ----------
    exp : `numpy.ndarray`
        Each row is an expression vector, must be binarized before proceed
    w : `Weights`
        A spatial weight instance
    permutations : int
        Number of permutations to perform
    alpha : float
    early_stop : bool

    Returns
    -------

    """
    if isinstance(exp.dtype, (bool, np.bool, np.bool_)):
        raise TypeError("Input array must be bool")
    return _py_join_counts(exp, w, permutations, alpha, early_stop)
