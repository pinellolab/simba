"""Functions and classes for the analysis after PBG training"""

import pandas as pd
import numpy as np


def discretize(adata,
               layer=None,
               n_bins=3,
               encode='ordinal',
               strategy='kmeans',
               dtype=None):
    """Discretize continous features
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.

    Returns
    -------
    updates `adata` with the following fields.
    X: `numpy.ndarray` (`adata.X`)
        Store #observations × #var_genes logarithmized data matrix.
    """

