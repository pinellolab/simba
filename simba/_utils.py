"""Utility functions and classes"""

import numpy as np
from kneed import KneeLocator
import tables
from anndata import AnnData


def locate_elbow(x, y, S=10, min_elbow=0,
                 curve='convex', direction='decreasing', online=False,
                 **kwargs):
    """Detect knee points

    Parameters
    ----------
    x : `array-like`
        x values
    y : `array-like`
        y values
    S : `float`, optional (default: 10)
        Sensitivity
    min_elbow: `int`, optional (default: 0)
        The minimum elbow location
    curve: `str`, optional (default: 'convex')
        Choose from {'convex','concave'}
        If 'concave', algorithm will detect knees,
        If 'convex', algorithm will detect elbows.
    direction: `str`, optional (default: 'decreasing')
        Choose from {'decreasing','increasing'}
    online: `bool`, optional (default: False)
        kneed will correct old knee points if True,
        kneed will return first knee if False.
    **kwargs: `dict`, optional
        Extra arguments to KneeLocator.

    Returns
    -------
    elbow: `int`
        elbow point
    """
    kneedle = KneeLocator(x[int(min_elbow):], y[int(min_elbow):],
                          S=S, curve=curve,
                          direction=direction,
                          online=online,
                          **kwargs,
                          )
    if kneedle.elbow is None:
        elbow = len(y)
    else:
        elbow = int(kneedle.elbow)
    return elbow


# modifed from
# scanpy https://github.com/theislab/scanpy/blob/master/scanpy/readwrite.py
def _read_legacy_10x_h5(filename, genome=None):
    """
    Read hdf5 file from Cell Ranger v2 or earlier versions.
    """
    with tables.open_file(str(filename), 'r') as f:
        try:
            children = [x._v_name for x in f.list_nodes(f.root)]
            if not genome:
                if len(children) > 1:
                    raise ValueError(
                        f"'{filename}' contains more than one genome. "
                        "For legacy 10x h5 "
                        "files you must specify the genome "
                        "if more than one is present. "
                        f"Available genomes are: {children}"
                    )
                genome = children[0]
            elif genome not in children:
                raise ValueError(
                    f"Could not find genome '{genome}' in '{filename}'. "
                    f'Available genomes are: {children}'
                )
            dsets = {}
            for node in f.walk_nodes('/' + genome, 'Array'):
                dsets[node.name] = node.read()
            # AnnData works with csr matrices
            # 10x stores the transposed data, so we do the transposition
            from scipy.sparse import csr_matrix

            M, N = dsets['shape']
            data = dsets['data']
            if dsets['data'].dtype == np.dtype('int32'):
                data = dsets['data'].view('float32')
                data[:] = dsets['data']
            matrix = csr_matrix(
                (data, dsets['indices'], dsets['indptr']),
                shape=(N, M),
            )
            # the csc matrix is automatically the transposed csr matrix
            # as scanpy expects it, so, no need for a further transpostion
            adata = AnnData(
                matrix,
                obs=dict(obs_names=dsets['barcodes'].astype(str)),
                var=dict(
                    var_names=dsets['gene_names'].astype(str),
                    gene_ids=dsets['genes'].astype(str),
                ),
            )
            return adata
        except KeyError:
            raise Exception('File is missing one or more required datasets.')


# modifed from
# scanpy https://github.com/theislab/scanpy/blob/master/scanpy/readwrite.py
def _read_v3_10x_h5(filename):
    """
    Read hdf5 file from Cell Ranger v3 or later versions.
    """
    with tables.open_file(str(filename), 'r') as f:
        try:
            dsets = {}
            for node in f.walk_nodes('/matrix', 'Array'):
                dsets[node.name] = node.read()
            from scipy.sparse import csr_matrix

            M, N = dsets['shape']
            data = dsets['data']
            if dsets['data'].dtype == np.dtype('int32'):
                data = dsets['data'].view('float32')
                data[:] = dsets['data']
            matrix = csr_matrix(
                (data, dsets['indices'], dsets['indptr']),
                shape=(N, M),
            )
            adata = AnnData(
                matrix,
                obs=dict(obs_names=dsets['barcodes'].astype(str)),
                var=dict(
                    var_names=dsets['name'].astype(str),
                    gene_ids=dsets['id'].astype(str),
                    feature_types=dsets['feature_type'].astype(str),
                    genome=dsets['genome'].astype(str),
                ),
            )
            return adata
        except KeyError:
            raise Exception('File is missing one or more required datasets.')
