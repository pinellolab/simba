"""Utility functions and classes"""

import numpy as np
from sklearn.neighbors import KDTree
from scipy.sparse import csr_matrix
from statsmodels.stats.multitest import fdrcorrection
from typing import Sequence
from scipy import sparse
from scipy.sparse import dok_matrix, hstack

def _uniquify(seq, sep='-'):
    """Uniquify a list of strings.

    Adding unique numbers to duplicate values.

    Parameters
    ----------
    seq : `list` or `array-like`
        A list of values
    sep : `str`
        Separator

    Returns
    -------
    seq: `list` or `array-like`
        A list of updated values
    """

    dups = {}

    for i, val in enumerate(seq):
        if val not in dups:
            # Store index of first occurrence and occurrence value
            dups[val] = [i, 1]
        else:
            # Increment occurrence value, index value doesn't matter anymore
            dups[val][1] += 1

            # Use stored occurrence value
            seq[i] += (sep+str(dups[val][1]))

    return seq

def _randomize_matrix(input_adj_graph, n_virtual_dest_nodes: int = None, method='random'):
    """
    Randomly shuffle edges of input adjacency matrix.
    
    Shuffle edges to produce matrix of new shape `(input_adj_graph.shape[0], n_virtual_dest_nodes)`.

    Parameters
    ----------
    input_adj_graph: `scipy.sparse` 
        Sparse input matrix to be shuffled.
    n_virtual_dest_nodes: `int`
        Number of destination nodes to be produced by shuffling.
        If None, return the matrix with the same shape as `input_adj_graph`.
    method: `str`, one of ['random', 'degPreserving', 'binnedDegPreserving', 'none'].
        'random' randomly samples node degree and edge weight from destination nodes of input graph for each node.
        'degPreserving' randomly shuffles source and destination node indices only within the nodes with same degrees.
        'binnedDegPreserving' quantizes the degree of source and destination node degrees to shuffle node indices within the quantized bins.
        'none' doesn't shuffle the graph.
    """
    if n_virtual_dest_nodes is None:
        n_virtual_dest_nodes = input_adj_graph.shape[1]
    if method == "random":
        return _randomShuffle(n_virtual_dest_nodes, input_adj_graph)
    elif method == "degPreserving":
        return _degreePreservingShuffle(input_adj_graph, n_virtual_dest_nodes)
    elif method == "binnedDegPreserving":
        return _digitizedDegreePreservingShuffle(input_adj_graph, n_virtual_dest_nodes=n_virtual_dest_nodes)
    elif method == 'none':
        return dok_matrix((input_adj_graph.shape[0], n_virtual_dest_nodes))
    else:
        raise ValueError('{} not implemented. method should be one of "random", "degPreserving", "binnedDegPreserving".')


def _randomShuffle(n_virtual_dest_nodes: int, original_adj_matrix: np.ndarray):
    # original_adj_matrix: n_source (cells) x  n_dest
    # reutnrs: n_source (cells) x n_virtual_dest
    virtual_adj_matrix = dok_matrix((original_adj_matrix.shape[0], n_virtual_dest_nodes))
    
    for vdidx in range(n_virtual_dest_nodes):
        didx_sampled = np.random.randint(original_adj_matrix.shape[1]) # select the destination node index to sample edge weight from
        edge_weights = original_adj_matrix[:,didx_sampled]
        pidx = np.random.permutation(range(edge_weights.shape[0]))
        virtual_edge_weight = edge_weights[pidx,0]
        try:
            virtual_adj_matrix[:,vdidx] = virtual_edge_weight
        except: return virtual_edge_weight
            
    return(virtual_adj_matrix)

def _digitizedDegreePreservingShuffle(input_adj_graph, mean_nodes = 100, n_virtual_dest_nodes: int = None):
    print(f"input graph: {input_adj_graph.shape}")
    n_orig_dest_nodes = input_adj_graph.shape[1]
    if n_virtual_dest_nodes is None:
        n_virtual_dest_nodes = n_orig_dest_nodes
    input_col_idx = np.tile(np.arange(n_orig_dest_nodes), n_virtual_dest_nodes // n_orig_dest_nodes)
    assert input_col_idx.ndim == 1
    input_col_idx = np.concatenate((input_col_idx, np.random.choice(n_orig_dest_nodes, n_virtual_dest_nodes % n_orig_dest_nodes)))
    
    assert input_col_idx.ndim == 1
    shuffled_adj_graph = input_adj_graph.copy()[:,input_col_idx].tocoo()
    row_idx = shuffled_adj_graph.row
    col_idx = shuffled_adj_graph.col
    data_vals = shuffled_adj_graph.data
    assert shuffled_adj_graph.shape == (input_adj_graph.shape[0], n_virtual_dest_nodes)
    source_degs = (input_adj_graph>0).toarray().sum(axis=1)
    dest_degs = np.squeeze(np.asarray((input_adj_graph>0).sum(axis=0)))[input_col_idx]

    row_conv_dict = dict()
    source_quantile = np.unique(np.quantile(source_degs.data, np.append(np.arange(0, 1, mean_nodes/n_orig_dest_nodes), [1.])))
    source_lb=0.0
    for source_q in source_quantile:
        source_deg_nodes_idx = np.where((source_lb <= source_degs) & (source_degs < source_q))[0]
        row_conv_dict.update(dict(zip(source_deg_nodes_idx, np.random.permutation(source_deg_nodes_idx))))
    shuffled_row_idx = np.vectorize(row_conv_dict.get)(row_idx)

    col_conv_dict = dict()
    dest_quantile = np.unique(np.quantile(dest_degs.data, np.append(np.arange(0, 1, mean_nodes/n_virtual_dest_nodes), [1.])))
    print(dest_quantile)
    dest_lb = 0.0
    for dest_q in dest_quantile:
        dest_deg_nodes_idx = np.where((dest_lb <= dest_degs) & (dest_degs < dest_q))[0]
        col_conv_dict.update(dict(zip(dest_deg_nodes_idx, np.random.permutation(dest_deg_nodes_idx))))
    shuffled_col_idx = np.vectorize(col_conv_dict.get)(col_idx)

    shuffled_adj_graph = sparse.coo_matrix((data_vals, (shuffled_row_idx, shuffled_col_idx)), shape=(input_adj_graph.shape[0], n_virtual_dest_nodes)).tocsr()
    return shuffled_adj_graph

def _degreePreservingShuffle(input_adj_graph, n_virtual_dest_nodes: int = None):
    print(f"input graph: {input_adj_graph.shape}")
    n_orig_dest_nodes = input_adj_graph.shape[1]
    if n_virtual_dest_nodes is None:
        n_virtual_dest_nodes = n_orig_dest_nodes
    input_col_idx = np.tile(np.arange(n_orig_dest_nodes), n_virtual_dest_nodes // n_orig_dest_nodes)
    assert input_col_idx.ndim == 1
    input_col_idx = np.concatenate((input_col_idx, np.random.choice(n_orig_dest_nodes, n_virtual_dest_nodes % n_orig_dest_nodes)))
    
    assert input_col_idx.ndim == 1
    if not isinstance(input_adj_graph, sparse.csr_matrix): 
        input_adj_graph = sparse.csr_matrix(input_adj_graph)
    shuffled_adj_graph = input_adj_graph.copy()[:,input_col_idx].tocoo()
    row_idx = shuffled_adj_graph.row
    col_idx = shuffled_adj_graph.col
    data_vals = shuffled_adj_graph.data
    assert shuffled_adj_graph.shape == (input_adj_graph.shape[0], n_virtual_dest_nodes)
    source_degs = (input_adj_graph>0).toarray().sum(axis=1)
    dest_degs = np.squeeze(np.asarray((input_adj_graph>0).sum(axis=0)))[input_col_idx]

    row_conv_dict = dict()
    for source_deg in np.unique(source_degs):
        source_deg_nodes_idx = np.where(source_degs == source_deg)[0]
        row_conv_dict.update(dict(zip(source_deg_nodes_idx, np.random.permutation(source_deg_nodes_idx))))
    shuffled_row_idx = np.vectorize(row_conv_dict.get)(row_idx)

    col_conv_dict = dict()
    for dest_deg in np.unique(dest_degs):
        dest_deg_nodes_idx = np.where(dest_degs == dest_deg)[0]
        col_conv_dict.update(dict(zip(dest_deg_nodes_idx, np.random.permutation(dest_deg_nodes_idx))))
    shuffled_col_idx = np.vectorize(col_conv_dict.get)(col_idx)

    shuffled_adj_graph = sparse.coo_matrix((data_vals, (shuffled_row_idx, shuffled_col_idx)), shape=(input_adj_graph.shape[0], n_virtual_dest_nodes)).tocsr()
    return shuffled_adj_graph


def _gini(array):
    """Calculate the Gini coefficient of a numpy array.
    """

    array = array.flatten().astype(float)
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1, array.shape[0]+1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))


def _knn(X_ref,
         X_query=None,
         k=20,
         leaf_size=40,
         metric='euclidean'):
    """Calculate K nearest neigbors for each row.
    """
    if X_query is None:
        X_query = X_ref.copy()
    kdt = KDTree(X_ref, leaf_size=leaf_size, metric=metric)
    kdt_d, kdt_i = kdt.query(X_query, k=k, return_distance=True)
    # kdt_i = kdt_i[:, 1:]  # exclude the point itself
    # kdt_d = kdt_d[:, 1:]  # exclude the point itself
    sp_row = np.repeat(np.arange(kdt_i.shape[0]), kdt_i.shape[1])
    sp_col = kdt_i.flatten()
    sp_conn = np.repeat(1, len(sp_row))
    sp_dist = kdt_d.flatten()
    mat_conn_ref_query = csr_matrix(
        (sp_conn, (sp_row, sp_col)),
        shape=(X_query.shape[0], X_ref.shape[0])).T
    mat_dist_ref_query = csr_matrix(
        (sp_dist, (sp_row, sp_col)),
        shape=(X_query.shape[0], X_ref.shape[0])).T
    return mat_conn_ref_query, mat_dist_ref_query

def _get_quantile(value, vector):
    return((value >= vector).sum()/len(vector))

def _fdr(p_vals, method: str = 'bh', lam: float = 0.4):
    """Calculates FDR from p-value.
    
    Args:
        method: Benjamini-Hochberg for 'bh', 'Stanley-Tibshirani' for 'st'.
            Stanley-Tibshirani has higher power, but estimation of lambda parameter
            is not implemented.
        lam: lambda parameter used for Stanley-Tibshirani method.
    """
    if method == 'bh':
        return fdrcorrection(p_vals)[-1]
    elif method == 'st':
        m = len(p_vals)
        pi_0 = (p_vals > lam).sum()/(len(p_vals)*(1-lam))
        return(p_vals.map(lambda t: (pi_0*m*t)/(p_vals<t).sum()))
    else:
        raise ValueError(f"Incorrect FDR correction method '{method}'.")

def _p_vals(samples, null_values):
    empirical_quantile = samples.map(lambda x: _get_quantile(x, null_values))
    p_vals = 1- empirical_quantile
    return p_vals

def _get_fdr(samples: Sequence[float], null_values: Sequence[float], method: str = 'bh'):
    """Calculates FDR from sample and null distribution of estimands.
    
    Obtains FDR for one-sided p-values for alternative hypothesis of
    sample statistic being larger than null empirical distribution.
    
    Args:
        samples: sample statistics
        null_values: null distribution of statistics
        method: Benjamini-Hochberg for 'bh', 'Stanley-Tibshirani' for 'st'.
            Stanley-Tibshirani has higher power, but estimation of lambda parameter
            is not implemented.
        lam: lambda parameter used for Stanley-Tibshirani method.
    """
    p_vals = _p_vals(samples, null_values)
    return(p_vals, _fdr(p_vals, method=method))
