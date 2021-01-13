"""Preprocess"""

import numpy as np
from scipy.sparse import (
    issparse,
    csr_matrix,
)
import re
from sklearn.decomposition import TruncatedSVD

# import sys
# sys.path.insert(0, '/Users/huidong/Projects/Github/simba/simba/preprocessing')
# from _utils import locate_elbow
from ._utils import (
    locate_elbow,
    cal_tf_idf
)


def log_transform(adata):
    """Return the natural logarithm of one plus the input array, element-wise.
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

    adata.X = np.log1p(adata.X)
    return None


def normalize(adata, method='lib_size', scale_factor=1e4, save_raw=True):
    """Normalize count matrix.
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    method: `str`, optional (default: 'lib_size')
        Choose from {{'lib_size','tf_idf'}}
        Method used for dimension reduction.
        'lib_size': Total-count normalize (library-size correct)
        'tf_idf': TF-IDF (term frequency–inverse document frequency)
                  transformation
    Returns
    -------
    updates `adata` with the following fields.
    X: `numpy.ndarray` (`adata.X`)
        Store #observations × #var_genes normalized data matrix.
    """
    if(method not in ['lib_size', 'tf_idf']):
        raise ValueError("unrecognized method '%s'" % method)
    if(save_raw):
        adata.layers['raw'] = adata.X
    if(method == 'lib_size'):
        adata.X = (np.divide(adata.X.T, adata.X.sum(axis=1)).T)*scale_factor
    if(method == 'tf_idf'):
        adata.X = cal_tf_idf(adata.X)


def cal_qc(adata, expr_cutoff=1, assay='rna'):
    """Calculate quality control metrics.

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    expr_cutoff: `float`, optional (default: 1)
        Expression cutoff.
        If greater than expr_cutoff,the feature is considered 'expressed'
    assay: `str`, optional (default: 'rna')
            Choose from {'rna','atac'},case insensitive
    Returns
    -------
    updates `adata` with the following fields.
    n_counts: `pandas.Series` (`adata.var['n_counts']`,dtype `int`)
       The number of read count each gene has.
    n_cells: `pandas.Series` (`adata.var['n_cells']`,dtype `int`)
       The number of cells in which each gene is expressed.
    pct_cells: `pandas.Series` (`adata.var['pct_cells']`,dtype `float`)
       The percentage of cells in which each gene is expressed.
    n_counts: `pandas.Series` (`adata.obs['n_counts']`,dtype `int`)
       The number of read count each cell has.
    n_genes: `pandas.Series` (`adata.obs['n_genes']`,dtype `int`)
       The number of genes expressed in each cell.
    pct_genes: `pandas.Series` (`adata.obs['pct_genes']`,dtype `float`)
       The percentage of genes expressed in each cell.
    n_peaks: `pandas.Series` (`adata.obs['n_peaks']`,dtype `int`)
       The number of peaks expressed in each cell.
    pct_peaks: `pandas.Series` (`adata.obs['pct_peaks']`,dtype `int`)
       The percentage of peaks expressed in each cell.
    pct_mt: `pandas.Series` (`adata.obs['pct_mt']`,dtype `float`)
       the percentage of counts in mitochondrial genes
    """

    assay = assay.lower()
    assert assay in ['rna', 'atac'], \
        "`assay` must be chosen from ['rna','atac']"

    if(not issparse(adata.X)):
        adata.X = csr_matrix(adata.X)

    n_counts = adata.X.sum(axis=0).A1
    adata.var['n_counts'] = n_counts
    n_cells = (adata.X >= expr_cutoff).sum(axis=0).A1
    adata.var['n_cells'] = n_cells
    adata.var['pct_cells'] = n_cells/adata.shape[0]

    n_counts = adata.X.sum(axis=1).A1
    adata.obs['n_counts'] = n_counts
    n_features = (adata.X >= expr_cutoff).sum(axis=1).A1
    if(assay == 'atac'):
        adata.obs['n_peaks'] = n_features
        adata.obs['pct_peaks'] = n_features/adata.shape[1]
    if(assay == 'rna'):
        adata.obs['n_genes'] = n_features
        adata.obs['pct_genes'] = n_features/adata.shape[1]
        r = re.compile("^MT-", flags=re.IGNORECASE)
        mt_genes = list(filter(r.match, adata.var_names))
        if(len(mt_genes) > 0):
            n_counts_mt = adata[:, mt_genes].X.sum(axis=1).A1
            adata.obs['pct_mt'] = n_counts_mt/n_counts
        else:
            adata.obs['pct_mt'] = 0
    adata.uns['assay'] = assay


def filter_cells(adata,
                 min_n_features=None,
                 max_n_features=None,
                 min_pct_features=None,
                 max_pct_features=None,
                 min_n_counts=None,
                 max_n_counts=None,
                 expr_cutoff=1,
                 assay=None):
    """Filter out cells based on different metrics.
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    min_n_features: `int`, optional (default: None)
        Minimum number of features expressed
    min_pct_features: `float`, optional (default: None)
        Minimum percentage of features expressed
    min_n_counts: `int`, optional (default: None)
        Minimum number of read count for one cell
    expr_cutoff: `float`, optional (default: 1)
        Expression cutoff.
        If greater than expr_cutoff,the gene is considered 'expressed'
    assay: `str`, optional (default: 'rna')
            Choose from {{'rna','atac'}},case insensitive
    Returns
    -------
    updates `adata` with a subset of cells that pass the filtering.
    updates `adata` with the following fields if cal_qc() was not performed.
    n_counts: `pandas.Series` (`adata.obs['n_counts']`,dtype `int`)
       The number of read count each cell has.
    n_genes: `pandas.Series` (`adata.obs['n_genes']`,dtype `int`)
       The number of genes expressed in each cell.
    pct_genes: `pandas.Series` (`adata.obs['pct_genes']`,dtype `float`)
       The percentage of genes expressed in each cell.
    n_peaks: `pandas.Series` (`adata.obs['n_peaks']`,dtype `int`)
       The number of peaks expressed in each cell.
    pct_peaks: `pandas.Series` (`adata.obs['pct_peaks']`,dtype `int`)
       The percentage of peaks expressed in each cell.
    """
    if(assay is None):
        if('assay' in adata.uns_keys()):
            assay = adata.uns['assay']
        else:
            raise Exception("Please either run 'st.cal_qc()' "
                            "or specify the parameter`assay`")
    assay = assay.lower()
    assert assay in ['rna', 'atac'], \
        "`assay` must be chosen from ['rna','atac']"

    if('n_counts' in adata.obs_keys()):
        n_counts = adata.obs['n_counts']
    else:
        n_counts = np.sum(adata.X, axis=1).astype(int)
        adata.obs['n_counts'] = n_counts

    feature = ""
    n_features = 0
    pct_features = 0.0
    if(assay == 'rna'):
        if('n_genes' in adata.obs_keys()):
            n_features = adata.obs['n_genes']
        else:
            n_features = np.sum(adata.X >= expr_cutoff, axis=1).astype(int)
            adata.obs['n_genes'] = n_features
        if('pct_genes' in adata.obs_keys()):
            pct_features = adata.obs['pct_genes']
        else:
            pct_features = n_features/adata.shape[1]
            adata.obs['pct_genes'] = pct_features
        feature = 'genes'
    if(assay == 'atac'):
        if('n_peaks' in adata.obs_keys()):
            n_features = adata.obs['n_peaks']
        else:
            n_features = np.sum(adata.X >= expr_cutoff, axis=1).astype(int)
            adata.obs['n_peaks'] = n_features
        if('pct_peaks' in adata.obs_keys()):
            pct_features = adata.obs['pct_peaks']
        else:
            pct_features = n_features/adata.shape[1]
            adata.obs['pct_peaks'] = pct_features
        feature = 'peaks'

    if(sum(list(map(lambda x: x is None,
                    [min_n_features,
                     min_pct_features,
                     min_n_counts,
                     max_n_features,
                     max_pct_features,
                     max_n_counts]))) == 6):
        print('No filtering')
    else:
        cell_subset = np.ones(len(adata.obs_names), dtype=bool)
        if(min_n_features is not None):
            print('filter cells based on min_n_features')
            cell_subset = (n_features >= min_n_features) & cell_subset
        if(max_n_features is not None):
            print('filter cells based on max_n_features')
            cell_subset = (n_features <= max_n_features) & cell_subset
        if(min_pct_features is not None):
            print('filter cells based on min_pct_features')
            cell_subset = (pct_features >= min_pct_features) & cell_subset
        if(max_pct_features is not None):
            print('filter cells based on max_pct_features')
            cell_subset = (pct_features <= max_pct_features) & cell_subset
        if(min_n_counts is not None):
            print('filter cells based on min_n_counts')
            cell_subset = (n_counts >= min_n_counts) & cell_subset
        if(max_n_counts is not None):
            print('filter cells based on max_n_counts')
            cell_subset = (n_counts <= max_n_counts) & cell_subset
        adata._inplace_subset_obs(cell_subset)
        print('after filtering out low-quality cells: ')
        print(str(adata.shape[0])+' cells, ' + str(adata.shape[1])+' '+feature)
    return None


def filter_features(adata,
                    min_n_cells=5,
                    max_n_cells=None,
                    min_pct_cells=None,
                    max_pct_cells=None,
                    min_n_counts=None,
                    max_n_counts=None,
                    expr_cutoff=1,
                    assay=None):
    """Filter out features based on different metrics.
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.
    min_n_cells: `int`, optional (default: 5)
        Minimum number of cells expressing one feature
    min_pct_cells: `float`, optional (default: None)
        Minimum percentage of cells expressing one feature
    min_n_counts: `int`, optional (default: None)
        Minimum number of read count for one feature
    expr_cutoff: `float`, optional (default: 1)
        Expression cutoff.
        If greater than expr_cutoff,the feature is considered 'expressed'
    assay: `str`, optional (default: 'rna')
            Choose from {{'rna','atac'}},case insensitive

    Returns
    -------
    updates `adata` with a subset of features that pass the filtering.
    updates `adata` with the following fields if cal_qc() was not performed.
    n_counts: `pandas.Series` (`adata.var['n_counts']`,dtype `int`)
       The number of read count each gene has.
    n_cells: `pandas.Series` (`adata.var['n_cells']`,dtype `int`)
       The number of cells in which each gene is expressed.
    pct_cells: `pandas.Series` (`adata.var['pct_cells']`,dtype `float`)
       The percentage of cells in which each gene is expressed.
    """
    if(assay is None):
        if('assay' in adata.uns_keys()):
            assay = adata.uns['assay']
        else:
            raise Exception(
                "Please either run 'st.cal_qc()' "
                "or specify the parameter`assay`"
            )
    assay = assay.lower()
    assert assay in ['rna', 'atac'], \
        "`assay` must be chosen from ['rna','atac']"

    feature = ""
    if(assay == 'rna'):
        feature = 'genes'
    if(assay == 'atac'):
        feature = 'peaks'

    if('n_counts' in adata.var_keys()):
        n_counts = adata.var['n_counts']
    else:
        n_counts = np.sum(adata.X, axis=0).astype(int)
        adata.var['n_counts'] = n_counts
    if('n_cells' in adata.var_keys()):
        n_cells = adata.var['n_cells']
    else:
        n_cells = np.sum(adata.X >= expr_cutoff, axis=0).astype(int)
        adata.var['n_cells'] = n_cells
    if('pct_cells' in adata.var_keys()):
        pct_cells = adata.var['pct_cells']
    else:
        pct_cells = n_cells/adata.shape[0]
        adata.var['pct_cells'] = pct_cells

    if(sum(list(map(lambda x: x is None,
                    [min_n_cells, min_pct_cells, min_n_counts,
                     max_n_cells, max_pct_cells, max_n_counts,
                     ]))) == 6):
        print('No filtering')
    else:
        feature_subset = np.ones(len(adata.var_names), dtype=bool)
        if(min_n_cells is not None):
            print('Filter '+feature+' based on min_n_cells')
            feature_subset = (n_cells >= min_n_cells) & feature_subset
        if(max_n_cells is not None):
            print('Filter '+feature+' based on max_n_cells')
            feature_subset = (n_cells <= max_n_cells) & feature_subset
        if(min_pct_cells is not None):
            print('Filter '+feature+' based on min_pct_cells')
            feature_subset = (pct_cells >= min_pct_cells) & feature_subset
        if(max_pct_cells is not None):
            print('Filter '+feature+' based on max_pct_cells')
            feature_subset = (pct_cells <= max_pct_cells) & feature_subset
        if(min_n_counts is not None):
            print('Filter '+feature+' based on min_n_counts')
            feature_subset = (n_counts >= min_n_counts) & feature_subset
        if(max_n_counts is not None):
            print('Filter '+feature+' based on max_n_counts')
            feature_subset = (n_counts <= max_n_counts) & feature_subset
        adata._inplace_subset_var(feature_subset)
        print('After filtering out low-expressed '+feature+': ')
        print(str(adata.shape[0])+' cells, ' + str(adata.shape[1])+' '+feature)
    return None


def pca(adata,
        n_components=50,
        algorithm='randomized',
        n_iter=5,
        random_state=2021,
        tol=0.0
        ):
    """perform Principal Component Analysis (PCA)
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.

    Returns
    -------
    updates `adata` with the following fields:
    `.obsm['X_pca']` : `array`
        PCA transformed X.
    `.varm['PCs']` : `array`
        Principal components in feature space,
        representing the directions of maximum variance in the data.
    `.uns['pca']['variance']` : `array`
        The variance of the training samples transformed by a
        projection to each component.
    `.uns['pca']['variance_ratio']` : `array`
        Percentage of variance explained by each of the selected components.
    """
    svd = TruncatedSVD(n_components=n_components,
                       algorithm=algorithm,
                       n_iter=n_iter,
                       random_state=random_state,
                       tol=0.0)
    svd.fit(adata.X)
    adata.obsm['X_pca'] = svd.transform(adata.X)
    adata.varm['PCs'] = svd.components_.T
    adata.uns['pca'] = dict()
    adata.uns['pca']['variance'] = svd.explained_variance_
    adata.uns['pca']['variance_ratio'] = svd.explained_variance_ratio_


def select_pcs(adata,
               n_pcs=None,
               S=1,
               curve='convex',
               direction='decreasing',
               online=False,
               min_elbow=None,
               **kwargs):
    """select top PCs based on variance_ratio
    """
    if(n_pcs is None):
        n_components = adata.obsm['X_pca'].shape[1]
        if(min_elbow is None):
            min_elbow = n_components/10
        n_pcs = locate_elbow(range(n_components),
                             adata.uns['pca']['variance_ratio'],
                             S=S,
                             min_elbow=min_elbow,
                             direction=direction,
                             online=online,
                             **kwargs)
        adata.uns['pca']['n_pcs'] = n_pcs
    else:
        adata.uns['pca']['n_pcs'] = n_pcs


def select_pcs_features(adata,
                        S=1,
                        curve='convex',
                        direction='decreasing',
                        online=False,
                        min_elbow=None,
                        **kwargs):
    """select features that contribute to the top PCs
    """
    n_pcs = adata.uns['pca']['n_pcs']
    n_features = adata.varm['PCs'].shape[0]
    if(min_elbow is None):
        min_elbow = n_features/10
    adata.uns['pca']['features'] = dict()
    ids_features = list()
    for i in range(n_pcs):
        elbow = locate_elbow(range(n_features),
                             np.sort(np.abs(adata.varm['PCs'][:, i],))[::-1],
                             S=S,
                             min_elbow=min_elbow,
                             **kwargs)
        ids_features_i = \
            list(np.argsort(np.abs(adata.varm['PCs'][:, i],))[::-1][:elbow])
        adata.uns['pca']['features'][f'pc_{i}'] = ids_features_i
        ids_features = ids_features + ids_features_i
    adata.var['top_pcs'] = False
    adata.var.loc[adata.var_names[np.unique(ids_features)], 'top_pcs'] = True
