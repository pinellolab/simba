"""General-purpose tools"""

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import KBinsDiscretizer

from .._settings import settings


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
    if layer is None:
        X = adata.X.copy()
    else:
        X = adata.layers[layer].copy()
    est = KBinsDiscretizer(n_bins=n_bins,
                           encode=encode,
                           strategy=strategy,
                           dtype=dtype)
    nonzero_cont = X.data.copy()
    nonzero_id = est.fit_transform(nonzero_cont.reshape(-1, 1))
    nonzero_disc = est.inverse_transform(nonzero_id).reshape(-1, )

    adata.layers['disc'] = adata.layers['raw'].copy()
    adata.layers['disc'].data = (nonzero_id+1).reshape(-1,)

    # discretized data transformed back to original feature space
    adata.uns['disc'] = dict()
    adata.uns['disc']['disc_ori'] = adata.layers['raw'].copy()
    adata.uns['disc']['disc_ori'].data = nonzero_disc.reshape(-1,)
    adata.uns['disc']['bin_edges'] = est.bin_edges_


def gen_graph(list_CP=None,
              list_PM=None,
              list_PK=None,
              list_CG=None,
              list_CC=None,
              prefix_C='C',
              prefix_P='P',
              prefix_M='M',
              prefix_K='K',
              prefix_G='G',
              copy=False,
              filename='pbg_graph.txt'
              ):
    """Generate graph for PBG training based on indices of obs and var

    Parameters
    ----------
    list_CP: `list`, optional (default: None)
        A list of anndata objects that store ATAC-seq data (Cells by Peaks)
    list_PM: `list`, optional (default: None)
        A list of anndata objects that store relation between Peaks and Motifs
    list_PK: `list`, optional (default: None)
        A list of anndata objects that store relation between Peaks and Kmers
    list_CG: `list`, optional (default: None)
        A list of anndata objects that store RNA-seq data (Cells by Genes)
    list_CC: `list`, optional (default: None)
        A list of anndata objects that store relation between Cells
        from two conditions

    Returns
    -------
    edges: `pd.DataFrame`
        The edges of the graph used for PBG training.
        Each line contains information about one edge.
        Using tabs as separators, each line contains the identifiers of
        the source entities, the relation types and the target entities.
    """

    if(sum(list(map(lambda x: x is None,
                    [list_CP,
                     list_PM,
                     list_PK,
                     list_CG,
                     list_CC]))) == 5):
        print('No graph is generated')

    # Collect the indices of entities
    dict_cells = dict()  # unique cell indices from all cell-centric datasets
    ids_genes = pd.Index([])
    ids_peaks = pd.Index([])
    ids_kmers = pd.Index([])
    ids_motifs = pd.Index([])

    if list_CP is not None:
        for adata in list_CP:
            ids_cells_i = adata.obs.index
            if(len(dict_cells) == 0):
                dict_cells[prefix_C] = ids_cells_i
            else:
                # check if cell indices are included in dict_cells
                flag_included = False
                for k in dict_cells.keys():
                    ids_cells_k = dict_cells[k]
                    if set(ids_cells_i) <= set(ids_cells_k):
                        flag_included = True
                        break
                if not flag_included:
                    # create a new set of entities
                    # when not all indices are included
                    dict_cells[f'{prefix_C}{len(dict_cells)+1}'] = ids_cells_i
            ids_peaks = ids_peaks.union(adata.var.index)
    if list_PM is not None:
        for adata in list_PM:
            ids_peaks = ids_peaks.union(adata.obs.index)
            ids_motifs = ids_motifs.union(adata.var.index)
    if list_PK is not None:
        for adata in list_PK:
            ids_peaks = ids_peaks.union(adata.obs.index)
            ids_kmers = ids_kmers.union(adata.var.index)
    if list_CG is not None:
        for adata in list_CG:
            ids_cells_i = adata.obs.index
            if(len(dict_cells) == 0):
                dict_cells[prefix_C] = ids_cells_i
            else:
                # check if cell indices are included in dict_cells
                flag_included = False
                for k in dict_cells.keys():
                    ids_cells_k = dict_cells[k]
                    if set(ids_cells_i) <= set(ids_cells_k):
                        flag_included = True
                        break
                if not flag_included:
                    # create a new set of entities
                    # when not all indices are included
                    dict_cells[f'{prefix_C}{len(dict_cells)+1}'] = ids_cells_i
            ids_genes = ids_genes.union(adata.var.index)

    dict_df_cells = dict()  # unique cell dataframes
    for k in dict_cells.keys():
        dict_df_cells[k] = pd.DataFrame(
            index=dict_cells[k],
            columns=['alias'],
            data=[f'{k}.{x}' for x in range(len(dict_cells[k]))])
    if(len(ids_genes) > 0):
        df_genes = pd.DataFrame(
                index=ids_genes,
                columns=['alias'],
                data=[f'{prefix_G}.{x}' for x in range(len(ids_genes))])
    if(len(ids_peaks) > 0):
        df_peaks = pd.DataFrame(
                index=ids_peaks,
                columns=['alias'],
                data=[f'{prefix_P}.{x}' for x in range(len(ids_peaks))])
    if(len(ids_kmers) > 0):
        df_kmers = pd.DataFrame(
                index=ids_kmers,
                columns=['alias'],
                data=[f'{prefix_K}.{x}' for x in range(len(ids_kmers))])
    if(len(ids_motifs) > 0):
        df_motifs = pd.DataFrame(
            index=ids_motifs,
            columns=['alias'],
            data=[f'{prefix_M}.{x}' for x in range(len(ids_motifs))])

    # generate edges
    col_names = ["source", "relation", "destination"]
    df_edges = pd.DataFrame(columns=col_names)
    id_r = 0

    if list_CP is not None:
        for adata in list_CP:
            # select reference of cells
            for key, df_cells in dict_df_cells.items():
                if set(adata.obs_names) <= set(df_cells.index):
                    break
            df_edges_x = pd.DataFrame(columns=col_names)
            df_edges_x['source'] = df_cells.loc[
                adata.obs_names[adata.X.nonzero()[0]],
                'alias'].values
            df_edges_x['relation'] = f'r{id_r}'
            df_edges_x['destination'] = df_peaks.loc[
                adata.var_names[adata.X.nonzero()[1]],
                'alias'].values
            print(f'relation{id_r}: '
                  f'source: {key}, '
                  f'destination: {prefix_P}\n'
                  f'#edges: {df_edges_x.shape[0]}')
            df_edges = df_edges.append(df_edges_x,
                                       ignore_index=True)
            id_r += 1
            adata.obs['pbg_id'] = df_cells.loc[adata.obs_names, 'alias'].copy()
            adata.var['pbg_id'] = df_peaks.loc[adata.var_names, 'alias'].copy()

    if list_PM is not None:
        for adata in list_PM:
            df_edges_x = pd.DataFrame(columns=col_names)
            df_edges_x['source'] = df_peaks.loc[
                adata.obs_names[adata.X.nonzero()[0]],
                'alias'].values
            df_edges_x['relation'] = f'r{id_r}'
            df_edges_x['destination'] = df_motifs.loc[
                adata.var_names[adata.X.nonzero()[1]],
                'alias'].values
            print(f'relation{id_r}: '
                  f'source: {prefix_P}, '
                  f'destination: {prefix_M}\n'
                  f'#edges: {df_edges_x.shape[0]}')
            df_edges = df_edges.append(df_edges_x,
                                       ignore_index=True)
            id_r += 1
            adata.obs['pbg_id'] = df_peaks.loc[adata.obs_names,
                                               'alias'].copy()
            adata.var['pbg_id'] = df_motifs.loc[adata.var_names,
                                                'alias'].copy()

    if list_PK is not None:
        for adata in list_PK:
            df_edges_x = pd.DataFrame(columns=col_names)
            df_edges_x['source'] = df_peaks.loc[
                adata.obs_names[adata.X.nonzero()[0]],
                'alias'].values
            df_edges_x['relation'] = f'r{id_r}'
            df_edges_x['destination'] = df_kmers.loc[
                adata.var_names[adata.X.nonzero()[1]],
                'alias'].values
            print(f'relation{id_r}: '
                  f'source: {prefix_P}, '
                  f'destination: {prefix_K}\n'
                  f'#edges: {df_edges_x.shape[0]}')
            df_edges = df_edges.append(df_edges_x,
                                       ignore_index=True)
            id_r += 1
            adata.obs['pbg_id'] = df_peaks.loc[adata.obs_names,
                                               'alias'].copy()
            adata.var['pbg_id'] = df_kmers.loc[adata.var_names,
                                               'alias'].copy()

    if list_CG is not None:
        for adata in list_CG:
            # select reference of cells
            for key, df_cells in dict_df_cells.items():
                if set(adata.obs_names) <= set(df_cells.index):
                    break
            expr_level = np.unique(adata.layers['disc'].data)
            for lvl in expr_level:
                df_edges_x = pd.DataFrame(columns=col_names)
                df_edges_x['source'] = df_cells.loc[
                    adata.obs_names[(adata.layers['disc'] == lvl)
                                    .astype(int).nonzero()[0]],
                    'alias'].values
                df_edges_x['relation'] = f'r{id_r}'
                df_edges_x['destination'] = df_genes.loc[
                    adata.var_names[(adata.layers['disc'] == lvl)
                                    .astype(int).nonzero()[1]],
                    'alias'].values
                print(f'relation{id_r}: '
                      f'source: {key}, '
                      f'destination: {prefix_G}\n'
                      f'#edges: {df_edges_x.shape[0]}')
                df_edges = df_edges.append(df_edges_x,
                                           ignore_index=True)
                id_r += 1

            adata.obs['pbg_id'] = df_cells.loc[adata.obs_names, 'alias'].copy()
            adata.var['pbg_id'] = df_genes.loc[adata.var_names, 'alias'].copy()

    if list_CC is not None:
        for adata in list_CC:
            # select reference of cells
            for key_obs, df_cells_obs in dict_df_cells.items():
                if set(adata.obs_names) <= set(df_cells_obs.index):
                    break
            for key_var, df_cells_var in dict_df_cells.items():
                if set(adata.var_names) <= set(df_cells_var.index):
                    break

            df_edges_x = pd.DataFrame(columns=col_names)
            df_edges_x['source'] = df_cells_obs.loc[
                adata.obs_names[adata.X.nonzero()[0]],
                'alias'].values
            df_edges_x['relation'] = f'r{id_r}'
            df_edges_x['destination'] = df_cells_var.loc[
                adata.var_names[adata.X.nonzero()[1]],
                'alias'].values
            print(f'relation{id_r}: '
                  f'source: {key_obs}, '
                  f'destination: {key_var}\n'
                  f'#edges: {df_edges_x.shape[0]}')
            df_edges = df_edges.append(df_edges_x,
                                       ignore_index=True)
            id_r += 1
            adata.obs['pbg_id'] = df_cells_obs.loc[adata.obs_names,
                                                   'alias'].copy()
            adata.var['pbg_id'] = df_cells_var.loc[adata.var_names,
                                                   'alias'].copy()

    print(f'Number of total edges: {df_edges.shape[0]}')
    filepath = os.path.join(settings.workdir, 'pbg')
    if(not os.path.exists(filepath)):
        os.makedirs(filepath)
    print(f'Writing "{filename}" to {filepath} ...')
    df_edges.to_csv(os.path.join(filepath, filename),
                    header=False,
                    index=False,
                    sep='\t')
    print("Finished.")
    if copy:
        return df_edges
    else:
        return None
