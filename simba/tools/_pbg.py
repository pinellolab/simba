"""PyTorch-BigGraph (PBG) for learning graph embeddings"""

import numpy as np
import pandas as pd
import os

from pathlib import Path
import attr
from torchbiggraph.config import (
    add_to_sys_path,
    ConfigFileLoader
)
from torchbiggraph.converters.importers import (
    convert_input_data,
    TSVEdgelistReader
)
from torchbiggraph.train import train
from torchbiggraph.util import (
    set_logging_verbosity,
    setup_logging,
    SubprocessInitializer,
)

from .._settings import settings


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
              filename='pbg_graph.txt',
              config=None,
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
    prefix_C: `str`, optional (default: 'C')
        Prefix to indicate the entity type of cells
    prefix_G: `str`, optional (default: 'G')
        Prefix to indicate the entity type of genes

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
    settings.set_pbg_params(config)

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
        settings.pbg_params['entities'][k] = {'num_partitions': 1}
    if(len(ids_genes) > 0):
        df_genes = pd.DataFrame(
                index=ids_genes,
                columns=['alias'],
                data=[f'{prefix_G}.{x}' for x in range(len(ids_genes))])
        settings.pbg_params['entities'][prefix_G] = {'num_partitions': 1}
    if(len(ids_peaks) > 0):
        df_peaks = pd.DataFrame(
                index=ids_peaks,
                columns=['alias'],
                data=[f'{prefix_P}.{x}' for x in range(len(ids_peaks))])
        settings.pbg_params['entities'][prefix_P] = {'num_partitions': 1}
    if(len(ids_kmers) > 0):
        df_kmers = pd.DataFrame(
                index=ids_kmers,
                columns=['alias'],
                data=[f'{prefix_K}.{x}' for x in range(len(ids_kmers))])
        settings.pbg_params['entities'][prefix_K] = {'num_partitions': 1}
    if(len(ids_motifs) > 0):
        df_motifs = pd.DataFrame(
            index=ids_motifs,
            columns=['alias'],
            data=[f'{prefix_M}.{x}' for x in range(len(ids_motifs))])
        settings.pbg_params['entities'][prefix_M] = {'num_partitions': 1}

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
            settings.pbg_params['relations'].append(
                {'name': f'r{id_r}',
                 'lhs': f'{key}',
                 'rhs': f'{prefix_P}',
                 'operator': 'none',
                 'weight': 1.0
                 })
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
            settings.pbg_params['relations'].append(
                {'name': f'r{id_r}',
                 'lhs': f'{prefix_P}',
                 'rhs': f'{prefix_M}',
                 'operator': 'none',
                 'weight': 1.0
                 })
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
            settings.pbg_params['relations'].append(
                {'name': f'r{id_r}',
                 'lhs': f'{prefix_P}',
                 'rhs': f'{prefix_K}',
                 'operator': 'none',
                 'weight': 1.0
                 })
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
                settings.pbg_params['relations'].append(
                    {'name': f'r{id_r}',
                     'lhs': f'{key}',
                     'rhs': f'{prefix_G}',
                     'operator': 'none',
                     'weight': 1.0
                     })
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
            settings.pbg_params['relations'].append(
                {'name': f'r{id_r}',
                 'lhs': f'{key_obs}',
                 'rhs': f'{key_var}',
                 'operator': 'none',
                 'weight': 1.0
                 })
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


def pbg_train(filename='pbg_graph.txt', overrides=None):
    """PBG training
    Parameters
    ----------
    adata: AnnData
        Annotated data matrix.

    Returns
    -------
    updates `adata` with the following fields:
    """

    os.environ["OMP_NUM_THREADS"] = "1"
    loader = ConfigFileLoader(settings.pbg_params)
    config = loader.load_config(overrides)
    set_logging_verbosity(config.verbose)

    filepath = os.path.join(settings.workdir, 'pbg')
    list_filenames = [os.path.join(filepath, filename)]
    input_edge_paths = [Path(name) for name in list_filenames]
    print("Converting input data ...")
    convert_input_data(
        config.entities,
        config.relations,
        config.entity_path,
        config.edge_paths,
        input_edge_paths,
        TSVEdgelistReader(lhs_col=0, rhs_col=2, rel_col=1),
        dynamic_relations=config.dynamic_relations,
        )

    subprocess_init = SubprocessInitializer()
    subprocess_init.register(setup_logging, config.verbose)
    subprocess_init.register(add_to_sys_path, loader.config_dir.name)

    train_config = attr.evolve(config, edge_paths=config.edge_paths)
    train(train_config, subprocess_init=subprocess_init)
