"""PyTorch-BigGraph (PBG) for learning graph embeddings"""

import numpy as np
import pandas as pd
import os
import json

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
              list_adata=None,
              prefix_C='C',
              prefix_P='P',
              prefix_M='M',
              prefix_K='K',
              prefix_G='G',
              prefix='E',
              layer='simba',
              copy=False,
              dirname='graph0',
              add_edge_weights=None,
              use_highly_variable=True,
              use_top_pcs=True,
              use_top_pcs_CP=None,
              use_top_pcs_PM=None,
              use_top_pcs_PK=None
              ):
    """Generate graph for PBG training.

    Observations and variables of each Anndata object will be encoded
    as nodes (entities). The non-zero values in `.layers['simba']` (by default)
    or `.X` (if `.layers['simba']` does not exist) indicate the edges
    between nodes. The values of `.layers['simba']` or `.X` will be used
    as the edge weights if `add_edge_weights` True.

    Nodes between different anndata objects will be automatically matched
    based on `.obs_names` and `.var_names`. Each anndata object indicates one
    or more relation types.

    It also generates an accompanying file 'entity_alias.tsv' to map
    the indices to the aliases used in the graph.

    Note when `add_edge_weights` is True, `list_CG` will only generate
    one relation of cells and genes, as opposed to multiple relations
    based on discretized levels.

    Parameters
    ----------
    list_CP: `list`, optional (default: None)
        A list of anndata objects that store ATAC-seq data (Cells by Peaks)
        The default weight of cell-peak relation type is 1.0.
    list_PM: `list`, optional (default: None)
        A list of anndata objects that store relation between Peaks and Motifs
    list_PK: `list`, optional (default: None)
        A list of anndata objects that store relation between Peaks and Kmers
    list_CG: `list`, optional (default: None)
        A list of anndata objects that store RNA-seq data (Cells by Genes)
    list_CC: `list`, optional (default: None)
        A list of anndata objects that store relation between Cells
        from two conditions
    list_adata: `list`, optional (default: None)
        A list of anndata objects. `.obs_names` and `.var_names`
        between anndata objects will be automatically matched.
        If `list_adata` is specified, the other lists including
        `list_CP`, `list_PM`,`list_PK`, `list_CG`, `list_CC` will be ignored.
    prefix_C: `str`, optional (default: 'C')
        Prefix to indicate the entity type of cells
    prefix_G: `str`, optional (default: 'G')
        Prefix to indicate the entity type of genes
    prefix: `str`, optional (default: 'E')
        Prefix to indicate general entities in `list_adata`
    layer: `str`, optional (default: 'simba')
        The layer in AnnData to use for constructing the graph.
        If `layer` is None or the specificed layer does not exist,
        `.X` in AnnData will be used instead.
    dirname: `str`, (default: 'graph0')
        The name of the directory in which each graph will be stored
    add_edge_weights: `bool`, optional (default: None)
        If True, the column of edge weigths will be added.
        If `list_adata` is specified, `add_edge_weights` is set True
        by default. Otherwise, it is set False.
    use_highly_variable: `bool`, optional (default: True)
        Use highly variable genes. Only valid for list_CG.
    use_top_pcs: `bool`, optional (default: True)
        Use top-PCs-associated features for CP, PM, PK
    use_top_pcs_CP: `bool`, optional (default: None)
        Use top-PCs-associated features for CP
        Once specified, it will overwrite `use_top_pcs`
    use_top_pcs_PM: `bool`, optional (default: None)
        Use top-PCs-associated features for PM
        Once specified, it will overwrite `use_top_pcs`
    use_top_pcs_PK: `bool`, optional (default: None)
        Use top-PCs-associated features for PK
        Once specified, it will overwrite `use_top_pcs
    copy: `bool`, optional (default: False)
        If True, it returns the graph file as a data frame

    Returns
    -------
    If `copy` is True,
    edges: `pd.DataFrame`
        The edges of the graph used for PBG training.
        Each line contains information about one edge.
        Using tabs as separators, each line contains the identifiers of
        the source entities, the relation types and the target entities.

    updates `.settings.pbg_params` with the following parameters.
    entity_path: `str`
        The path of the directory containing entity count files.
    edge_paths: `list`
        A list of paths to directories containing (partitioned) edgelists.
        Typically a single path is provided.
    entities: `dict`
        The entity types.
    relations: `list`
        The relation types.

    updates `.settings.graph_stats` with the following parameters.
    `dirname`: `dict`
        Statistics of input graph
    """

    if sum(list(map(lambda x: x is None,
                    [list_CP,
                     list_PM,
                     list_PK,
                     list_CG,
                     list_CC,
                     list_adata]))) == 6:
        return 'No graph is generated'

    filepath = os.path.join(settings.workdir, 'pbg', dirname)
    settings.pbg_params['entity_path'] = \
        os.path.join(filepath, "input/entity")
    settings.pbg_params['edge_paths'] = \
        [os.path.join(filepath, "input/edge"), ]
    settings.pbg_params['entity_path'] = \
        os.path.join(filepath, "input/entity")
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    if add_edge_weights is None:
        if list_adata is None:
            add_edge_weights = False
        else:
            add_edge_weights = True

    if list_adata is not None:
        id_ent = pd.Index([])  # ids of all entities
        dict_ent_type = dict()
        ctr_ent = 0  # counter for entity types
        entity_alias = pd.DataFrame(columns=['alias'])
        dict_graph_stats = dict()
        if add_edge_weights:
            col_names = ["source", "relation", "destination", "weight"]
        else:
            col_names = ["source", "relation", "destination"]
        df_edges = pd.DataFrame(columns=col_names)
        settings.pbg_params['relations'] = []
        for ctr_rel, adata_ori in enumerate(list_adata):
            obs_names = adata_ori.obs_names
            var_names = adata_ori.var_names
            if len(set(obs_names).intersection(id_ent)) == 0:
                prefix_i = f'{prefix}{ctr_ent}'
                id_ent = id_ent.union(adata_ori.obs_names)
                entity_alias_obs = pd.DataFrame(
                    index=obs_names,
                    columns=['alias'],
                    data=[f'{prefix_i}.{x}'
                          for x in range(len(obs_names))])
                settings.pbg_params['entities'][
                    prefix_i] = {'num_partitions': 1}
                dict_ent_type[prefix_i] = obs_names
                entity_alias = pd.concat(
                        [entity_alias, entity_alias_obs],
                        ignore_index=False)
                obs_type = prefix_i
                ctr_ent += 1
            else:
                for k, item in dict_ent_type.items():
                    if len(set(obs_names).intersection(item)) > 0:
                        obs_type = k
                        break
                if not set(obs_names).issubset(id_ent):
                    id_ent = id_ent.union(adata_ori.obs_names)
                    adt_obs_names = list(set(obs_names)-set(item))
                    entity_alias_obs = pd.DataFrame(
                        index=adt_obs_names,
                        columns=['alias'],
                        data=[f'{prefix_i}.{len(item)+x}'
                              for x in range(len(adt_obs_names))])
                    dict_ent_type[obs_type] = obs_names.union(adt_obs_names)
                    entity_alias = pd.concat(
                            [entity_alias, entity_alias_obs],
                            ignore_index=False)
            if len(set(var_names).intersection(id_ent)) == 0:
                prefix_i = f'{prefix}{ctr_ent}'
                id_ent = id_ent.union(adata_ori.var_names)
                entity_alias_var = pd.DataFrame(
                    index=var_names,
                    columns=['alias'],
                    data=[f'{prefix_i}.{x}'
                          for x in range(len(var_names))])
                settings.pbg_params['entities'][
                    prefix_i] = {'num_partitions': 1}
                dict_ent_type[prefix_i] = var_names
                entity_alias = pd.concat(
                    [entity_alias, entity_alias_var],
                    ignore_index=False)
                var_type = prefix_i
                ctr_ent += 1
            else:
                for k, item in dict_ent_type.items():
                    if len(set(var_names).intersection(item)) > 0:
                        var_type = k
                        break
                if not set(var_names).issubset(id_ent):
                    id_ent = id_ent.union(adata_ori.var_names)
                    adt_var_names = list(set(var_names)-set(item))
                    entity_alias_var = pd.DataFrame(
                        index=adt_var_names,
                        columns=['alias'],
                        data=[f'{prefix_i}.{len(item)+x}'
                              for x in range(len(adt_var_names))])
                    dict_ent_type[var_type] = var_names.union(adt_var_names)
                    entity_alias = pd.concat(
                        [entity_alias, entity_alias_var],
                        ignore_index=False)

            # generate edges
            if layer is not None:
                if layer in adata_ori.layers.keys():
                    arr_simba = adata_ori.layers[layer]
                else:
                    print(f'`{layer}` does not exist in adata {ctr_rel} '
                          'in `list_adata`.`.X` is being used instead.')
                    arr_simba = adata_ori.X
            else:
                arr_simba = adata_ori.X
            _row, _col = arr_simba.nonzero()
            df_edges_x = pd.DataFrame(columns=col_names)
            df_edges_x['source'] = entity_alias.loc[
                obs_names[_row], 'alias'].values
            df_edges_x['relation'] = f'r{ctr_rel}'
            df_edges_x['destination'] = entity_alias.loc[
                var_names[_col], 'alias'].values
            if add_edge_weights:
                df_edges_x['weight'] = \
                    arr_simba[_row, _col].A.flatten()
            settings.pbg_params['relations'].append({
                'name': f'r{ctr_rel}',
                'lhs': f'{obs_type}',
                'rhs': f'{var_type}',
                'operator': 'none',
                'weight': 1.0
                })
            dict_graph_stats[f'relation{ctr_rel}'] = {
                'source': obs_type,
                'destination': var_type,
                'n_edges': df_edges_x.shape[0]}
            print(
                f'relation{ctr_rel}: '
                f'source: {obs_type}, '
                f'destination: {var_type}\n'
                f'#edges: {df_edges_x.shape[0]}')

            df_edges = pd.concat(
                [df_edges, df_edges_x],
                ignore_index=True)
            adata_ori.obs['pbg_id'] = ""
            adata_ori.var['pbg_id'] = ""
            adata_ori.obs.loc[obs_names, 'pbg_id'] = \
                entity_alias.loc[obs_names, 'alias'].copy()
            adata_ori.var.loc[var_names, 'pbg_id'] = \
                entity_alias.loc[var_names, 'alias'].copy()

    else:
        # Collect the indices of entities
        dict_cells = dict()  # unique cell indices from all anndata objects
        ids_genes = pd.Index([])
        ids_peaks = pd.Index([])
        ids_kmers = pd.Index([])
        ids_motifs = pd.Index([])

        if list_CP is not None:
            for adata_ori in list_CP:
                if use_top_pcs_CP is None:
                    flag_top_pcs = use_top_pcs
                else:
                    flag_top_pcs = use_top_pcs_CP
                if flag_top_pcs:
                    adata = adata_ori[:, adata_ori.var['top_pcs']].copy()
                else:
                    adata = adata_ori.copy()
                ids_cells_i = adata.obs.index
                if len(dict_cells) == 0:
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
                        dict_cells[
                            f'{prefix_C}{len(dict_cells)+1}'] = \
                                ids_cells_i
                ids_peaks = ids_peaks.union(adata.var.index)
        if list_PM is not None:
            for adata_ori in list_PM:
                if use_top_pcs_PM is None:
                    flag_top_pcs = use_top_pcs
                else:
                    flag_top_pcs = use_top_pcs_PM
                if flag_top_pcs:
                    adata = adata_ori[:, adata_ori.var['top_pcs']].copy()
                else:
                    adata = adata_ori.copy()
                ids_peaks = ids_peaks.union(adata.obs.index)
                ids_motifs = ids_motifs.union(adata.var.index)
        if list_PK is not None:
            for adata_ori in list_PK:
                if use_top_pcs_PK is None:
                    flag_top_pcs = use_top_pcs
                else:
                    flag_top_pcs = use_top_pcs_PK
                if flag_top_pcs:
                    adata = adata_ori[:, adata_ori.var['top_pcs']].copy()
                else:
                    adata = adata_ori.copy()
                ids_peaks = ids_peaks.union(adata.obs.index)
                ids_kmers = ids_kmers.union(adata.var.index)
        if list_CG is not None:
            for adata_ori in list_CG:
                if use_highly_variable:
                    adata = adata_ori[
                        :, adata_ori.var['highly_variable']].copy()
                else:
                    adata = adata_ori.copy()
                ids_cells_i = adata.obs.index
                if len(dict_cells) == 0:
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
                        dict_cells[
                            f'{prefix_C}{len(dict_cells)+1}'] = \
                                ids_cells_i
                ids_genes = ids_genes.union(adata.var.index)

        entity_alias = pd.DataFrame(columns=['alias'])
        dict_df_cells = dict()  # unique cell dataframes
        for k in dict_cells.keys():
            dict_df_cells[k] = pd.DataFrame(
                index=dict_cells[k],
                columns=['alias'],
                data=[f'{k}.{x}' for x in range(len(dict_cells[k]))])
            settings.pbg_params['entities'][k] = {'num_partitions': 1}
            entity_alias = pd.concat(
                [entity_alias, dict_df_cells[k]],
                ignore_index=False)
        if len(ids_genes) > 0:
            df_genes = pd.DataFrame(
                    index=ids_genes,
                    columns=['alias'],
                    data=[f'{prefix_G}.{x}' for x in range(len(ids_genes))])
            settings.pbg_params['entities'][prefix_G] = {'num_partitions': 1}
            entity_alias = pd.concat(
                [entity_alias, df_genes],
                ignore_index=False)
        if len(ids_peaks) > 0:
            df_peaks = pd.DataFrame(
                    index=ids_peaks,
                    columns=['alias'],
                    data=[f'{prefix_P}.{x}' for x in range(len(ids_peaks))])
            settings.pbg_params['entities'][prefix_P] = {'num_partitions': 1}
            entity_alias = pd.concat(
                [entity_alias, df_peaks],
                ignore_index=False)
        if len(ids_kmers) > 0:
            df_kmers = pd.DataFrame(
                    index=ids_kmers,
                    columns=['alias'],
                    data=[f'{prefix_K}.{x}' for x in range(len(ids_kmers))])
            settings.pbg_params['entities'][prefix_K] = {'num_partitions': 1}
            entity_alias = pd.concat(
                [entity_alias, df_kmers],
                ignore_index=False)
        if len(ids_motifs) > 0:
            df_motifs = pd.DataFrame(
                index=ids_motifs,
                columns=['alias'],
                data=[f'{prefix_M}.{x}' for x in range(len(ids_motifs))])
            settings.pbg_params['entities'][prefix_M] = {'num_partitions': 1}
            entity_alias = pd.concat(
                [entity_alias, df_motifs],
                ignore_index=False)

        # generate edges
        dict_graph_stats = dict()
        if add_edge_weights:
            col_names = ["source", "relation", "destination", "weight"]
        else:
            col_names = ["source", "relation", "destination"]
        df_edges = pd.DataFrame(columns=col_names)
        id_r = 0
        settings.pbg_params['relations'] = []

        if list_CP is not None:
            for i, adata_ori in enumerate(list_CP):
                if use_top_pcs:
                    adata = adata_ori[:, adata_ori.var['top_pcs']].copy()
                else:
                    adata = adata_ori.copy()
                # select reference of cells
                for key, df_cells in dict_df_cells.items():
                    if set(adata.obs_names) <= set(df_cells.index):
                        break
                if layer is not None:
                    if layer in adata.layers.keys():
                        arr_simba = adata.layers[layer]
                    else:
                        print(f'`{layer}` does not exist in anndata {i} '
                              'in `list_CP`.`.X` is being used instead.')
                        arr_simba = adata.X
                else:
                    arr_simba = adata.X
                _row, _col = arr_simba.nonzero()
                df_edges_x = pd.DataFrame(columns=col_names)
                df_edges_x['source'] = df_cells.loc[
                    adata.obs_names[_row], 'alias'].values
                df_edges_x['relation'] = f'r{id_r}'
                df_edges_x['destination'] = df_peaks.loc[
                    adata.var_names[_col], 'alias'].values
                if add_edge_weights:
                    df_edges_x['weight'] = \
                        arr_simba[_row, _col].A.flatten()
                    settings.pbg_params['relations'].append({
                        'name': f'r{id_r}',
                        'lhs': f'{key}',
                        'rhs': f'{prefix_P}',
                        'operator': 'none',
                        'weight': 1.0
                        })
                else:
                    settings.pbg_params['relations'].append({
                        'name': f'r{id_r}',
                        'lhs': f'{key}',
                        'rhs': f'{prefix_P}',
                        'operator': 'none',
                        'weight': 1.0
                        })
                dict_graph_stats[f'relation{id_r}'] = {
                    'source': key,
                    'destination': prefix_P,
                    'n_edges': df_edges_x.shape[0]}
                print(
                    f'relation{id_r}: '
                    f'source: {key}, '
                    f'destination: {prefix_P}\n'
                    f'#edges: {df_edges_x.shape[0]}')

                id_r += 1
                df_edges = pd.concat(
                    [df_edges, df_edges_x],
                    ignore_index=True)
                adata_ori.obs['pbg_id'] = ""
                adata_ori.var['pbg_id'] = ""
                adata_ori.obs.loc[adata.obs_names, 'pbg_id'] = \
                    df_cells.loc[adata.obs_names, 'alias'].copy()
                adata_ori.var.loc[adata.var_names, 'pbg_id'] = \
                    df_peaks.loc[adata.var_names, 'alias'].copy()

        if list_PM is not None:
            for i, adata_ori in enumerate(list_PM):
                if use_top_pcs:
                    adata = adata_ori[:, adata_ori.var['top_pcs']].copy()
                else:
                    adata = adata_ori.copy()
                if layer is not None:
                    if layer in adata.layers.keys():
                        arr_simba = adata.layers[layer]
                    else:
                        print(f'`{layer}` does not exist in anndata {i} '
                              'in `list_PM`.`.X` is being used instead.')
                        arr_simba = adata.X
                else:
                    arr_simba = adata.X
                _row, _col = arr_simba.nonzero()
                df_edges_x = pd.DataFrame(columns=col_names)
                df_edges_x['source'] = df_peaks.loc[
                    adata.obs_names[_row], 'alias'].values
                df_edges_x['relation'] = f'r{id_r}'
                df_edges_x['destination'] = df_motifs.loc[
                    adata.var_names[_col], 'alias'].values
                if add_edge_weights:
                    df_edges_x['weight'] = \
                        arr_simba[_row, _col].A.flatten()
                    settings.pbg_params['relations'].append({
                        'name': f'r{id_r}',
                        'lhs': f'{prefix_P}',
                        'rhs': f'{prefix_M}',
                        'operator': 'none',
                        'weight': 1.0
                        })
                else:
                    settings.pbg_params['relations'].append({
                        'name': f'r{id_r}',
                        'lhs': f'{prefix_P}',
                        'rhs': f'{prefix_M}',
                        'operator': 'none',
                        'weight': 0.2
                        })
                dict_graph_stats[f'relation{id_r}'] = {
                    'source': prefix_P,
                    'destination': prefix_M,
                    'n_edges': df_edges_x.shape[0]}
                print(
                    f'relation{id_r}: '
                    f'source: {prefix_P}, '
                    f'destination: {prefix_M}\n'
                    f'#edges: {df_edges_x.shape[0]}')

                id_r += 1
                df_edges = pd.concat(
                    [df_edges, df_edges_x],
                    ignore_index=True)
                adata_ori.obs['pbg_id'] = ""
                adata_ori.var['pbg_id'] = ""
                adata_ori.obs.loc[adata.obs_names, 'pbg_id'] = \
                    df_peaks.loc[adata.obs_names, 'alias'].copy()
                adata_ori.var.loc[adata.var_names, 'pbg_id'] = \
                    df_motifs.loc[adata.var_names, 'alias'].copy()

        if list_PK is not None:
            for i, adata_ori in enumerate(list_PK):
                if use_top_pcs:
                    adata = adata_ori[:, adata_ori.var['top_pcs']].copy()
                else:
                    adata = adata_ori.copy()
                if layer is not None:
                    if layer in adata.layers.keys():
                        arr_simba = adata.layers[layer]
                    else:
                        print(f'`{layer}` does not exist in anndata {i} '
                              'in `list_PK`.`.X` is being used instead.')
                        arr_simba = adata.X
                else:
                    arr_simba = adata.X
                _row, _col = arr_simba.nonzero()
                df_edges_x = pd.DataFrame(columns=col_names)
                df_edges_x['source'] = df_peaks.loc[
                    adata.obs_names[_row], 'alias'].values
                df_edges_x['relation'] = f'r{id_r}'
                df_edges_x['destination'] = df_kmers.loc[
                    adata.var_names[_col], 'alias'].values
                if add_edge_weights:
                    df_edges_x['weight'] = \
                        arr_simba[_row, _col].A.flatten()
                    settings.pbg_params['relations'].append({
                        'name': f'r{id_r}',
                        'lhs': f'{prefix_P}',
                        'rhs': f'{prefix_K}',
                        'operator': 'none',
                        'weight': 1
                        })
                else:
                    settings.pbg_params['relations'].append({
                        'name': f'r{id_r}',
                        'lhs': f'{prefix_P}',
                        'rhs': f'{prefix_K}',
                        'operator': 'none',
                        'weight': 0.02
                        })
                print(
                    f'relation{id_r}: '
                    f'source: {prefix_P}, '
                    f'destination: {prefix_K}\n'
                    f'#edges: {df_edges_x.shape[0]}')
                dict_graph_stats[f'relation{id_r}'] = {
                    'source': prefix_P,
                    'destination': prefix_K,
                    'n_edges': df_edges_x.shape[0]}

                id_r += 1
                df_edges = pd.concat(
                    [df_edges, df_edges_x],
                    ignore_index=True)
                adata_ori.obs['pbg_id'] = ""
                adata_ori.var['pbg_id'] = ""
                adata_ori.obs.loc[adata.obs_names, 'pbg_id'] = \
                    df_peaks.loc[adata.obs_names, 'alias'].copy()
                adata_ori.var.loc[adata.var_names, 'pbg_id'] = \
                    df_kmers.loc[adata.var_names, 'alias'].copy()

        if list_CG is not None:
            for i, adata_ori in enumerate(list_CG):
                if use_highly_variable:
                    adata = adata_ori[
                        :, adata_ori.var['highly_variable']].copy()
                else:
                    adata = adata_ori.copy()
                # select reference of cells
                for key, df_cells in dict_df_cells.items():
                    if set(adata.obs_names) <= set(df_cells.index):
                        break
                if layer is not None:
                    if layer in adata.layers.keys():
                        arr_simba = adata.layers[layer]
                    else:
                        print(f'`{layer}` does not exist in anndata {i} '
                              'in `list_CG`.`.X` is being used instead.')
                        arr_simba = adata.X
                else:
                    arr_simba = adata.X
                if add_edge_weights:
                    _row, _col = arr_simba.nonzero()
                    df_edges_x = pd.DataFrame(columns=col_names)
                    df_edges_x['source'] = df_cells.loc[
                        adata.obs_names[_row], 'alias'].values
                    df_edges_x['relation'] = f'r{id_r}'
                    df_edges_x['destination'] = df_genes.loc[
                        adata.var_names[_col], 'alias'].values
                    df_edges_x['weight'] = \
                        arr_simba[_row, _col].A.flatten()
                    settings.pbg_params['relations'].append({
                        'name': f'r{id_r}',
                        'lhs': f'{key}',
                        'rhs': f'{prefix_G}',
                        'operator': 'none',
                        'weight': 1.0,
                        })
                    print(
                        f'relation{id_r}: '
                        f'source: {key}, '
                        f'destination: {prefix_G}\n'
                        f'#edges: {df_edges_x.shape[0]}')
                    dict_graph_stats[f'relation{id_r}'] = {
                        'source': key,
                        'destination': prefix_G,
                        'n_edges': df_edges_x.shape[0]}
                    id_r += 1
                    df_edges = pd.concat(
                        [df_edges, df_edges_x],
                        ignore_index=True)
                else:
                    expr_level = np.unique(arr_simba.data)
                    expr_weight = np.linspace(
                        start=1, stop=5, num=len(expr_level))
                    for i_lvl, lvl in enumerate(expr_level):
                        _row, _col = (arr_simba == lvl).astype(int).nonzero()
                        df_edges_x = pd.DataFrame(columns=col_names)
                        df_edges_x['source'] = df_cells.loc[
                            adata.obs_names[_row], 'alias'].values
                        df_edges_x['relation'] = f'r{id_r}'
                        df_edges_x['destination'] = df_genes.loc[
                            adata.var_names[_col], 'alias'].values
                        settings.pbg_params['relations'].append({
                            'name': f'r{id_r}',
                            'lhs': f'{key}',
                            'rhs': f'{prefix_G}',
                            'operator': 'none',
                            'weight': round(expr_weight[i_lvl], 2),
                            })
                        print(
                            f'relation{id_r}: '
                            f'source: {key}, '
                            f'destination: {prefix_G}\n'
                            f'#edges: {df_edges_x.shape[0]}')
                        dict_graph_stats[f'relation{id_r}'] = {
                            'source': key,
                            'destination': prefix_G,
                            'n_edges': df_edges_x.shape[0]}
                        id_r += 1
                        df_edges = pd.concat(
                            [df_edges, df_edges_x], ignore_index=True)

                adata_ori.obs['pbg_id'] = ""
                adata_ori.var['pbg_id'] = ""
                adata_ori.obs.loc[adata.obs_names, 'pbg_id'] = \
                    df_cells.loc[adata.obs_names, 'alias'].copy()
                adata_ori.var.loc[adata.var_names, 'pbg_id'] = \
                    df_genes.loc[adata.var_names, 'alias'].copy()

        if list_CC is not None:
            for i, adata in enumerate(list_CC):
                # select reference of cells
                for key_obs, df_cells_obs in dict_df_cells.items():
                    if set(adata.obs_names) <= set(df_cells_obs.index):
                        break
                for key_var, df_cells_var in dict_df_cells.items():
                    if set(adata.var_names) <= set(df_cells_var.index):
                        break
                if layer is not None:
                    if layer in adata.layers.keys():
                        arr_simba = adata.layers[layer]
                    else:
                        print(f'`{layer}` does not exist in anndata {i} '
                              'in `list_PM`.`.X` is being used instead.')
                        arr_simba = adata.X
                else:
                    arr_simba = adata.X
                _row, _col = arr_simba.nonzero()
                #  edges between ref and query
                df_edges_x = pd.DataFrame(columns=col_names)
                df_edges_x['source'] = df_cells_obs.loc[
                    adata.obs_names[_row], 'alias'].values
                df_edges_x['relation'] = f'r{id_r}'
                df_edges_x['destination'] = df_cells_var.loc[
                    adata.var_names[_col], 'alias'].values
                if add_edge_weights:
                    df_edges_x['weight'] = \
                        arr_simba[_row, _col].A.flatten()
                    settings.pbg_params['relations'].append({
                        'name': f'r{id_r}',
                        'lhs': f'{key_obs}',
                        'rhs': f'{key_var}',
                        'operator': 'none',
                        'weight': 1.0
                        })
                else:
                    settings.pbg_params['relations'].append({
                        'name': f'r{id_r}',
                        'lhs': f'{key_obs}',
                        'rhs': f'{key_var}',
                        'operator': 'none',
                        'weight': 10.0
                        })
                print(
                    f'relation{id_r}: '
                    f'source: {key_obs}, '
                    f'destination: {key_var}\n'
                    f'#edges: {df_edges_x.shape[0]}')
                dict_graph_stats[f'relation{id_r}'] = {
                    'source': key_obs,
                    'destination': key_var,
                    'n_edges': df_edges_x.shape[0]}

                id_r += 1
                df_edges = pd.concat(
                    [df_edges, df_edges_x],
                    ignore_index=True)
                adata.obs['pbg_id'] = df_cells_obs.loc[
                    adata.obs_names, 'alias'].copy()
                adata.var['pbg_id'] = df_cells_var.loc[
                    adata.var_names, 'alias'].copy()

    print(f'Total number of edges: {df_edges.shape[0]}')
    dict_graph_stats['n_edges'] = df_edges.shape[0]
    settings.graph_stats[dirname] = dict_graph_stats

    print(f'Writing graph file "pbg_graph.txt" to "{filepath}" ...')
    df_edges.to_csv(os.path.join(filepath, "pbg_graph.txt"),
                    header=False,
                    index=False,
                    sep='\t')
    entity_alias.to_csv(os.path.join(filepath, 'entity_alias.txt'),
                        header=True,
                        index=True,
                        sep='\t')
    with open(os.path.join(filepath, 'graph_stats.json'), 'w') as fp:
        json.dump(dict_graph_stats,
                  fp,
                  sort_keys=True,
                  indent=4,
                  separators=(',', ': '))
    print("Finished.")
    if copy:
        return df_edges
    else:
        return None


def pbg_train(dirname=None,
              pbg_params=None,
              output='model',
              auto_wd=True,
              save_wd=False,
              use_edge_weights=False):
    """PBG training

    Parameters
    ----------
    dirname: `str`, optional (default: None)
        The name of the directory in which graph is stored
        If None, it will be inferred from `pbg_params['entity_path']`
    pbg_params: `dict`, optional (default: None)
        Configuration for pbg training.
        If specified, it will be used instead of the default setting
    output: `str`, optional (default: 'model')
        The name of the directory where training output will be written to.
        It overrides `pbg_params` if `checkpoint_path` is specified in it
    auto_wd: `bool`, optional (default: True)
        If True, it will override `pbg_params['wd']` with a new weight decay
        estimated based on training sample size
        Recommended for relative small training sample size (<1e7)
    save_wd: `bool`, optional (default: False)
        If True, estimated `wd` will be saved to `settings.pbg_params['wd']`
    use_edge_weights: `bool`, optional (default: False)
        If True, the edge weights are used for the training;
        If False, the weights of relation types are used instead,
        and edge weights will be ignored.

    Returns
    -------
    updates `settings.pbg_params` with the following parameter
    checkpoint_path:
        The path to the directory where checkpoints (and thus the output)
        will be written to.
        If checkpoints are found in it, training will resume from them.
    """

    if pbg_params is None:
        pbg_params = settings.pbg_params.copy()
    else:
        assert isinstance(pbg_params, dict),\
            "`pbg_params` must be dict"

    if dirname is None:
        filepath = Path(pbg_params['entity_path']).parent.parent.as_posix()
    else:
        filepath = os.path.join(settings.workdir, 'pbg', dirname)

    pbg_params['checkpoint_path'] = os.path.join(filepath, output)
    settings.pbg_params['checkpoint_path'] = pbg_params['checkpoint_path']

    if auto_wd:
        # empirical numbers from simulation experiments
        if settings.graph_stats[
                os.path.basename(filepath)]['n_edges'] < 5e7:
            # optimial wd (0.013) for sample size (2725781)
            wd = np.around(
                0.013 * 2725781 / settings.graph_stats[
                    os.path.basename(filepath)]['n_edges'],
                decimals=6)
        else:
            # optimial wd (0.0004) for sample size (59103481)
            wd = np.around(
                0.0004 * 59103481 / settings.graph_stats[
                    os.path.basename(filepath)]['n_edges'],
                decimals=6)
        print(f'Auto-estimated weight decay is {wd}')
        pbg_params['wd'] = wd
        if save_wd:
            settings.pbg_params['wd'] = pbg_params['wd']
            print(f"`.settings.pbg_params['wd']` has been updated to {wd}")

    # to avoid oversubscription issues in workloads
    # that involve nested parallelism
    os.environ["OMP_NUM_THREADS"] = "1"

    loader = ConfigFileLoader()
    config = loader.load_config_simba(pbg_params)
    set_logging_verbosity(config.verbose)

    list_filenames = [os.path.join(filepath, "pbg_graph.txt")]
    input_edge_paths = [Path(name) for name in list_filenames]
    print("Converting input data ...")
    if use_edge_weights:
        print("Edge weights are being used ...")
        convert_input_data(
            config.entities,
            config.relations,
            config.entity_path,
            config.edge_paths,
            input_edge_paths,
            TSVEdgelistReader(lhs_col=0, rhs_col=2, rel_col=1, weight_col=3),
            dynamic_relations=config.dynamic_relations,
            )
    else:
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
    print("Starting training ...")
    train(train_config, subprocess_init=subprocess_init)
    print("Finished")
