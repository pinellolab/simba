"""reading and writing"""

import os
import pandas as pd
import json
from anndata import (
    AnnData,
    read_h5ad,
    read_csv,
    read_excel,
    read_hdf,
    read_loom,
    read_mtx,
    read_text,
    read_umi_tools,
    read_zarr,
)
from pathlib import Path
import tables

from ._settings import settings
from ._utils import _read_legacy_10x_h5, _read_v3_10x_h5


def read_embedding(path_emb=None,
                   path_entity=None,
                   convert_alias=True,
                   path_entity_alias=None,
                   prefix=None,
                   num_epochs=None,
                   get_marker_significance=False,
                   path_entity_alias_marker=None):
    """Read in entity embeddings from pbg training

    Parameters
    ----------
    path_emb: `str`, optional (default: None)
        Path to directory for pbg embedding model
        If None, .settings.pbg_params['checkpoint_path'] will be used.
    path_entity: `str`, optional (default: None)
        Path to entity name file
    prefix: `list`, optional (default: None)
        A list of entity type prefixes to include.
        By default, it reads in the embeddings of all entities.
    convert_alias: `bool`, optional (default: True)
        If True, it will convert entity aliases to the original indices
    path_entity_alias: `str`, optional (default: None)
        Path to entity alias file
    num_epochs: `int`, optional (default: None)
        The embedding result associated with num_epochs to read in

    Returns
    -------
    dict_adata: `dict`
        A dictionary of anndata objects of shape
        (#entities x #dimensions)
    """
    pbg_params = settings.pbg_params
    if path_emb is None:
        path_emb = pbg_params['checkpoint_path']
    if path_entity is None:
        path_entity = pbg_params['entity_path']
    if num_epochs is None:
        num_epochs = pbg_params["num_epochs"]
    if prefix is None:
        prefix = []
    assert isinstance(prefix, list), \
        "`prefix` must be list"
    if convert_alias:
        if path_entity_alias is None:
            path_entity_alias = Path(path_emb).parent.as_posix()
        df_entity_alias = pd.read_csv(
            os.path.join(path_entity_alias, 'entity_alias.txt'),
            header=0,
            index_col=0,
            sep='\t')
        df_entity_alias['id'] = df_entity_alias.index
        df_entity_alias.index = df_entity_alias['alias'].values

    if get_marker_significance:
        path_emb_marker = os.path.join(Path(path_emb).parent.as_posix() + "_with_sig", os.path.basename(path_emb))
        path_entity_marker = os.path.join(Path(path_entity).parent.parent.as_posix() + "_with_sig", 'input/entity')
        if path_entity_alias_marker is None:
            path_entity_alias = Path(path_emb).parent.as_posix() + "_with_sig"
        dict_adata_with_sig = read_embedding(
            path_emb=path_emb_marker,
            path_entity=path_entity_marker,
            convert_alias=True,
            path_entity_alias=path_entity_alias_marker,
            prefix=prefix,
            num_epochs=num_epochs,
            get_marker_significance=False
        )

    dict_adata = dict()
    for x in os.listdir(path_emb):
        if x.startswith('embeddings'):
            entity_type = x.split('_')[1]
            if (len(prefix) == 0) or (entity_type in prefix):
                adata = \
                    read_hdf(os.path.join(path_emb,
                                          f'embeddings_{entity_type}_0.'
                                          f'v{num_epochs}.h5'),
                             key="embeddings")
                with open(
                    os.path.join(path_entity,
                                 f'entity_names_{entity_type}_0.json'), "rt")\
                        as tf:
                    names_entity = json.load(tf)
                if convert_alias:
                    names_entity = \
                        df_entity_alias.loc[names_entity, 'id'].tolist()
                adata.obs.index = names_entity
                dict_adata[entity_type] = adata
                if get_marker_significance:
                    try:
                        dict_adata[f"n{entity_type}"] = dict_adata_with_sig[f"n{entity_type}"]
                    except KeyError:
                        print(f"Null feature nodes for entity {entity_type} not embedded.")

    return dict_adata


# modifed from
# scanpy https://github.com/theislab/scanpy/blob/master/scanpy/readwrite.py
def read_10x_h5(filename,
                genome=None,
                gex_only=True):
    """Read 10x-Genomics-formatted hdf5 file.

    Parameters
    ----------
    filename
        Path to a 10x hdf5 file.
    genome
        Filter expression to genes within this genome. For legacy 10x h5
        files, this must be provided if the data contains more than one genome.
    gex_only
        Only keep 'Gene Expression' data and ignore other feature types,
        e.g. 'Antibody Capture', 'CRISPR Guide Capture', or 'Custom'

    Returns
    -------
    adata: AnnData
        Annotated data matrix, where observations/cells are named by their
        barcode and variables/genes by gene name
    """
    with tables.open_file(str(filename), 'r') as f:
        v3 = '/matrix' in f
    if v3:
        adata = _read_v3_10x_h5(filename)
        if genome:
            if genome not in adata.var['genome'].values:
                raise ValueError(
                    f"Could not find data corresponding to "
                    f"genome '{genome}' in '{filename}'. "
                    f'Available genomes are:'
                    f' {list(adata.var["genome"].unique())}.'
                )
            adata = adata[:, adata.var['genome'] == genome]
        if gex_only:
            adata = adata[:, adata.var['feature_types'] == 'Gene Expression']
        if adata.is_view:
            adata = adata.copy()
    else:
        adata = _read_legacy_10x_h5(filename, genome=genome)
    return adata


def load_pbg_config(path=None):
    """Load PBG configuration into global setting

    Parameters
    ----------
    path: `str`, optional (default: None)
        Path to the directory for pbg configuration file
        If None, `.settings.pbg_params['checkpoint_path']` will be used

    Returns
    -------
    Updates `.settings.pbg_params`

    """
    if path is None:
        path = settings.pbg_params['checkpoint_path']
    path = os.path.normpath(path)
    with open(os.path.join(path, 'config.json'), "rt") as tf:
        pbg_params = json.load(tf)
    settings.set_pbg_params(config=pbg_params)


def load_graph_stats(path=None):
    """Load graph statistics into global setting

    Parameters
    ----------
    path: `str`, optional (default: None)
        Path to the directory for graph statistics file
        If None, `.settings.pbg_params['checkpoint_path']` will be used

    Returns
    -------
    Updates `.settings.graph_stats`
    """
    if path is None:
        path = \
            Path(settings.pbg_params['entity_path']).parent.parent.as_posix()
    path = os.path.normpath(path)
    with open(os.path.join(path, 'graph_stats.json'), "rt") as tf:
        dict_graph_stats = json.load(tf)
    dirname = os.path.basename(path)
    settings.graph_stats[dirname] = dict_graph_stats.copy()


def write_bed(adata,
              use_top_pcs=True,
              filename=None
              ):
    """Write peaks into .bed file

    Parameters
    ----------
    adata: AnnData
        Annotated data matrix with peaks as variables.
    use_top_pcs: `bool`, optional (default: True)
        Use top-PCs-associated features
    filename: `str`, optional (default: None)
        Filename name for peaks.
        By default, a file named 'peaks.bed' will be written to
        `.settings.workdir`
    """
    if filename is None:
        filename = os.path.join(settings.workdir, 'peaks.bed')
    for x in ['chr', 'start', 'end']:
        if x not in adata.var_keys():
            raise ValueError(f"could not find {x} in `adata.var_keys()`")
    if use_top_pcs:
        assert 'top_pcs' in adata.var_keys(), \
            "please run `si.pp.select_pcs_features()` first"
        peaks_selected = adata.var[
            adata.var['top_pcs']][['chr', 'start', 'end']]
    else:
        peaks_selected = adata.var[
            ['chr', 'start', 'end']]
    peaks_selected.to_csv(filename,
                          sep='\t',
                          header=False,
                          index=False)
    fp, fn = os.path.split(filename)
    print(f'"{fn}" was written to "{fp}".')
