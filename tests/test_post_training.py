import simba as si
import pytest


@pytest.fixture
def dict_adata():

    return si.read_embedding(
        path_emb='tests/data/pbg_training/model/',
        path_entity='tests/data/pbg_training/input/entity/',
        path_entity_alias='tests/data/pbg_training')


def test_embeddding_rna(dict_adata, tmp_path):
    si.settings.set_workdir(tmp_path / "simba_rna")
    adata_C = dict_adata['C']
    adata_G = dict_adata['G']
    adata_all_CG = si.tl.embed(
        adata_ref=adata_C,
        list_adata_query=[adata_G])
    # add annotations of cells and genes
    adata_all_CG.obs['entity_anno'] = ""
    adata_all_CG.obs.loc[adata_C.obs_names, 'entity_anno'] = 'cell'
    adata_all_CG.obs.loc[adata_G.obs_names, 'entity_anno'] = 'gene'

    si.tl.umap(adata_all_CG,
               n_neighbors=15,
               n_components=2)
    adata_cmp = si.tl.compare_entities(
        adata_ref=adata_C,
        adata_query=adata_G)
    si.pl.entity_metrics(adata_cmp,
                         x='max',
                         y='gini',
                         show_contour=False,
                         texts=adata_G.obs_names[:2],
                         show_texts=True,
                         show_cutoff=True,
                         size=5,
                         text_expand=(1.3, 1.5),
                         cutoff_x=1.,
                         cutoff_y=0.3,
                         save_fig=True)
    si.pl.entity_barcode(adata_cmp,
                         layer='softmax',
                         entities=list(adata_G.obs_names[:2]),
                         show_cutoff=True,
                         cutoff=0.001,
                         fig_size=(5, 2.5),
                         save_fig=True)
    query_result = si.tl.query(adata_all_CG,
                               entity=list(adata_C.obs_names[:2]),
                               obsm='X_umap',
                               use_radius=False,
                               k=50,
                               anno_filter='entity_anno',
                               filters=['gene'])
    print(query_result.head())
    si.pl.query(adata_all_CG,
                show_texts=False,
                color=['entity_anno'],
                alpha=0.9,
                alpha_bg=0.1,
                save_fig=True)
