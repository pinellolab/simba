import simba as si
import pytest


@pytest.fixture
def adata_CG():
    return si.read_h5ad(
        "tests/data/preprocessed/rna_preprocessed.h5ad")


@pytest.fixture
def adata_CP():
    return si.read_h5ad(
        "tests/data/preprocessed/atac_preprocessed.h5ad")


def test_pbg_training_rna(adata_CG, tmp_path):
    si.settings.set_workdir(tmp_path / "simba_rna")
    si.tl.gen_graph(list_CG=[adata_CG],
                    copy=False,
                    dirname='graph0')
    si.tl.pbg_train(auto_wd=True,
                    output='model')
    si.pl.pbg_metrics(fig_ncol=1,
                      save_fig=True)

def test_pbg_training_rna_significance(adata_CG, tmp_path):
    si.settings.set_workdir(tmp_path / "simba_rna_sig")
    si.tl.gen_graph(list_CG=[adata_CG],
                    copy=False,
                    get_marker_significance=True,
                    dirname='graph0')
    si.tl.pbg_train(auto_wd=True,
                    output='model')
    si.pl.pbg_metrics(fig_ncol=1,
                      save_fig=True)
    dict_adata = si.read_embedding(get_marker_significance=True)
    adata_C = dict_adata['C'] 
    adata_G = dict_adata['G'] 
    adata_nG = dict_adata['nG']
    adata_cmp = si.tl.compare_entities(adata_ref=adata_C, adata_query=adata_G, adata_query_null=adata_nG)
    assert "max_fdr" in adata_cmp.var.columns
    assert ((adata_cmp.var["max_fdr"] <= 1.0) & (adata_cmp.var["max_fdr"] >= 0.0)).all()
    si.pl.entity_metrics(adata_cmp,
        x='max', y='gini', cutoff_fdr=0.05, color_by_fdr='gini', save_fig=True, fig_path='.')

def test_pbg_training_atac(adata_CP, tmp_path):
    si.settings.set_workdir(tmp_path / "simba_atac")
    si.tl.gen_graph(list_CP=[adata_CP],
                    copy=False,
                    dirname='graph0')
    si.tl.pbg_train(auto_wd=True,
                    output='model')
    si.pl.pbg_metrics(fig_ncol=1,
                      save_fig=True)
