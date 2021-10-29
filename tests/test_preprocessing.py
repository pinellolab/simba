import simba as si
import pytest


@pytest.fixture
def adata_CG():
    return si.read_h5ad("tests/data/10xpbmc_rna_subset.h5ad")


@pytest.fixture
def adata_CP():
    return si.read_h5ad("tests/data/10xpbmc_atac_subset.h5ad")


def test_rna(adata_CG, tmp_path):
    si.settings.set_workdir(tmp_path / "simba_rna")
    si.settings.set_figure_params(dpi=80,
                                  style='white',
                                  fig_size=[5, 5],
                                  rc={'image.cmap': 'viridis'})
    si.pp.filter_genes(adata_CG, min_n_cells=3)
    si.pp.cal_qc_rna(adata_CG)
    si.pl.violin(adata_CG,
                 list_obs=['n_counts', 'n_genes', 'pct_mt'],
                 save_fig=True,
                 fig_name='plot_violin.png')
    si.pp.filter_cells_rna(adata_CG, min_n_genes=2)
    si.pp.normalize(adata_CG, method='lib_size')
    si.pp.log_transform(adata_CG)
    si.pp.select_variable_genes(adata_CG, n_top_genes=2000)
    si.pl.variable_genes(adata_CG,
                         show_texts=True,
                         save_fig=True,
                         fig_name='plot_variable_genes.png')
    si.tl.discretize(adata_CG, n_bins=5)
    si.pl.discretize(adata_CG,
                     save_fig=True,
                     fig_name='plot_discretize.png')


def test_atac(adata_CP):
    si.pp.filter_peaks(adata_CP, min_n_cells=5)
    si.pp.cal_qc_atac(adata_CP)
    si.pl.hist(adata_CP,
               list_obs=['n_counts', 'n_peaks', 'pct_peaks'],
               log=True,
               list_var=['n_cells'],
               fig_size=(3, 3),
               save_fig=True,
               fig_name='plot_histogram.png')
    si.pp.filter_cells_atac(adata_CP, min_n_peaks=5)
    si.pp.pca(adata_CP, n_components=30)
    si.pl.pca_variance_ratio(adata_CP,
                             show_cutoff=True,
                             save_fig=True,
                             fig_name='plot_variance_ratio.png')
    si.pp.select_pcs(adata_CP, n_pcs=10)
    si.pp.select_pcs_features(adata_CP)
    si.pl.pcs_features(adata_CP,
                       fig_ncol=5,
                       save_fig=True,
                       fig_name='plot_pcs_features.png')
