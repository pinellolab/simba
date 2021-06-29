import urllib.request
from tqdm import tqdm
import os

from .._settings import settings
from ..readwrite import read_h5ad


class DownloadProgressBar(tqdm):
    def update_to(self,
                  b=1,
                  bsize=1,
                  tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url,
                 output_path,
                 desc=None):
    if desc is None:
        desc = url.split('/')[-1]
    with DownloadProgressBar(
        unit='B',
        unit_scale=True,
        miniters=1,
        desc=desc
    ) as t:
        urllib.request.urlretrieve(
            url,
            filename=output_path,
            reporthook=t.update_to)


def rna_10xpmbc3k():
    """10X human peripheral blood mononuclear cells (PBMCs) scRNA-seq data

    Returns
    -------
    adata: `AnnData`
        Anndata object
    """
    url = 'https://www.dropbox.com/s/087wuliddmbp3oe/rna_seq.h5ad?dl=1'
    filename = 'rna_10xpmbc3k.h5ad'
    filepath = os.path.join(settings.workdir, 'data')
    fullpath = os.path.join(filepath, filename)
    if(not os.path.exists(fullpath)):
        print('Downloading data ...')
        os.makedirs(filepath, exist_ok=True)
        download_url(url,
                     fullpath,
                     desc=filename)
        print(f'Downloaded to {filepath}.')
    adata = read_h5ad(fullpath)
    return adata


def rna_han2018():
    """single-cell microwell-seq mouse cell atlas data

    ref: Han, X. et al. Mapping the mouse cell atlas by microwell-seq.
    Cell 172, 1091-1107. e1017 (2018).

    Returns
    -------
    adata: `AnnData`
        Anndata object
    """
    url = 'https://www.dropbox.com/s/nxbszjbir44g99n/rna_seq_mi.h5ad?dl=1'
    filename = 'rna_han2018.h5ad'
    filepath = os.path.join(settings.workdir, 'data')
    fullpath = os.path.join(filepath, filename)
    if(not os.path.exists(fullpath)):
        print('Downloading data ...')
        os.makedirs(filepath, exist_ok=True)
        download_url(url,
                     fullpath,
                     desc=filename)
        print(f'Downloaded to {filepath}.')
    adata = read_h5ad(fullpath)
    return adata


def rna_tmc2018():
    """single-cell Smart-Seq2 mouse cell atlas data

    ref: Tabula Muris Consortium. Single-cell transcriptomics of 20 mouse
    organs creates a Tabula Muris. Nature 562, 367-372 (2018).

    Returns
    -------
    adata: `AnnData`
        Anndata object
    """
    url = 'https://www.dropbox.com/s/rnpyp6vfpuiptkz/rna_seq_sm.h5ad?dl=1'
    filename = 'rna_tmc2018.h5ad'
    filepath = os.path.join(settings.workdir, 'data')
    fullpath = os.path.join(filepath, filename)
    if(not os.path.exists(fullpath)):
        print('Downloading data ...')
        os.makedirs(filepath, exist_ok=True)
        download_url(url,
                     fullpath,
                     desc=filename)
        print(f'Downloaded to {filepath}.')
    adata = read_h5ad(fullpath)
    return adata


def rna_baron2016():
    """single-cell RNA-seq human pancreas data

    ref: Baron, M. et al. A single-cell transcriptomic map of the human and
    mouse pancreas reveals inter-and intra-cell population structure. Cell
    systems 3, 346-360. e344 (2016)

    Returns
    -------
    adata: `AnnData`
        Anndata object
    """
    url = 'https://www.dropbox.com/s/bvziclu6d3fdzow/rna_seq_baron.h5ad?dl=1'
    filename = 'rna_baron2016.h5ad'
    filepath = os.path.join(settings.workdir, 'data')
    fullpath = os.path.join(filepath, filename)
    if(not os.path.exists(fullpath)):
        print('Downloading data ...')
        os.makedirs(filepath, exist_ok=True)
        download_url(url,
                     fullpath,
                     desc=filename)
        print(f'Downloaded to {filepath}.')
    adata = read_h5ad(fullpath)
    return adata


def rna_muraro2016():
    """single-cell RNA-seq human pancreas data

    ref: Muraro, M.J. et al. A single-cell transcriptome atlas of the
    human pancreas.Cell systems 3, 385-394. e383 (2016).

    Returns
    -------
    adata: `AnnData`
        Anndata object
    """
    url = 'https://www.dropbox.com/s/ginc9rbo4qmobwx/rna_seq_muraro.h5ad?dl=1'
    filename = 'rna_muraro2016.h5ad'
    filepath = os.path.join(settings.workdir, 'data')
    fullpath = os.path.join(filepath, filename)
    if(not os.path.exists(fullpath)):
        print('Downloading data ...')
        os.makedirs(filepath, exist_ok=True)
        download_url(url,
                     fullpath,
                     desc=filename)
        print(f'Downloaded to {filepath}.')
    adata = read_h5ad(fullpath)
    return adata


def rna_segerstolpe2016():
    """single-cell RNA-seq human pancreas data

    ref: Segerstolpe, Å. et al. Single-cell transcriptome profiling of human
    pancreatic islets in health and type 2 diabetes.
    Cell metabolism 24, 593-607 (2016).

    Returns
    -------
    adata: `AnnData`
        Anndata object
    """
    url = 'https://www.dropbox.com/s/qomnf4860jwm9pd/rna_seq_segerstolpe.h5ad?dl=1'
    filename = 'rna_segerstolpe2016.h5ad'
    filepath = os.path.join(settings.workdir, 'data')
    fullpath = os.path.join(filepath, filename)
    if(not os.path.exists(fullpath)):
        print('Downloading data ...')
        os.makedirs(filepath, exist_ok=True)
        download_url(url,
                     fullpath,
                     desc=filename)
        print(f'Downloaded to {filepath}.')
    adata = read_h5ad(fullpath)
    return adata


def rna_wang2016():
    """single-cell RNA-seq human pancreas data

    ref: Wang, Y.J. et al. Single-cell transcriptomics of the human endocrine
    pancreas. Diabetes 65, 3028-3038 (2016).

    Returns
    -------
    adata: `AnnData`
        Anndata object
    """
    url = 'https://www.dropbox.com/s/9tv44nugwpx9t4c/rna_seq_wang.h5ad?dl=1'
    filename = 'rna_wang2016.h5ad'
    filepath = os.path.join(settings.workdir, 'data')
    fullpath = os.path.join(filepath, filename)
    if(not os.path.exists(fullpath)):
        print('Downloading data ...')
        os.makedirs(filepath, exist_ok=True)
        download_url(url,
                     fullpath,
                     desc=filename)
        print(f'Downloaded to {filepath}.')
    adata = read_h5ad(fullpath)
    return adata


def rna_xin2016():
    """single-cell RNA-seq human pancreas data

    ref: Xin, Y. et al. RNA sequencing of single human islet cells reveals
    type 2 diabetes genes. Cell metabolism 24, 608-615 (2016).

    Returns
    -------
    adata: `AnnData`
        Anndata object
    """
    url = 'https://www.dropbox.com/s/j483i47mxty6rzo/rna_seq_xin.h5ad?dl=1'
    filename = 'rna_xin2016.h5ad'
    filepath = os.path.join(settings.workdir, 'data')
    fullpath = os.path.join(filepath, filename)
    if(not os.path.exists(fullpath)):
        print('Downloading data ...')
        os.makedirs(filepath, exist_ok=True)
        download_url(url,
                     fullpath,
                     desc=filename)
        print(f'Downloaded to {filepath}.')
    adata = read_h5ad(fullpath)
    return adata


def atac_buenrostro2018():
    """single cell ATAC-seq human blood data

    ref: Buenrostro, J.D. et al. Integrated Single-Cell Analysis Maps the
    Continuous RegulatoryLandscape of Human Hematopoietic Differentiation.
    Cell 173, 1535-1548 e1516 (2018).

    Returns
    -------
    adata: `AnnData`
        Anndata object
    """
    url = 'https://www.dropbox.com/s/7hxjqgdxtbna1tm/atac_seq.h5ad?dl=1'
    filename = 'atac_buenrostro2018.h5ad'
    filepath = os.path.join(settings.workdir, 'data')
    fullpath = os.path.join(filepath, filename)
    if(not os.path.exists(fullpath)):
        print('Downloading data ...')
        os.makedirs(filepath, exist_ok=True)
        download_url(url,
                     fullpath,
                     desc=filename)
        print(f'Downloaded to {filepath}.')
    adata = read_h5ad(fullpath)
    return adata


def atac_10xpbmc5k():
    """10X human peripheral blood mononuclear cells (PBMCs) scATAC-seq data

    Returns
    -------
    adata: `AnnData`
        Anndata object
    """
    url = 'https://www.dropbox.com/s/xa8u7rlskc5h7iv/atac_seq.h5ad?dl=1'
    filename = 'atac_10xpbmc5k.h5ad'
    filepath = os.path.join(settings.workdir, 'data')
    fullpath = os.path.join(filepath, filename)
    if(not os.path.exists(fullpath)):
        print('Downloading data ...')
        os.makedirs(filepath, exist_ok=True)
        download_url(url,
                     fullpath,
                     desc=filename)
        print(f'Downloaded to {filepath}.')
    adata = read_h5ad(fullpath)
    return adata


def atac_cusanovich2018_subset():
    """downsampled sci-ATAC-seq mouse tissue data

    ref: Cusanovich, D.A. et al. A Single-Cell Atlas of In Vivo Mammalian
    Chromatin Accessibility. Cell 174, 1309-1324 e1318 (2018).

    Returns
    -------
    adata: `AnnData`
        Anndata object
    """
    url = 'https://www.dropbox.com/s/e8iqwm93m33i5wt/atac_seq.h5ad?dl=1'
    filename = 'atac_cusanovich2018_subset.h5ad'
    filepath = os.path.join(settings.workdir, 'data')
    fullpath = os.path.join(filepath, filename)
    if(not os.path.exists(fullpath)):
        print('Downloading data ...')
        os.makedirs(filepath, exist_ok=True)
        download_url(url,
                     fullpath,
                     desc=filename)
        print(f'Downloaded to {filepath}.')
    adata = read_h5ad(fullpath)
    return adata


def atac_chen2019():
    """simulated scATAC-seq bone marrow data with a noise level of 0.4
    and a coverage of 2500 fragments

    ref: Chen, H. et al. Assessment of computational methods for the analysis
    of single-cell ATAC-seq data. Genome Biology 20, 241 (2019).

    Returns
    -------
    adata: `AnnData`
        Anndata object
    """
    url = 'https://www.dropbox.com/s/fthhh3mz5b39d4y/atac_seq.h5ad?dl=1'
    filename = 'atac_chen2019.h5ad'
    filepath = os.path.join(settings.workdir, 'data')
    fullpath = os.path.join(filepath, filename)
    if(not os.path.exists(fullpath)):
        print('Downloading data ...')
        os.makedirs(filepath, exist_ok=True)
        download_url(url,
                     fullpath,
                     desc=filename)
        print(f'Downloaded to {filepath}.')
    adata = read_h5ad(fullpath)
    return adata


def multiome_ma2020_fig4():
    """single cell multiome mouse skin data (SHARE-seq)

    ref: Ma, S. et al. Chromatin Potential Identified by Shared Single-Cell
    Profiling of RNA and Chromatin. Cell (2020).

    Returns
    -------
    dict_adata: `dict`
        A dictionary of anndata objects
    """
    url_rna = 'https://www.dropbox.com/s/gmmf77l8kzle6o7/rna_seq_fig4.h5ad?dl=1'
    url_atac = 'https://www.dropbox.com/s/ts0v2y2m5fcumcb/atac_seq_fig4.h5ad?dl=1'
    filename_rna = 'multiome_ma2020_fig4_rna.h5ad'
    filename_atac = 'multiome_ma2020_fig4_atac.h5ad'
    filepath = os.path.join(settings.workdir, 'data')
    fullpath_rna = os.path.join(filepath, filename_rna)
    fullpath_atac = os.path.join(filepath, filename_atac)

    if(not os.path.exists(fullpath_rna)):
        print('Downloading data ...')
        os.makedirs(filepath, exist_ok=True)
        download_url(url_rna,
                     fullpath_rna,
                     desc=filename_rna)
        print(f'Downloaded to {filepath}.')
    if(not os.path.exists(fullpath_atac)):
        print('Downloading data ...')
        os.makedirs(filepath, exist_ok=True)
        download_url(url_atac,
                     fullpath_atac,
                     desc=filename_atac)
        print(f'Downloaded to {filepath}.')
    adata_rna = read_h5ad(fullpath_rna)
    adata_atac = read_h5ad(fullpath_atac)
    dict_adata = {'rna': adata_rna,
                  'atac': adata_atac}
    return dict_adata


def multiome_chen2019():
    """single cell multiome neonatal mouse cerebral cortex data (SNARE-seq)

    ref: Chen, S., Lake, B.B. & Zhang, K. High-throughput sequencing of the
    transcriptome and chromatin accessibility in the same cell.
    Nat Biotechnol (2019).

    Returns
    -------
    dict_adata: `dict`
        A dictionary of anndata objects
    """
    url_rna = 'https://www.dropbox.com/s/b1bbcs500q0pigt/rna_seq.h5ad?dl=1'
    url_atac = 'https://www.dropbox.com/s/ljepkfber68pdvc/atac_seq.h5ad?dl=1'
    filename_rna = 'multiome_chen2019_rna.h5ad'
    filename_atac = 'multiome_chen2019_atac.h5ad'
    filepath = os.path.join(settings.workdir, 'data')
    fullpath_rna = os.path.join(filepath, filename_rna)
    fullpath_atac = os.path.join(filepath, filename_atac)

    if(not os.path.exists(fullpath_rna)):
        print('Downloading data ...')
        os.makedirs(filepath, exist_ok=True)
        download_url(url_rna,
                     fullpath_rna,
                     desc=filename_rna)
        print(f'Downloaded to {filepath}.')
    if(not os.path.exists(fullpath_atac)):
        print('Downloading data ...')
        os.makedirs(filepath, exist_ok=True)
        download_url(url_atac,
                     fullpath_atac,
                     desc=filename_atac)
        print(f'Downloaded to {filepath}.')
    adata_rna = read_h5ad(fullpath_rna)
    adata_atac = read_h5ad(fullpath_atac)
    dict_adata = {'rna': adata_rna,
                  'atac': adata_atac}
    return dict_adata


def multiome_10xpbmc10k():
    """single cell 10X human peripheral blood mononuclear cells (PBMCs)
    multiome data

    Returns
    -------
    dict_adata: `dict`
        A dictionary of anndata objects
    """
    url_rna = 'https://www.dropbox.com/s/zwlim6vljnbfp43/rna_seq.h5ad?dl=1'
    url_atac = 'https://www.dropbox.com/s/163msz0k9hkfrt7/atac_seq.h5ad?dl=1'
    filename_rna = 'multiome_10xpbmc10k_rna.h5ad'
    filename_atac = 'multiome_10xpbmc10k_atac.h5ad'
    filepath = os.path.join(settings.workdir, 'data')
    fullpath_rna = os.path.join(filepath, filename_rna)
    fullpath_atac = os.path.join(filepath, filename_atac)

    if(not os.path.exists(fullpath_rna)):
        print('Downloading data ...')
        os.makedirs(filepath, exist_ok=True)
        download_url(url_rna,
                     fullpath_rna,
                     desc=filename_rna)
        print(f'Downloaded to {filepath}.')
    if(not os.path.exists(fullpath_atac)):
        print('Downloading data ...')
        os.makedirs(filepath, exist_ok=True)
        download_url(url_atac,
                     fullpath_atac,
                     desc=filename_atac)
        print(f'Downloaded to {filepath}.')
    adata_rna = read_h5ad(fullpath_rna)
    adata_atac = read_h5ad(fullpath_atac)
    dict_adata = {'rna': adata_rna,
                  'atac': adata_atac}
    return dict_adata
