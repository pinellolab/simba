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
    """10X PBMCs single cell RNA-seq data

    Returns
    -------
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
    adata = read_h5ad(fullpath)
    return adata


def atac_buenrostro2018():
    """single cell ATAC-seq data

    ref: Buenrostro, J.D. et al. Integrated Single-Cell Analysis Maps the
    Continuous RegulatoryLandscape of Human Hematopoietic Differentiation.
    Cell 173, 1535-1548 e1516 (2018).

    Returns
    -------
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
    adata = read_h5ad(fullpath)
    return adata


def multiome_ma2020_fig4():
    """single cell multiome data (SHARE-seq)

    ref: Ma, S. et al. Chromatin Potential Identified by Shared Single-Cell
    Profiling of RNA and Chromatin. Cell (2020).

    Returns
    -------
    dict_adata: `dict`
        A dictionary of anndata objects
    """

    url_rna = 'https://www.dropbox.com/s/7px0meac4vcg5iv/rna_seq_fig4.h5ad?dl=1'
    # url_atac = 'https://www.dropbox.com/s/utrp4fmgapz5yty/atac_seq_fig4.h5ad?dl=1'
    url_atac = 'https://www.dropbox.com/s/087wuliddmbp3oe/rna_seq.h5ad?dl=1'
    filename_rna = 'multiome_ma2020_fig4_rna.h5ad'
    filename_atac = 'multiome_ma2020_fig4_atac.h5ad'
    # filepath = os.path.join(settings.workdir, 'data')
    filepath = './tmp'
    fullpath_rna = os.path.join(filepath, filename_rna)
    fullpath_atac = os.path.join(filepath, filename_atac)

    if(not os.path.exists(fullpath_rna)):
        print('Downloading data ...')
        os.makedirs(filepath, exist_ok=True)
        download_url(url_rna,
                     fullpath_rna,
                     desc=filename_rna)
    if(not os.path.exists(fullpath_atac)):
        print('Downloading data ...')
        os.makedirs(filepath, exist_ok=True)
        download_url(url_atac,
                     fullpath_atac,
                     desc=filename_atac)
    adata_rna = read_h5ad(fullpath_rna)
    adata_atac = read_h5ad(fullpath_atac)
    dict_adata = {'rna': adata_rna,
                  'atac': adata_atac}
    return dict_adata
