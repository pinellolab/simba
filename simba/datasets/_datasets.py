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
    if(not os.path.exists(filepath)):
        print('Downloading data ...')
        os.makedirs(filepath)
        download_url(url,
                     os.path.join(filepath, filename),
                     desc=filename)
    adata = read_h5ad(os.path.join(filepath, filename))
    return adata
