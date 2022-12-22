[![documentation](https://readthedocs.org/projects/simba-bio/badge/?version=latest)](https://simba-bio.readthedocs.io/en/latest/)
[![CI](https://github.com/pinellolab/simba/actions/workflows/CI.yml/badge.svg)](https://github.com/pinellolab/simba/actions/workflows/CI.yml)
[![Install with conda](https://anaconda.org/bioconda/simba/badges/version.svg)](https://anaconda.org/bioconda/simba)
[![codecov](https://codecov.io/gh/pinellolab/simba/branch/master/graph/badge.svg?token=NDQJQPL18K)](https://codecov.io/gh/pinellolab/simba)

# SIMBA

SIMBA: **SI**ngle-cell e**MB**edding **A**long with features

Website: https://simba-bio.readthedocs.io

Preprint: Huidong Chen, Jayoung Ryu, Michael E. Vinyard, Adam Lerer & Luca Pinello. ["SIMBA: SIngle-cell eMBedding Along with features. *bioRxiv, 2021.10.17.464750v1* (2021)."](https://www.biorxiv.org/content/10.1101/2021.10.17.464750v1)

<img src="./docs/source/_static/img/logo_simba.png?raw=true" width="450">

## Installation
Before installing SIMBA make sure to have the correct channels priority by executing these commands:
```
conda config --add channels defaults
conda config --add channels bioconda
conda config --add channels conda-forge
conda config --set channel_priority strict
```

To install the simba package with conda, run:
```
conda create -n env_simba jupyter simba
```

To enable the k-mer and TF analyses please install these additional dependencies(optional):
```
conda install r-essentials r-optparse bioconductor-jaspar2020 bioconductor-biostrings bioconductor-tfbstools bioconductor-motifmatchr bioconductor-summarizedexperiment r-doparallel bioconductor-rhdf5 bioconductor-hdf5array
```

## [SIMBA v1.2 (dev)](https://github.com/pinellolab/simba/tree/dev) update
We have added the support for
* Continuous edge weight encoding for scRNA-seq ([tutorial](https://github.com/pinellolab/simba_tutorials/blob/main/v1.2/rna_10xpmbc_edgeweigts.ipynb))
* Significance testing of features' cell type specificity metrics ([tutorial](https://github.com/pinellolab/simba_tutorials/tree/main/v1.1sig))

### SIMBA v1.2 Installation
To install simba from this branch:
```
conda create -n env_simba_dev jupyter pytorch pybedtools -y
pip install 'simba @ git+https://github.com/pinellolab/simba@dev'
```
To enable the k-mer and TF analyses please install these additional dependencies(optional):
```
conda install r-essentials r-optparse bioconductor-jaspar2020 bioconductor-biostrings bioconductor-tfbstools bioconductor-motifmatchr bioconductor-summarizedexperiment r-doparallel bioconductor-rhdf5 bioconductor-hdf5array
```
