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
This branch supports marker significance calculation and continuous edge weights. For the relevant documentation, see the updated tutorial. 

To install simba from this branch:
```
conda install --channel conda-forge --channel bioconda pybedtools
pip install 'simba @ git+https://github.com/pinellolab/simba@dev'
```

Make sure you don't have existing `torchbiggraph` installation. 
We recommend setting up the clean new environment:
```
conda create -n simba_dev python
conda config --add channels bioconda
conda config --add channels conda-forge
conda install --channel conda-forge --channel bioconda pybedtools
pip install 'simba @ git+https://github.com/pinellolab/simba@dev'
```

## Documentation
Documentation for the continuous edge weight can be found [here](https://github.com/huidongchen/simba_tutorials/tree/main/v1.2). 
Documentation for the marker feature significance calculation can be found [here](https://github.com/huidongchen/simba_tutorials/tree/dev/v1.1sig).
