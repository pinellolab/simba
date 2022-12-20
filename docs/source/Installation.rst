Installation
============

Anaconda
~~~~~~~~

To install the `simba <https://anaconda.org/bioconda/simba>`_ package with conda, run::

    conda install -c bioconda simba

**Recommended**: install *simba* in a new virtual enviroment::

    conda create -n env_simba python simba
    conda activate env_simba
    
    conda config --add channels defaults
    conda config --add channels bioconda
    conda config --add channels conda-forge
    conda config --set channel_priority strict


Dev version
~~~~~~~~~~~

To install the development version on `GitHub <https://github.com/pinellolab/simba/tree/dev>`_, run following on top of the stable installation::
    
    pip install 'simba @ git+https://github.com/pinellolab/simba@dev'
    
