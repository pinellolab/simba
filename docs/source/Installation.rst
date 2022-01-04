Installation
============

Anaconda
~~~~~~~~

To install the `simba <https://anaconda.org/bioconda/simba>`_ package with conda, run::

    conda install -c bioconda simba

**Recommended**: install *simba* in a new vitural enviroment::

    conda create -n env_simba python simba
    conda activate env_simba


Dev version
~~~~~~~~~~~

To install the latest version on `GitHub <https://github.com/pinellolab/simba>`_, 

first install `simba_pbg <https://anaconda.org/bioconda/simba_pbg>`_ ::

    conda install -c bioconda simba_pbg


then run::

    git clone https://github.com/pinellolab/simba.git
    pip install simba --user

or::

    pip install git+https://github.com/pinellolab/simba
