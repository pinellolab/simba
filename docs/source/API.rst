.. automodule:: simba

API
===

Import simba as::

   import simba as si

.. Configuration for SIMBA
.. ~~~~~~~~~~~~~~~~~~~~~~~
.. .. autoclass:: settings
..     :members:
..     :undoc-members:
..     :show-inheritance:

Reading
~~~~~~~

.. autosummary::
   :toctree: _autosummary

   read_embedding
   load_pbg_config
   load_graph_stats


Preprocessing
~~~~~~~~~~~~~

Preprocessing functions

.. autosummary::
   :toctree: _autosummary

   pp.log_transform
   pp.normalize

Tools
~~~~~

.. autosummary::
   :toctree: _autosummary

   tl.gen_graph