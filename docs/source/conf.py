# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../simba'))
import simba


# -- Project information -----------------------------------------------------

project = 'SIMBA'
copyright = '2021, Huidong Chen'
author = 'Huidong Chen'

# The full version, including alpha/beta/rc tags
release = simba.__version__


# -- Retrieve notebooks (borrowed from scVelo) -------------------------------

# from urllib.request import urlretrieve

# notebooks_url = "https://github.com/huidongchen/simba_tutorials/raw/main/"
# notebooks = [
#     "rna_10xpmbc_all_genes",
#     "atac_buenrostro2018_peaks_and_sequences",
#     "multiome_shareseq",
#     "multiome_shareseq_GRN",
#     "rna_mouse_atlas",
#     "rna_human_pancreas",
#     "multiome_10xpmbc10k_integration",
# ]
# for nb in notebooks:
#     try:
#         urlretrieve(notebooks_url + nb, nb)
#     except Exception as e:
#         pass


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

needs_sphinx = "3.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    'sphinx.ext.napoleon',
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "nbsphinx"]

autosummary_generate = True

# Napoleon settings
napoleon_google_docstring = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    "navigation_depth": 1,
    "titles_only": True,
    'logo_only': True,
}
html_show_sphinx = False
html_logo = '_static/img/logo_simba.png'
html_favicon = '_static/img/lion_icon.svg'
html_context = dict(
    display_github=True,
    github_user='pinellolab',
    github_repo='simba',
    github_version='simba_v0.1',
    conf_py_path='/docs/source/',
)


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".

html_static_path = ['_static']
