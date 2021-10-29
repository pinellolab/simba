"""plotting functions"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pandas.core.dtypes.common import is_numeric_dtype
import seaborn as sns
from adjustText import adjust_text
from pandas.api.types import (
    is_string_dtype,
    is_categorical_dtype,
)
from scipy.sparse import find
import warnings
# import plotly.express as px
# import plotly.graph_objects as go


from .._settings import settings
from ._utils import (
    generate_palette
)


def violin(adata,
           list_obs=None,
           list_var=None,
           jitter=0.4,
           size=1,
           log=False,
           pad=1.08,
           w_pad=None,
           h_pad=3,
           fig_size=(3, 3),
           fig_ncol=3,
           save_fig=False,
           fig_path=None,
           fig_name='plot_violin.pdf',
           **kwargs):
    """Violin plot

    Parameters
    ----------
    adata : `Anndata`
        Annotated data matrix.
    list_obs : `list`, optional (default: None)
        A list of observations to plot.
    list_var : `list`, optional (default: None)
        A list of variables to plot.
    jitter : `float`, optional (default: 0.4)
        Amount of jitter to apply.
    size : `int`, optional (default: 1)
        The marker size
    log : `bool`, optional (default: False)
        If True, natural logarithm transformation will be performed.
    pad: `float`, optional (default: 1.08)
        Padding between the figure edge and the edges of subplots,
        as a fraction of the font size.
    h_pad, w_pad: `float`, optional (default: None)
        Padding (height/width) between edges of adjacent subplots,
        as a fraction of the font size. Defaults to pad.
    fig_size: `tuple`, optional (default: (3,3))
        figure size.
    fig_ncol: `int`, optional (default: 3)
        the number of columns of the figure panel
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_path: `str`, optional (default: None)
        If save_fig is True, specify figure path.
    fig_name: `str`, optional (default: 'plot_violin.pdf')
        if `save_fig` is True, specify figure name.
    **kwargs: `dict`, optional
        Other keyword arguments are passed through to ``sns.violinplot``

    Returns
    -------
    None
    """
    if fig_size is None:
        fig_size = mpl.rcParams['figure.figsize']
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, 'figures')
    if list_obs is None:
        list_obs = []
    if list_var is None:
        list_var = []
    for obs in list_obs:
        if(obs not in adata.obs_keys()):
            raise ValueError(f"could not find {obs} in `adata.obs_keys()`")
    for var in list_var:
        if(var not in adata.var_keys()):
            raise ValueError(f"could not find {var} in `adata.var_keys()`")
    if(len(list_obs) > 0):
        df_plot = adata.obs[list_obs].copy()
        if(log):
            df_plot = pd.DataFrame(data=np.log1p(df_plot.values),
                                   index=df_plot.index,
                                   columns=df_plot.columns)
        fig_nrow = int(np.ceil(len(list_obs)/fig_ncol))
        fig = plt.figure(figsize=(fig_size[0]*fig_ncol*1.05,
                         fig_size[1]*fig_nrow))
        for i, obs in enumerate(list_obs):
            ax_i = fig.add_subplot(fig_nrow, fig_ncol, i+1)
            sns.violinplot(ax=ax_i,
                           y=obs,
                           data=df_plot,
                           inner=None,
                           **kwargs)
            sns.stripplot(ax=ax_i,
                          y=obs,
                          data=df_plot,
                          color='black',
                          jitter=jitter,
                          s=size)

            ax_i.set_title(obs)
            ax_i.set_ylabel('')
            ax_i.locator_params(axis='y', nbins=6)
            ax_i.tick_params(axis="y", pad=-2)
            ax_i.spines['right'].set_visible(False)
            ax_i.spines['top'].set_visible(False)
        plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
        if(save_fig):
            if(not os.path.exists(fig_path)):
                os.makedirs(fig_path)
            plt.savefig(os.path.join(fig_path, fig_name),
                        pad_inches=1,
                        bbox_inches='tight')
            plt.close(fig)
    if(len(list_var) > 0):
        df_plot = adata.var[list_var].copy()
        if(log):
            df_plot = pd.DataFrame(data=np.log1p(df_plot.values),
                                   index=df_plot.index,
                                   columns=df_plot.columns)
        fig_nrow = int(np.ceil(len(list_obs)/fig_ncol))
        fig = plt.figure(figsize=(fig_size[0]*fig_ncol*1.05,
                                  fig_size[1]*fig_nrow))
        for i, var in enumerate(list_var):
            ax_i = fig.add_subplot(fig_nrow, fig_ncol, i+1)
            sns.violinplot(ax=ax_i,
                           y=var,
                           data=df_plot,
                           inner=None,
                           **kwargs)
            sns.stripplot(ax=ax_i,
                          y=var,
                          data=df_plot,
                          color='black',
                          jitter=jitter,
                          s=size)

            ax_i.set_title(var)
            ax_i.set_ylabel('')
            ax_i.locator_params(axis='y', nbins=6)
            ax_i.tick_params(axis="y", pad=-2)
            ax_i.spines['right'].set_visible(False)
            ax_i.spines['top'].set_visible(False)
        plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
        if(save_fig):
            if(not os.path.exists(fig_path)):
                os.makedirs(fig_path)
            plt.savefig(os.path.join(fig_path, fig_name),
                        pad_inches=1,
                        bbox_inches='tight')
            plt.close(fig)


def hist(adata,
         list_obs=None,
         list_var=None,
         kde=True,
         log=False,
         pad=1.08,
         w_pad=None,
         h_pad=3,
         fig_size=(3, 3),
         fig_ncol=3,
         save_fig=False,
         fig_path=None,
         fig_name='plot_histogram.pdf',
         **kwargs
         ):
    """histogram plot

    Parameters
    ----------
    adata : `Anndata`
        Annotated data matrix.
    list_obs : `list`, optional (default: None)
        A list of observations to plot.
    list_var : `list`, optional (default: None)
        A list of variables to plot.
    kde : `bool`, optional (default: True)
        If True, compute a kernel density estimate to smooth the distribution
        and show on the plot
    log : `bool`, optional (default: False)
        If True, natural logarithm transformation will be performed.
    pad: `float`, optional (default: 1.08)
        Padding between the figure edge and the edges of subplots,
        as a fraction of the font size.
    h_pad, w_pad: `float`, optional (default: None)
        Padding (height/width) between edges of adjacent subplots,
        as a fraction of the font size. Defaults to pad.
    fig_size: `tuple`, optional (default: (3,3))
        figure size.
    fig_ncol: `int`, optional (default: 3)
        the number of columns of the figure panel
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_path: `str`, optional (default: None)
        If save_fig is True, specify figure path.
    fig_name: `str`, optional (default: 'plot_violin.pdf')
        if `save_fig` is True, specify figure name.
    **kwargs: `dict`, optional
        Other keyword arguments are passed through to ``sns.histplot``

    Returns
    -------
    None
    """
    if fig_size is None:
        fig_size = mpl.rcParams['figure.figsize']
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, 'figures')
    if list_obs is None:
        list_obs = []
    if list_var is None:
        list_var = []
    for obs in list_obs:
        if(obs not in adata.obs_keys()):
            raise ValueError(f"could not find {obs} in `adata.obs_keys()`")
    for var in list_var:
        if(var not in adata.var_keys()):
            raise ValueError(f"could not find {var} in `adata.var_keys()`")

    if(len(list_obs) > 0):
        df_plot = adata.obs[list_obs].copy()
        if(log):
            df_plot = pd.DataFrame(data=np.log1p(df_plot.values),
                                   index=df_plot.index,
                                   columns=df_plot.columns)
        fig_nrow = int(np.ceil(len(list_obs)/fig_ncol))
        fig = plt.figure(figsize=(fig_size[0]*fig_ncol*1.05,
                                  fig_size[1]*fig_nrow))
        for i, obs in enumerate(list_obs):
            ax_i = fig.add_subplot(fig_nrow, fig_ncol, i+1)
            sns.histplot(ax=ax_i,
                         x=obs,
                         data=df_plot,
                         kde=kde,
                         **kwargs)
            ax_i.locator_params(axis='y', nbins=6)
            ax_i.tick_params(axis="y", pad=-2)
            ax_i.spines['right'].set_visible(False)
            ax_i.spines['top'].set_visible(False)
        plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
        if(save_fig):
            if(not os.path.exists(fig_path)):
                os.makedirs(fig_path)
            plt.savefig(os.path.join(fig_path, fig_name),
                        pad_inches=1,
                        bbox_inches='tight')
            plt.close(fig)
    if(len(list_var) > 0):
        df_plot = adata.var[list_var].copy()
        if(log):
            df_plot = pd.DataFrame(data=np.log1p(df_plot.values),
                                   index=df_plot.index,
                                   columns=df_plot.columns)
        fig_nrow = int(np.ceil(len(list_obs)/fig_ncol))
        fig = plt.figure(figsize=(fig_size[0]*fig_ncol*1.05,
                                  fig_size[1]*fig_nrow))
        for i, var in enumerate(list_var):
            ax_i = fig.add_subplot(fig_nrow, fig_ncol, i+1)
            sns.histplot(ax=ax_i,
                         x=var,
                         data=df_plot,
                         kde=kde,
                         **kwargs)
            ax_i.locator_params(axis='y', nbins=6)
            ax_i.tick_params(axis="y", pad=-2)
            ax_i.spines['right'].set_visible(False)
            ax_i.spines['top'].set_visible(False)
        plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
        if(save_fig):
            if(not os.path.exists(fig_path)):
                os.makedirs(fig_path)
            plt.savefig(os.path.join(fig_path, fig_name),
                        pad_inches=1,
                        bbox_inches='tight')
            plt.close(fig)


def pca_variance_ratio(adata,
                       log=True,
                       show_cutoff=True,
                       fig_size=(4, 4),
                       save_fig=None,
                       fig_path=None,
                       fig_name='plot_variance_ratio.pdf',
                       pad=1.08,
                       w_pad=None,
                       h_pad=None,
                       **kwargs):
    """Plot the variance ratio.

    Parameters
    ----------
    adata : `Anndata`
        Annotated data matrix.
    log : `bool`, optional (default: True)
        If True, variance_ratio will be log-transformed.
    show_cutoff : `bool`, optional (default: True)
        If True, cutoff on `n_pcs` will be shown
    pad: `float`, optional (default: 1.08)
        Padding between the figure edge and the edges of subplots,
        as a fraction of the font size.
    h_pad, w_pad: `float`, optional (default: None)
        Padding (height/width) between edges of adjacent subplots,
        as a fraction of the font size. Defaults to pad.
    fig_size: `tuple`, optional (default: (3,3))
        figure size.
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_path: `str`, optional (default: None)
        If save_fig is True, specify figure path.
    fig_name: `str`, optional (default: 'plot_variance_ratio.pdf')
        if `save_fig` is True, specify figure name.
    **kwargs: `dict`, optional
        Other keyword arguments are passed through to ``plt.plot``

    Returns
    -------
    None
    """
    if fig_size is None:
        fig_size = mpl.rcParams['figure.figsize']
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, 'figures')

    n_components = len(adata.uns['pca']['variance_ratio'])

    fig = plt.figure(figsize=fig_size)
    if(log):
        plt.plot(range(n_components),
                 np.log(adata.uns['pca']['variance_ratio']),
                 **kwargs)
    else:
        plt.plot(range(n_components),
                 adata.uns['pca']['variance_ratio'],
                 **kwargs)
    if(show_cutoff):
        n_pcs = adata.uns['pca']['n_pcs']
        print(f'the number of selected PC is: {n_pcs}')
        plt.axvline(n_pcs, ls='--', c='red')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Ratio')
    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    if(save_fig):
        if(not os.path.exists(fig_path)):
            os.makedirs(fig_path)
        plt.savefig(os.path.join(fig_path, fig_name),
                    pad_inches=1,
                    bbox_inches='tight')
        plt.close(fig)


def pcs_features(adata,
                 log=False,
                 size=3,
                 show_cutoff=True,
                 pad=1.08,
                 w_pad=None,
                 h_pad=None,
                 fig_size=(3, 3),
                 fig_ncol=3,
                 save_fig=None,
                 fig_path=None,
                 fig_name='plot_pcs_features.pdf',
                 **kwargs):
    """Plot features that contribute to the top PCs.

    Parameters
    ----------
    adata : `Anndata`
        Annotated data matrix.
    log : `bool`, optional (default: True)
        If True, variance_ratio will be log-transformed.
    show_cutoff : `bool`, optional (default: True)
        If True, cutoff on `n_pcs` will be shown
    size : `int`, optional (default: 3)
        The marker size
    pad: `float`, optional (default: 1.08)
        Padding between the figure edge and the edges of subplots,
        as a fraction of the font size.
    h_pad, w_pad: `float`, optional (default: None)
        Padding (height/width) between edges of adjacent subplots,
        as a fraction of the font size. Defaults to pad.
    fig_size: `tuple`, optional (default: (3,3))
        figure size.
    fig_ncol: `int`, optional (default: 3)
        the number of columns of the figure panel
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_path: `str`, optional (default: None)
        If save_fig is True, specify figure path.
    fig_name: `str`, optional (default: 'plot_pcs_features.pdf')
        if `save_fig` is True, specify figure name.
    **kwargs: `dict`, optional
        Other keyword arguments are passed through to ``plt.scatter``

    Returns
    -------
    None
    """

    if fig_size is None:
        fig_size = mpl.rcParams['figure.figsize']
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, 'figures')

    n_pcs = adata.uns['pca']['n_pcs']
    n_features = adata.uns['pca']['PCs'].shape[0]

    fig_nrow = int(np.ceil(n_pcs/fig_ncol))
    fig = plt.figure(figsize=(fig_size[0]*fig_ncol*1.05, fig_size[1]*fig_nrow))

    for i in range(n_pcs):
        ax_i = fig.add_subplot(fig_nrow, fig_ncol, i+1)
        if(log):
            ax_i.scatter(range(n_features),
                         np.log(np.sort(
                             np.abs(adata.uns['pca']['PCs'][:, i],))[::-1]),
                         s=size,
                         **kwargs)
        else:
            ax_i.scatter(range(n_features),
                         np.sort(
                             np.abs(adata.uns['pca']['PCs'][:, i],))[::-1],
                         s=size,
                         **kwargs)
        n_ft_selected_i = len(adata.uns['pca']['features'][f'pc_{i}'])
        if(show_cutoff):
            ax_i.axvline(n_ft_selected_i, ls='--', c='red')
        ax_i.set_xlabel('Feautures')
        ax_i.set_ylabel('Loadings')
        ax_i.locator_params(axis='x', nbins=3)
        ax_i.locator_params(axis='y', nbins=5)
        ax_i.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax_i.set_title(f'PC {i}')
    plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    if(save_fig):
        if(not os.path.exists(fig_path)):
            os.makedirs(fig_path)
        plt.savefig(os.path.join(fig_path, fig_name),
                    pad_inches=1,
                    bbox_inches='tight')
        plt.close(fig)


def variable_genes(adata,
                   show_texts=False,
                   n_texts=10,
                   size=8,
                   text_size=10,
                   pad=1.08,
                   w_pad=None,
                   h_pad=None,
                   fig_size=(4, 4),
                   save_fig=None,
                   fig_path=None,
                   fig_name='plot_variable_genes.pdf',
                   **kwargs):
    """Plot highly variable genes.

    Parameters
    ----------
    adata : `Anndata`
        Annotated data matrix.
    show_texts : `bool`, optional (default: False)
        If True, text annotation will be shown.
    n_texts : `int`, optional (default: 10)
        The number of texts to plot.
    size : `int`, optional (default: 8)
        The marker size
    text_size : `int`, optional (default: 10)
        The text size
    pad: `float`, optional (default: 1.08)
        Padding between the figure edge and the edges of subplots,
        as a fraction of the font size.
    h_pad, w_pad: `float`, optional (default: None)
        Padding (height/width) between edges of adjacent subplots,
        as a fraction of the font size. Defaults to pad.
    fig_size: `tuple`, optional (default: (3,3))
        figure size.
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_path: `str`, optional (default: None)
        If save_fig is True, specify figure path.
    fig_name: `str`, optional (default: 'plot_variable_genes.pdf')
        if `save_fig` is True, specify figure name.
    **kwargs: `dict`, optional
        Other keyword arguments are passed through to ``plt.scatter``

    Returns
    -------
    None
    """

    if fig_size is None:
        fig_size = mpl.rcParams['figure.figsize']
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, 'figures')

    means = adata.var['means']
    variances_norm = adata.var['variances_norm']
    mask = adata.var['highly_variable']
    genes = adata.var_names

    fig, ax = plt.subplots(figsize=fig_size)
    ax.scatter(means[~mask],
               variances_norm[~mask],
               s=size,
               c='#1F2433',
               **kwargs)
    ax.scatter(means[mask],
               variances_norm[mask],
               s=size,
               c='#ce3746',
               **kwargs)
    ax.set_xscale(value='log')

    if show_texts:
        ids = variances_norm.values.argsort()[-n_texts:][::-1]
        texts = [plt.text(means[i], variances_norm[i], genes[i],
                 fontdict={'family': 'serif',
                           'color': 'black',
                           'weight': 'normal',
                           'size': text_size})
                 for i in ids]
        adjust_text(texts,
                    arrowprops=dict(arrowstyle='-', color='black'))

    ax.set_xlabel('average expression')
    ax.set_ylabel('standardized variance')
    ax.locator_params(axis='x', tight=True)
    ax.locator_params(axis='y', tight=True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    if(save_fig):
        if(not os.path.exists(fig_path)):
            os.makedirs(fig_path)
        fig.savefig(os.path.join(fig_path, fig_name),
                    pad_inches=1,
                    bbox_inches='tight')
        plt.close(fig)


def _scatterplot2d(df,
                   x,
                   y,
                   list_hue=None,
                   hue_palette=None,
                   drawing_order='sorted',
                   dict_drawing_order=None,
                   size=8,
                   show_texts=False,
                   texts=None,
                   text_size=10,
                   text_expand=(1.05, 1.2),
                   fig_size=None,
                   fig_ncol=3,
                   fig_legend_ncol=1,
                   fig_legend_order=None,
                   vmin=None,
                   vmax=None,
                   alpha=0.8,
                   pad=1.08,
                   w_pad=None,
                   h_pad=None,
                   save_fig=None,
                   fig_path=None,
                   fig_name='scatterplot2d.pdf',
                   copy=False,
                   **kwargs):
    """2d scatter plot

    Parameters
    ----------
    data: `pd.DataFrame`
        Input data structure of shape (n_samples, n_features).
    x: `str`
        Variable in `data` that specify positions on the x axis.
    y: `str`
        Variable in `data` that specify positions on the x axis.
    list_hue: `str`, optional (default: None)
        A list of variables that will produce points with different colors.
    drawing_order: `str` (default: 'sorted')
        The order in which values are plotted, This can be
        one of the following values
        - 'original': plot points in the same order as in input dataframe
        - 'sorted' : plot points with higher values on top.
        - 'random' : plot points in a random order
    fig_size: `tuple`, optional (default: None)
        figure size.
    fig_ncol: `int`, optional (default: 3)
        the number of columns of the figure panel
    fig_legend_order: `dict`,optional (default: None)
        Specified order for the appearance of the annotation keys.
        Only valid for categorical/string variable
        e.g. fig_legend_order = {'ann1':['a','b','c'],'ann2':['aa','bb','cc']}
    fig_legend_ncol: `int`, optional (default: 1)
        The number of columns that the legend has.
    vmin,vmax: `float`, optional (default: None)
        The min and max values are used to normalize continuous values.
        If None, the respective min and max of continuous values is used.
    alpha: `float`, optional (default: 0.8)
        0.0 transparent through 1.0 opaque
    pad: `float`, optional (default: 1.08)
        Padding between the figure edge and the edges of subplots,
        as a fraction of the font size.
    h_pad, w_pad: `float`, optional (default: None)
        Padding (height/width) between edges of adjacent subplots,
        as a fraction of the font size. Defaults to pad.
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_path: `str`, optional (default: None)
        If save_fig is True, specify figure path.
    fig_name: `str`, optional (default: 'scatterplot2d.pdf')
        if save_fig is True, specify figure name.
    Returns
    -------
    None
    """

    if fig_size is None:
        fig_size = mpl.rcParams['figure.figsize']
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, 'figures')

    list_ax = list()
    if list_hue is None:
        list_hue = [None]
    else:
        for hue in list_hue:
            if(hue not in df.columns):
                raise ValueError(f"could not find {hue}")
        if hue_palette is None:
            hue_palette = dict()
        assert isinstance(hue_palette, dict), "`hue_palette` must be dict"
        legend_order = {hue: np.unique(df[hue]) for hue in list_hue
                        if (is_string_dtype(df[hue])
                            or is_categorical_dtype(df[hue]))}
        if(fig_legend_order is not None):
            if(not isinstance(fig_legend_order, dict)):
                raise TypeError("`fig_legend_order` must be a dictionary")
            for hue in fig_legend_order.keys():
                if(hue in legend_order.keys()):
                    legend_order[hue] = fig_legend_order[hue]
                else:
                    print(f"{hue} is ignored for ordering legend labels"
                          "due to incorrect name or data type")

    if dict_drawing_order is None:
        dict_drawing_order = dict()
    assert drawing_order in ['sorted', 'random', 'original'],\
        "`drawing_order` must be one of ['original', 'sorted', 'random']"

    if(len(list_hue) < fig_ncol):
        fig_ncol = len(list_hue)
    fig_nrow = int(np.ceil(len(list_hue)/fig_ncol))
    fig = plt.figure(figsize=(fig_size[0]*fig_ncol*1.05, fig_size[1]*fig_nrow))
    for i, hue in enumerate(list_hue):
        ax_i = fig.add_subplot(fig_nrow, fig_ncol, i+1)
        if hue is None:
            sc_i = sns.scatterplot(ax=ax_i,
                                   x=x,
                                   y=y,
                                   data=df,
                                   alpha=alpha,
                                   linewidth=0,
                                   s=size,
                                   **kwargs)
        else:
            if(is_string_dtype(df[hue]) or is_categorical_dtype(df[hue])):
                if hue in hue_palette.keys():
                    palette = hue_palette[hue]
                else:
                    palette = None
                if hue in dict_drawing_order.keys():
                    param_drawing_order = dict_drawing_order[hue]
                else:
                    param_drawing_order = drawing_order
                if param_drawing_order == 'sorted':
                    df_updated = df.sort_values(by=hue)
                elif param_drawing_order == 'random':
                    df_updated = df.sample(frac=1, random_state=100)
                else:
                    df_updated = df
                sc_i = sns.scatterplot(ax=ax_i,
                                       x=x,
                                       y=y,
                                       hue=hue,
                                       hue_order=legend_order[hue],
                                       data=df_updated,
                                       alpha=alpha,
                                       linewidth=0,
                                       palette=palette,
                                       s=size,
                                       **kwargs)
                ax_i.legend(bbox_to_anchor=(1, 0.5),
                            loc='center left',
                            ncol=fig_legend_ncol,
                            frameon=False,
                            )
            else:
                vmin_i = df[hue].min() if vmin is None else vmin
                vmax_i = df[hue].max() if vmax is None else vmax
                if hue in dict_drawing_order.keys():
                    param_drawing_order = dict_drawing_order[hue]
                else:
                    param_drawing_order = drawing_order
                if param_drawing_order == 'sorted':
                    df_updated = df.sort_values(by=hue)
                elif param_drawing_order == 'random':
                    df_updated = df.sample(frac=1, random_state=100)
                else:
                    df_updated = df
                sc_i = ax_i.scatter(df_updated[x],
                                    df_updated[y],
                                    c=df_updated[hue],
                                    vmin=vmin_i,
                                    vmax=vmax_i,
                                    alpha=alpha,
                                    s=size,
                                    **kwargs)
                cbar = plt.colorbar(sc_i,
                                    ax=ax_i,
                                    pad=0.01,
                                    fraction=0.05,
                                    aspect=40)
                cbar.solids.set_edgecolor("face")
                cbar.ax.locator_params(nbins=5)
        if show_texts:
            if texts is not None:
                plt_texts = [plt.text(df[x][t],
                                      df[y][t],
                                      t,
                                      fontdict={'family': 'serif',
                                                'color': 'black',
                                                'weight': 'normal',
                                                'size': text_size})
                             for t in texts]
                adjust_text(plt_texts,
                            expand_text=text_expand,
                            expand_points=text_expand,
                            expand_objects=text_expand,
                            arrowprops=dict(arrowstyle='->', color='black'))
        ax_i.set_xlabel(x)
        ax_i.set_ylabel(y)
        ax_i.locator_params(axis='x', nbins=5)
        ax_i.locator_params(axis='y', nbins=5)
        ax_i.tick_params(axis="both", labelbottom=True, labelleft=True)
        ax_i.set_title(hue)
        list_ax.append(ax_i)
    plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    if save_fig:
        if(not os.path.exists(fig_path)):
            os.makedirs(fig_path)
        plt.savefig(os.path.join(fig_path, fig_name),
                    pad_inches=1,
                    bbox_inches='tight')
        plt.close(fig)
    if copy:
        return list_ax


# def _scatterplot2d_plotly(df,
#                           x,
#                           y,
#                           list_hue=None,
#                           hue_palette=None,
#                           drawing_order='sorted',
#                           fig_size=None,
#                           fig_ncol=3,
#                           fig_legend_order=None,
#                           alpha=0.8,
#                           save_fig=None,
#                           fig_path=None,
#                           **kwargs):
#     """interactive 2d scatter plot by Plotly

#     Parameters
#     ----------
#     data: `pd.DataFrame`
#         Input data structure of shape (n_samples, n_features).
#     x: `str`
#         Variable in `data` that specify positions on the x axis.
#     y: `str`
#         Variable in `data` that specify positions on the x axis.
#     list_hue: `str`, optional (default: None)
#         A list of variables that will produce points with different colors.
#     drawing_order: `str` (default: 'sorted')
#         The order in which values are plotted, This can be
#         one of the following values
#         - 'original': plot points in the same order as in input dataframe
#         - 'sorted' : plot points with higher values on top.
#         - 'random' : plot points in a random order
#     fig_size: `tuple`, optional (default: None)
#         figure size.
#     fig_ncol: `int`, optional (default: 3)
#         the number of columns of the figure panel
#     fig_legend_order: `dict`,optional (default: None)
#         Specified order for the appearance of the annotation keys.
#         Only valid for categorical/string variable
#         e.g. fig_legend_order = {'ann1':['a','b','c'],
#                                  'ann2':['aa','bb','cc']}
#     fig_legend_ncol: `int`, optional (default: 1)
#         The number of columns that the legend has.
#     vmin,vmax: `float`, optional (default: None)
#         The min and max values are used to normalize continuous values.
#         If None, the respective min and max of continuous values is used.
#     alpha: `float`, optional (default: 0.8)
#         0.0 transparent through 1.0 opaque
#     pad: `float`, optional (default: 1.08)
#         Padding between the figure edge and the edges of subplots,
#         as a fraction of the font size.
#     h_pad, w_pad: `float`, optional (default: None)
#         Padding (height/width) between edges of adjacent subplots,
#         as a fraction of the font size. Defaults to pad.
#     save_fig: `bool`, optional (default: False)
#         if True,save the figure.
#     fig_path: `str`, optional (default: None)
#         If save_fig is True, specify figure path.
#     fig_name: `str`, optional (default: 'scatterplot2d.pdf')
#         if save_fig is True, specify figure name.
#     Returns
#     -------
#     None
#     """

#     if fig_size is None:
#         fig_size = mpl.rcParams['figure.figsize']
#     if save_fig is None:
#         save_fig = settings.save_fig
#     if fig_path is None:
#         fig_path = os.path.join(settings.workdir, 'figures')

#     for hue in list_hue:
#         if(hue not in df.columns):
#             raise ValueError(f"could not find {hue} in `df.columns`")
#     if hue_palette is None:
#         hue_palette = dict()
#     assert isinstance(hue_palette, dict), "`hue_palette` must be dict"

#     assert drawing_order in ['sorted', 'random', 'original'],\
#         "`drawing_order` must be one of ['original', 'sorted', 'random']"

#     legend_order = {hue: np.unique(df[hue]) for hue in list_hue
#                     if (is_string_dtype(df[hue])
#                         or is_categorical_dtype(df[hue]))}
#     if(fig_legend_order is not None):
#         if(not isinstance(fig_legend_order, dict)):
#             raise TypeError("`fig_legend_order` must be a dictionary")
#         for hue in fig_legend_order.keys():
#             if(hue in legend_order.keys()):
#                 legend_order[hue] = fig_legend_order[hue]
#             else:
#                 print(f"{hue} is ignored for ordering legend labels"
#                       "due to incorrect name or data type")

#     if(len(list_hue) < fig_ncol):
#         fig_ncol = len(list_hue)
#     fig_nrow = int(np.ceil(len(list_hue)/fig_ncol))
#     fig = plt.figure(figsize=(fig_size[0]*fig_ncol*1.05,
#                      fig_size[1]*fig_nrow))
#     for hue in list_hue:
#         if hue in hue_palette.keys():
#             palette = hue_palette[hue]
#         else:
#             palette = None
#         if drawing_order == 'sorted':
#             df_updated = df.sort_values(by=hue)
#         elif drawing_order == 'random':
#             df_updated = df.sample(frac=1, random_state=100)
#         else:
#             df_updated = df
#         fig = px.scatter(df_updated,
#                          x=x,
#                          y=y,
#                          color=hue,
#                          opacity=alpha,
#                          color_continuous_scale=px.colors.sequential.Viridis,
#                          color_discrete_map=palette,
#                          **kwargs)
#         fig.update_layout(legend={'itemsizing': 'constant'},
#                           width=500,
#                           height=500)
#         fig.show(renderer="notebook")


# TO-DO add 3D plot
def umap(adata,
         color=None,
         dict_palette=None,
         n_components=None,
         size=8,
         drawing_order='sorted',
         dict_drawing_order=None,
         show_texts=False,
         texts=None,
         text_size=10,
         text_expand=(1.05, 1.2),
         fig_size=None,
         fig_ncol=3,
         fig_legend_ncol=1,
         fig_legend_order=None,
         vmin=None,
         vmax=None,
         alpha=1,
         pad=1.08,
         w_pad=None,
         h_pad=None,
         save_fig=None,
         fig_path=None,
         fig_name='plot_umap.pdf',
         plolty=False,
         **kwargs):
    """ Plot coordinates in UMAP

    Parameters
    ----------
    data: `pd.DataFrame`
        Input data structure of shape (n_samples, n_features).
    x: `str`
        Variable in `data` that specify positions on the x axis.
    y: `str`
        Variable in `data` that specify positions on the x axis.
    color: `list`, optional (default: None)
        A list of variables that will produce points with different colors.
        e.g. color = ['anno1', 'anno2']
    dict_palette: `dict`,optional (default: None)
        A dictionary of palettes for different variables in `color`.
        Only valid for categorical/string variables
        e.g. dict_palette = {'ann1': {},'ann2': {}}
    drawing_order: `str` (default: 'sorted')
        The order in which values are plotted, This can be
        one of the following values
        - 'original': plot points in the same order as in input dataframe
        - 'sorted' : plot points with higher values on top.
        - 'random' : plot points in a random order
    dict_drawing_order: `dict`,optional (default: None)
        A dictionary of drawing_order for different variables in `color`.
        Only valid for categorical/string variables
        e.g. dict_drawing_order = {'ann1': 'original','ann2': 'sorted'}
    size: `int` (default: 8)
        Point size.
    show_texts : `bool`, optional (default: False)
        If True, text annotation will be shown.
    text_size : `int`, optional (default: 10)
        The text size.
    texts: `list` optional (default: None)
        Point names to plot.
    text_expand : `tuple`, optional (default: (1.05, 1.2))
        Two multipliers (x, y) by which to expand the bounding box of texts
        when repelling them from each other/points/other objects.
    fig_size: `tuple`, optional (default: None)
        figure size.
    fig_ncol: `int`, optional (default: 3)
        the number of columns of the figure panel
    fig_legend_order: `dict`,optional (default: None)
        Specified order for the appearance of the annotation keys.
        Only valid for categorical/string variable
        e.g. fig_legend_order = {'ann1':['a','b','c'],'ann2':['aa','bb','cc']}
    fig_legend_ncol: `int`, optional (default: 1)
        The number of columns that the legend has.
    vmin,vmax: `float`, optional (default: None)
        The min and max values are used to normalize continuous values.
        If None, the respective min and max of continuous values is used.
    alpha: `float`, optional (default: 0.8)
        0.0 transparent through 1.0 opaque
    pad: `float`, optional (default: 1.08)
        Padding between the figure edge and the edges of subplots,
        as a fraction of the font size.
    h_pad, w_pad: `float`, optional (default: None)
        Padding (height/width) between edges of adjacent subplots,
        as a fraction of the font size. Defaults to pad.
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_path: `str`, optional (default: None)
        If save_fig is True, specify figure path.
    fig_name: `str`, optional (default: 'plot_umap.pdf')
        if save_fig is True, specify figure name.
    Returns
    -------
    None
    """

    if fig_size is None:
        fig_size = mpl.rcParams['figure.figsize']
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, 'figures')

    if(n_components is None):
        n_components = min(3, adata.obsm['X_umap'].shape[1])
    if n_components not in [2, 3]:
        raise ValueError("n_components should be 2 or 3")
    if(n_components > adata.obsm['X_umap'].shape[1]):
        print(f"`n_components` is greater than the available dimension.\n"
              f"It is corrected to {adata.obsm['X_umap'].shape[1]}")
        n_components = adata.obsm['X_umap'].shape[1]

    if dict_palette is None:
        dict_palette = dict()
    df_plot = pd.DataFrame(index=adata.obs.index,
                           data=adata.obsm['X_umap'],
                           columns=['UMAP'+str(x+1) for x in
                                    range(adata.obsm['X_umap'].shape[1])])
    if color is None:
        _scatterplot2d(df_plot,
                       x='UMAP1',
                       y='UMAP2',
                       drawing_order=drawing_order,
                       size=size,
                       show_texts=show_texts,
                       text_size=text_size,
                       texts=texts,
                       text_expand=text_expand,
                       fig_size=fig_size,
                       alpha=alpha,
                       pad=pad,
                       w_pad=w_pad,
                       h_pad=h_pad,
                       save_fig=save_fig,
                       fig_path=fig_path,
                       fig_name=fig_name,
                       **kwargs)
    else:
        color = list(dict.fromkeys(color))  # remove duplicate keys
        for ann in color:
            if(ann in adata.obs_keys()):
                df_plot[ann] = adata.obs[ann]
                if(not is_numeric_dtype(df_plot[ann])):
                    if 'color' not in adata.uns_keys():
                        adata.uns['color'] = dict()

                    if ann not in dict_palette.keys():
                        if (ann+'_color' in adata.uns['color'].keys()) \
                            and \
                            (all(np.isin(np.unique(df_plot[ann]),
                                         list(adata.uns['color']
                                         [ann+'_color'].keys())))):
                            dict_palette[ann] = \
                                adata.uns['color'][ann+'_color']
                        else:
                            dict_palette[ann] = \
                                generate_palette(adata.obs[ann])
                            adata.uns['color'][ann+'_color'] = \
                                dict_palette[ann].copy()
                    else:
                        if ann+'_color' not in adata.uns['color'].keys():
                            adata.uns['color'][ann+'_color'] = \
                                dict_palette[ann].copy()

            elif(ann in adata.var_names):
                df_plot[ann] = adata.obs_vector(ann)
            else:
                raise ValueError(f"could not find {ann} in `adata.obs.columns`"
                                 " and `adata.var_names`")
        if plolty:
            print('Plotly is not supported yet.')
            # _scatterplot2d_plotly(df_plot,
            #                       x='UMAP1',
            #                       y='UMAP2',
            #                       list_hue=color,
            #                       hue_palette=dict_palette,
            #                       drawing_order=drawing_order,
            #                       fig_size=fig_size,
            #                       fig_ncol=fig_ncol,
            #                       fig_legend_order=fig_legend_order,
            #                       alpha=alpha,
            #                       save_fig=save_fig,
            #                       fig_path=fig_path,
            #                       **kwargs)
        else:
            _scatterplot2d(df_plot,
                           x='UMAP1',
                           y='UMAP2',
                           list_hue=color,
                           hue_palette=dict_palette,
                           drawing_order=drawing_order,
                           dict_drawing_order=dict_drawing_order,
                           size=size,
                           show_texts=show_texts,
                           text_size=text_size,
                           text_expand=text_expand,
                           texts=texts,
                           fig_size=fig_size,
                           fig_ncol=fig_ncol,
                           fig_legend_ncol=fig_legend_ncol,
                           fig_legend_order=fig_legend_order,
                           vmin=vmin,
                           vmax=vmax,
                           alpha=alpha,
                           pad=pad,
                           w_pad=w_pad,
                           h_pad=h_pad,
                           save_fig=save_fig,
                           fig_path=fig_path,
                           fig_name=fig_name,
                           **kwargs)


def discretize(adata,
               kde=None,
               fig_size=(6, 6),
               pad=1.08,
               w_pad=None,
               h_pad=None,
               save_fig=None,
               fig_path=None,
               fig_name='plot_discretize.pdf',
               **kwargs):
    """Plot original data VS discretized data

    Parameters
    ----------
    adata : `Anndata`
        Annotated data matrix.
    kde : `bool`, optional (default: None)
        If True, compute a kernel density estimate to smooth the distribution
        and show on the plot. Invalid as of v0.2.
    pad: `float`, optional (default: 1.08)
        Padding between the figure edge and the edges of subplots,
        as a fraction of the font size.
    h_pad, w_pad: `float`, optional (default: None)
        Padding (height/width) between edges of adjacent subplots,
        as a fraction of the font size. Defaults to pad.
    fig_size: `tuple`, optional (default: (5,8))
        figure size.
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_path: `str`, optional (default: None)
        If save_fig is True, specify figure path.
    fig_name: `str`, optional (default: 'plot_discretize.pdf')
        if `save_fig` is True, specify figure name.
    **kwargs: `dict`, optional
        Other keyword arguments are passed through to ``plt.hist()``

    Returns
    -------
    None
    """
    if kde is not None:
        warnings.warn("kde is not supported as of v0.2", DeprecationWarning)
    if fig_size is None:
        fig_size = mpl.rcParams['figure.figsize']
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, 'figures')

    assert 'disc' in adata.uns_keys(), \
        "please run `si.tl.discretize()` first"
    if kde is not None:
        warnings.warn("kde is no longer supported as of v1.1",
                      DeprecationWarning)

    hist_edges = adata.uns['disc']['hist_edges']
    hist_count = adata.uns['disc']['hist_count']
    bin_edges = adata.uns['disc']['bin_edges']
    bin_count = adata.uns['disc']['bin_count']

    fig, ax = plt.subplots(2, 1, figsize=fig_size)
    _ = ax[0].hist(hist_edges[:-1],
                   hist_edges,
                   weights=hist_count,
                   linewidth=0,
                   **kwargs)
    _ = ax[1].hist(bin_edges[:-1],
                   bin_edges,
                   weights=bin_count,
                   **kwargs)
    ax[0].set_xlabel('Non-zero values')
    ax[0].set_ylabel('Count')
    ax[0].set_title('Original')
    ax[1].set_xlabel('Non-zero values')
    ax[1].set_ylabel('Count')
    ax[1].set_title('Discretized')
    plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    if(save_fig):
        if(not os.path.exists(fig_path)):
            os.makedirs(fig_path)
        plt.savefig(os.path.join(fig_path, fig_name),
                    pad_inches=1,
                    bbox_inches='tight')
        plt.close(fig)


def node_similarity(adata,
                    bins=20,
                    log=True,
                    show_cutoff=True,
                    cutoff=None,
                    n_edges=5000,
                    fig_size=(5, 3),
                    pad=1.08,
                    w_pad=None,
                    h_pad=None,
                    save_fig=None,
                    fig_path=None,
                    fig_name='plot_node_similarity.pdf',
                    ):
    """Plot similarity scores of nodes

    Parameters
    ----------
    adata : `Anndata`
        Annotated data matrix.
    bins : `int`, optional (default: 20)
        The number of equal-width bins in the given range for histogram plot.
    log : `bool`, optional (default: True)
        If True, log scale will be used for y axis.
    show_cutoff : `bool`, optional (default: True)
        If True, cutoff on scores will be shown
    cutoff: `int`, optional (default: None)
        Cutoff used to select edges
    n_edges: `int`, optional (default: 5000)
        The number of edges to select.
    pad: `float`, optional (default: 1.08)
        Padding between the figure edge and the edges of subplots,
        as a fraction of the font size.
    h_pad, w_pad: `float`, optional (default: None)
        Padding (height/width) between edges of adjacent subplots,
        as a fraction of the font size. Defaults to pad.
    fig_size: `tuple`, optional (default: (5,8))
        figure size.
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_path: `str`, optional (default: None)
        If save_fig is True, specify figure path.
    fig_name: `str`, optional (default: 'plot_node_similarity.pdf')
        if `save_fig` is True, specify figure name.

    Returns
    -------
    None
    """
    if fig_size is None:
        fig_size = mpl.rcParams['figure.figsize']
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, 'figures')

    mat_sim = adata.X

    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    ax.hist(mat_sim.data, bins=bins)
    if log:
        ax.set_yscale('log')
    if(show_cutoff):
        if cutoff is None:
            if n_edges is None:
                raise ValueError('"cutoff" or "n_edges" has to be specified')
            else:
                cutoff = \
                    np.partition(mat_sim.data,
                                 (mat_sim.size-n_edges))[mat_sim.size-n_edges]
        id_x, id_y, _ = find(mat_sim > cutoff)
        print(f'#selected edges: {len(id_x)}')
        plt.axvline(cutoff, ls='--', c='red')
    ax.set_xlabel('similariy scores')
    ax.set_title('Node similarity')
    plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    if(save_fig):
        if(not os.path.exists(fig_path)):
            os.makedirs(fig_path)
        fig.savefig(os.path.join(fig_path, fig_name),
                    pad_inches=1,
                    bbox_inches='tight')
        plt.close(fig)


def svd_nodes(adata,
              comp1=1,
              comp2=2,
              color=None,
              dict_palette=None,
              cutoff=None,
              n_edges=5000,
              size=8,
              drawing_order='random',
              dict_drawing_order=None,
              fig_size=(4, 4),
              fig_ncol=3,
              fig_legend_ncol=1,
              fig_legend_order=None,
              alpha=1,
              pad=1.08,
              w_pad=None,
              h_pad=None,
              save_fig=None,
              fig_path=None,
              fig_name='plot_svd_nodes.pdf',
              vmin=None,
              vmax=None,
              **kwargs):
    """Plot SVD coordinates

    Parameters
    ----------
    adata : `Anndata`
        Annotated data matrix.
    comp1: `int`, optional (default: 1)
        Component used for x axis.
    comp2: `int`, optional (default: 2)
        Component used for y axis.
    color: `list`, optional (default: None)
        A list of variables that will produce points with different colors.
        e.g. color = ['anno1', 'anno2']
    cutoff: `int`, optional (default: None)
        Cutoff used to select edges
    n_edges: `int`, optional (default: 5000)
        The number of edges to select
    dict_palette: `dict`,optional (default: None)
        A dictionary of palettes for different variables in `color`.
        Only valid for categorical/string variables
        e.g. dict_palette = {'ann1': {},'ann2': {}}
    drawing_order: `str` (default: 'random')
        The order in which values are plotted, This can be
        one of the following values
        - 'original': plot points in the same order as in input dataframe
        - 'sorted' : plot points with higher values on top.
        - 'random' : plot points in a random order
    dict_drawing_order: `dict`,optional (default: None)
        A dictionary of drawing_order for different variables in `color`.
        Only valid for categorical/string variables
        e.g. dict_drawing_order = {'ann1': 'original','ann2': 'sorted'}
    size: `int` (default: 8)
        Point size.
    fig_size: `tuple`, optional (default: (4, 4))
        figure size.
    fig_ncol: `int`, optional (default: 3)
        the number of columns of the figure panel
    fig_legend_order: `dict`,optional (default: None)
        Specified order for the appearance of the annotation keys.
        Only valid for categorical/string variable
        e.g. fig_legend_order = {'ann1':['a','b','c'],'ann2':['aa','bb','cc']}
    fig_legend_ncol: `int`, optional (default: 1)
        The number of columns that the legend has.
    vmin,vmax: `float`, optional (default: None)
        The min and max values are used to normalize continuous values.
        If None, the respective min and max of continuous values is used.
    alpha: `float`, optional (default: 1)
        0.0 transparent through 1.0 opaque
    pad: `float`, optional (default: 1.08)
        Padding between the figure edge and the edges of subplots,
        as a fraction of the font size.
    h_pad, w_pad: `float`, optional (default: None)
        Padding (height/width) between edges of adjacent subplots,
        as a fraction of the font size. Defaults to pad.
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_path: `str`, optional (default: None)
        If save_fig is True, specify figure path.
    fig_name: `str`, optional (default: 'plot_umap.pdf')
        if save_fig is True, specify figure name.
    Returns
    -------
    None
    """
    if fig_size is None:
        fig_size = mpl.rcParams['figure.figsize']
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, 'figures')

    mat_sim = adata.X
    if cutoff is None:
        if n_edges is None:
            raise ValueError('"cutoff" or "n_edges" has to be specified')
        else:
            cutoff = \
                np.partition(mat_sim.data,
                             (mat_sim.size-n_edges))[mat_sim.size-n_edges]
    id_x, id_y, _ = find(mat_sim > cutoff)

    X_cca_ref = adata.obsm['svd']
    X_cca_query = adata.varm['svd']

    df_plot_ref = pd.DataFrame(data=X_cca_ref[:, [comp1-1, comp2-1]],
                               index=adata.obs.index,
                               columns=[f'Dim {comp1}', f'Dim {comp2}'])
    df_plot_ref['group'] = 'ref'
    df_plot_ref['selected'] = 'no'
    df_plot_ref.loc[df_plot_ref.index[id_x], 'selected'] = 'yes'
    df_plot_query = pd.DataFrame(data=X_cca_query[:, [comp1-1, comp2-1]],
                                 index=adata.var.index,
                                 columns=[f'Dim {comp1}', f'Dim {comp2}'])
    df_plot_query['group'] = 'query'
    df_plot_query['selected'] = 'no'
    df_plot_query.loc[df_plot_query.index[id_y], 'selected'] = 'yes'

    df_plot = pd.concat([df_plot_ref, df_plot_query], axis=0)
    if dict_palette is None:
        dict_palette = dict()
    dict_palette['group'] = {'query': '#4c72b0', 'ref': '#dd8452'}
    dict_palette['selected'] = {'yes': '#000000', 'no': '#D4D3D3'}
    if dict_drawing_order is None:
        dict_drawing_order = dict()
    dict_drawing_order['group'] = 'random'
    dict_drawing_order['selected'] = 'sorted'

    adata.uns['color'] = dict_palette.copy()
    if color is None:
        color = []
    else:
        color = list(dict.fromkeys(color))  # remove duplicate keys
    for ann in color:
        if (ann in adata.obs_keys()) and (ann in adata.var_keys()):
            df_plot[ann] = pd.concat([adata.obs[ann], adata.var[ann]], axis=0)
            if(not is_numeric_dtype(df_plot[ann])):
                if ann not in dict_palette.keys():
                    if (ann+'_color' in adata.uns['color'].keys()) \
                        and \
                        (all(np.isin(np.unique(df_plot[ann]),
                                     list(adata.uns['color']
                                     [ann+'_color'].keys())))):
                        dict_palette[ann] = \
                            adata.uns['color'][ann+'_color']
                    else:
                        dict_palette[ann] = \
                            generate_palette(adata.obs[ann])
                        adata.uns['color'][ann+'_color'] = \
                            dict_palette[ann].copy()
                else:
                    if ann+'_color' not in adata.uns['color'].keys():
                        adata.uns['color'][ann+'_color'] = \
                            dict_palette[ann].copy()
        else:
            raise ValueError(f"could not find {ann} in both "
                             "`adata.obs.columns`"
                             " and `adata.var.columns`")
    color = ['group', 'selected'] + color
    _scatterplot2d(df_plot,
                   x=f'Dim {comp1}',
                   y=f'Dim {comp2}',
                   list_hue=color,
                   hue_palette=dict_palette,
                   drawing_order=drawing_order,
                   dict_drawing_order=dict_drawing_order,
                   size=size,
                   fig_size=fig_size,
                   fig_ncol=fig_ncol,
                   fig_legend_ncol=fig_legend_ncol,
                   fig_legend_order=fig_legend_order,
                   vmin=vmin,
                   vmax=vmax,
                   alpha=alpha,
                   pad=pad,
                   w_pad=w_pad,
                   h_pad=h_pad,
                   save_fig=save_fig,
                   fig_path=fig_path,
                   fig_name=fig_name,
                   **kwargs)
