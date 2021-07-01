"""post-training plotting functions"""

import os
import numpy as np
import pandas as pd
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.collections import LineCollection
from adjustText import adjust_text
from pandas.api.types import (
    is_numeric_dtype
)
from scipy.stats import rankdata

from ._utils import (
    get_colors,
    generate_palette
)
from .._settings import settings
from ._plot import _scatterplot2d


def pbg_metrics(metrics=['mrr'],
                path_emb=None,
                fig_size=(5, 3),
                fig_ncol=1,
                save_fig=None,
                fig_path=None,
                fig_name='pbg_metrics.pdf',
                pad=1.08,
                w_pad=None,
                h_pad=None,
                **kwargs):
    """Plot PBG training metrics

    Parameters
    ----------
    metrics: `list`, optional (default: ['mrr])
        Evalulation metrics for PBG training. Possible metrics:

        - 'pos_rank' : the average of the ranks of all positives
          (lower is better, best is 1).
        - 'mrr' : the average of the reciprocal of the ranks of all positives
          (higher is better, best is 1).
        - 'r1' : the fraction of positives that rank better than
           all their negatives, i.e., have a rank of 1
           (higher is better, best is 1).
        - 'r10' : the fraction of positives that rank in the top 10
           among their negatives
           (higher is better, best is 1).
        - 'r50' : the fraction of positives that rank in the top 50
           among their negatives
           (higher is better, best is 1).
        - 'auc' : Area Under the Curve (AUC)
    path_emb: `str`, optional (default: None)
        Path to directory for pbg embedding model.
        If None, .settings.pbg_params['checkpoint_path'] will be used.
    pad: `float`, optional (default: 1.08)
        Padding between the figure edge and the edges of subplots,
        as a fraction of the font size.
    h_pad, w_pad: `float`, optional (default: None)
        Padding (height/width) between edges of adjacent subplots,
        as a fraction of the font size. Defaults to pad.
    fig_size: `tuple`, optional (default: (5, 3))
        figure size.
    fig_ncol: `int`, optional (default: 1)
        the number of columns of the figure panel
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
    if save_fig is None:
        save_fig = settings.save_fig
    if fig_path is None:
        fig_path = os.path.join(settings.workdir, 'figures')

    assert isinstance(metrics, list), "`metrics` must be list"
    for x in metrics:
        if(x not in ['pos_rank', 'mrr', 'r1',
                     'r10', 'r50', 'auc']):
            raise ValueError(f'unrecognized metric {x}')
    pbg_params = settings.pbg_params
    if path_emb is None:
        path_emb = pbg_params['checkpoint_path']
    training_loss = []
    eval_stats_before = dict()
    with open(os.path.join(path_emb, 'training_stats.json'), 'r') as f:
        for line in f:
            line_json = json.loads(line)
            if('stats' in line_json.keys()):
                training_loss.append(line_json['stats']['metrics']['loss'])
                line_stats_before = line_json['eval_stats_before']['metrics']
                for x in line_stats_before.keys():
                    if(x not in eval_stats_before.keys()):
                        eval_stats_before[x] = [line_stats_before[x]]
                    else:
                        eval_stats_before[x].append(line_stats_before[x])
    df_metrics = pd.DataFrame(index=range(pbg_params['num_epochs']))
    df_metrics['epoch'] = range(pbg_params['num_epochs'])
    df_metrics['training_loss'] = training_loss
    df_metrics['validation_loss'] = eval_stats_before['loss']
    for x in metrics:
        df_metrics[x] = eval_stats_before[x]

    fig_nrow = int(np.ceil((df_metrics.shape[1]-1)/fig_ncol))
    fig = plt.figure(figsize=(fig_size[0]*fig_ncol*1.05,
                              fig_size[1]*fig_nrow))
    dict_palette = generate_palette(df_metrics.columns[1:].values)
    for i, metric in enumerate(df_metrics.columns[1:]):
        ax_i = fig.add_subplot(fig_nrow, fig_ncol, i+1)
        ax_i.scatter(df_metrics['epoch'],
                     df_metrics[metric],
                     c=dict_palette[metric],
                     **kwargs)
        ax_i.set_title(metric)
        ax_i.set_xlabel('epoch')
        ax_i.set_ylabel(metric)
    plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    if(save_fig):
        if(not os.path.exists(fig_path)):
            os.makedirs(fig_path)
        plt.savefig(os.path.join(fig_path, fig_name),
                    pad_inches=1,
                    bbox_inches='tight')
        plt.close(fig)


def entity_metrics(adata_cmp,
                   x,
                   y,
                   show_texts=True,
                   show_cutoff=False,
                   show_contour=True,
                   levels=4,
                   thresh=0.05,
                   cutoff_x=0,
                   cutoff_y=0,
                   n_texts=10,
                   size=8,
                   texts=None,
                   text_size=10,
                   text_expand=(1.05, 1.2),
                   fig_size=None,
                   save_fig=None,
                   fig_path=None,
                   fig_name='entity_metrics.pdf',
                   pad=1.08,
                   w_pad=None,
                   h_pad=None,
                   **kwargs):
    """Plot entity metrics

    Parameters
    ----------
    adata_cmp: `AnnData`
        Anndata object from `compare_entities`
    x, y: `str`
        Variables that specify positions on the x and y axes.
        Possible values:
        - max (The average maximum dot product of top-rank reference entities,
        based on normalized dot product)
        - std (standard deviation of reference entities,
        based on dot product)
        - gini (Gini coefficients of reference entities,
        based on softmax probability)
        - entropy (The entropy of reference entities,
        based on softmax probability)
    show_texts : `bool`, optional (default: True)
        If True, text annotation will be shown.
    show_cutoff : `bool`, optional (default: False)
        If True, cutoff of `x` and `y` will be shown.
    show_contour : `bool`, optional (default: True)
        If True, the plot will overlaid with contours
    texts: `list` optional (default: None)
        Entity names to plot
    text_size : `int`, optional (default: 10)
        The text size
    text_expand : `tuple`, optional (default: (1.05, 1.2))
        Two multipliers (x, y) by which to expand the bounding box of texts
        when repelling them from each other/points/other objects.
    cutoff_x : `float`, optional (default: 0)
        Cutoff of axis x
    cutoff_y : `float`, optional (default: 0)
        Cutoff of axis y
    levels: `int`, optional (default: 6)
        Number of contour levels or values to draw contours at
    thresh: `float`, optional ([0, 1], default: 0.05)
        Lowest iso-proportion level at which to draw a contour line.
    pad: `float`, optional (default: 1.08)
        Padding between the figure edge and the edges of subplots,
        as a fraction of the font size.
    h_pad, w_pad: `float`, optional (default: None)
        Padding (height/width) between edges of adjacent subplots,
        as a fraction of the font size. Defaults to pad.
    fig_size: `tuple`, optional (default: None)
        figure size.
        If None, `mpl.rcParams['figure.figsize']` will be used.
    fig_ncol: `int`, optional (default: 1)
        the number of columns of the figure panel
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

    assert (x in ['max', 'std', 'gini', 'entropy']), \
        "x must be one of ['max','std','gini','entropy']"
    assert (y in ['max', 'std', 'gini', 'entropy']), \
        "y must be one of ['max','std','gini','entropy']"

    fig, ax = plt.subplots(figsize=fig_size)
    ax.scatter(adata_cmp.var[x],
               adata_cmp.var[y],
               s=size,
               **kwargs)
    if show_texts:
        if texts is not None:
            plt_texts = [plt.text(adata_cmp.var[x][t],
                                  adata_cmp.var[y][t],
                                  t,
                         fontdict={'family': 'serif',
                                   'color': 'black',
                                   'weight': 'normal',
                                   'size': text_size})
                         for t in texts]
        else:
            if x == 'entropy':
                ranks_x = rankdata(-adata_cmp.var[x])
            else:
                ranks_x = rankdata(adata_cmp.var[x])
            if y == 'entropy':
                ranks_y = rankdata(-adata_cmp.var[y])
            else:
                ranks_y = rankdata(adata_cmp.var[y])
            ids = np.argsort(ranks_x + ranks_y)[::-1][:n_texts]
            plt_texts = [plt.text(adata_cmp.var[x][i],
                                  adata_cmp.var[y][i],
                                  adata_cmp.var_names[i],
                         fontdict={'family': 'serif',
                                   'color': 'black',
                                   'weight': 'normal',
                                   'size': text_size})
                         for i in ids]
        adjust_text(plt_texts,
                    expand_text=text_expand,
                    expand_points=text_expand,
                    expand_objects=text_expand,
                    arrowprops=dict(arrowstyle='-', color='black'))
    if show_cutoff:
        ax.axvline(x=cutoff_x, linestyle='--', color='#CE3746')
        ax.axhline(y=cutoff_y, linestyle='--', color='#CE3746')
    if show_contour:
        sns.kdeplot(ax=ax,
                    data=adata_cmp.var,
                    x=x,
                    y=y,
                    alpha=0.7,
                    color='black',
                    levels=levels,
                    thresh=thresh)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.locator_params(axis='x', tight=True)
    ax.locator_params(axis='y', tight=True)
    fig.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    if(save_fig):
        if(not os.path.exists(fig_path)):
            os.makedirs(fig_path)
        fig.savefig(os.path.join(fig_path, fig_name),
                    pad_inches=1,
                    bbox_inches='tight')
        plt.close(fig)


def entity_barcode(adata_cmp,
                   entities,
                   anno_ref=None,
                   layer='softmax',
                   palette=None,
                   alpha=0.8,
                   linewidths=1,
                   show_cutoff=False,
                   cutoff=0.5,
                   min_rank=None,
                   max_rank=None,
                   fig_size=(6, 2),
                   fig_ncol=1,
                   save_fig=None,
                   fig_path=None,
                   fig_name='plot_barcode.pdf',
                   pad=1.08,
                   w_pad=None,
                   h_pad=None,
                   **kwargs
                   ):
    """Plot query entity barcode

    Parameters
    ----------
    adata_cmp : `AnnData`
        Anndata object from `compare_entities`
    entities : `list`
        Entity names to plot.
    anno_ref :  `str`
        Annotation used for reference entity
    layer : `str`, optional (default: 'softmax')
        Layer to use make barcode plots
    palette : `dict`, optional (default: None)
        Color palette used for `anno_ref`
    alpha : `float`, optional (default: 0.8)
        0.0 transparent through 1.0 opaque
    linewidths : `int`, optional (default: 1)
        The width of each line.
    show_cutoff : `bool`, optional (default: True)
        If True, cutoff will be shown
    cutoff : `float`, optional (default: 0.5)
        Cutoff value for y axis
    min_rank : `int`, optional (default: None)
        Specify the minimum rank of observations to show.
        If None, `min_rank` is set to 0.
    max_rank : `int`, optional (default: None)
        Specify the maximum rank of observations to show.
        If None, `max_rank` is set to the number of observations.
    fig_size: `tuple`, optional (default: (6,2))
        figure size.
    fig_ncol: `int`, optional (default: 1)
        the number of columns of the figure panel
    save_fig: `bool`, optional (default: False)
        if True,save the figure.
    fig_path: `str`, optional (default: None)
        If save_fig is True, specify figure path.
    fig_name: `str`, optional (default: 'plot_barcode.pdf')
        if `save_fig` is True, specify figure name.
    **kwargs: `dict`, optional
        Other keyword arguments are passed through to
        ``mpl.collections.LineCollection``

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

    assert isinstance(entities, list), "`entities` must be list"

    if layer is None:
        X = adata_cmp[:, entities].X.copy()
    else:
        X = adata_cmp[:, entities].layers[layer].copy()
    df_scores = pd.DataFrame(
        data=X,
        index=adata_cmp.obs_names,
        columns=entities)

    if min_rank is None:
        min_rank = 0
    if max_rank is None:
        max_rank = df_scores.shape[0]

    n_plots = len(entities)
    fig_nrow = int(np.ceil(n_plots/fig_ncol))
    fig = plt.figure(figsize=(fig_size[0]*fig_ncol*1.05,
                              fig_size[1]*fig_nrow))

    for i, x in enumerate(entities):
        ax_i = fig.add_subplot(fig_nrow, fig_ncol, i+1)
        scores_x_sorted = df_scores[x].sort_values(ascending=False)
        lines = []
        for xx, yy in zip(np.arange(len(scores_x_sorted))[min_rank:max_rank],
                          scores_x_sorted[min_rank:max_rank]):
            lines.append([(xx, 0), (xx, yy)])
        if anno_ref is None:
            colors = get_colors(np.array([""]*len(scores_x_sorted)))
        else:
            ids_ref = scores_x_sorted.index
            if palette is None:
                colors = get_colors(adata_cmp[ids_ref, :].obs[anno_ref])
            else:
                colors = [palette[adata_cmp.obs.loc[xx, anno_ref]]
                          for xx in scores_x_sorted.index]
        stemlines = LineCollection(
            lines,
            colors=colors,
            alpha=alpha,
            linewidths=linewidths,
            **kwargs)
        ax_i.add_collection(stemlines)
        ax_i.autoscale()
        ax_i.set_title(x)
        ax_i.set_ylabel(layer)
        ax_i.locator_params(axis='y', tight=True)
        if show_cutoff:
            ax_i.axhline(y=cutoff,
                         color='#CC6F47',
                         linestyle='--')
    plt.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad)
    if(save_fig):
        if(not os.path.exists(fig_path)):
            os.makedirs(fig_path)
        plt.savefig(os.path.join(fig_path, fig_name),
                    pad_inches=1,
                    bbox_inches='tight')
        plt.close(fig)


def query(adata,
          comp1=1,
          comp2=2,
          obsm='X_umap',
          layer=None,
          color=None,
          dict_palette=None,
          size=8,
          drawing_order='random',
          dict_drawing_order=None,
          show_texts=False,
          texts=None,
          text_expand=(1.05, 1.2),
          text_size=10,
          n_texts=8,
          fig_size=None,
          fig_ncol=3,
          fig_legend_ncol=1,
          fig_legend_order=None,
          alpha=0.9,
          alpha_bg=0.3,
          pad=1.08,
          w_pad=None,
          h_pad=None,
          save_fig=None,
          fig_path=None,
          fig_name='plot_query.pdf',
          vmin=None,
          vmax=None,
          **kwargs):
    """Plot query output

    Parameters
    ----------
    adata : `Anndata`
        Annotated data matrix.
    comp1 : `int`, optional (default: 1)
        Component used for x axis.
    comp2 : `int`, optional (default: 2)
        Component used for y axis.
    obsm : `str`, optional (default: 'X_umap')
        The field to use for plotting
    layer : `str`, optional (default: None)
        The layer to use for plotting
    color: `list`, optional (default: None)
        A list of variables that will produce points with different colors.
        e.g. color = ['anno1', 'anno2']
    dict_palette: `dict`,optional (default: None)
        A dictionary of palettes for different variables in `color`.
        Only valid for categorical/string variables
        e.g. dict_palette = {'ann1': {},'ann2': {}}
    size: `int` (default: 8)
        Point size.
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
    show_texts : `bool`, optional (default: False)
        If True, text annotation will be shown.
    text_size : `int`, optional (default: 10)
        The text size.
    texts: `list` optional (default: None)
        Point names to plot.
    text_expand : `tuple`, optional (default: (1.05, 1.2))
        Two multipliers (x, y) by which to expand the bounding box of texts
        when repelling them from each other/points/other objects.
    n_texts : `int`, optional (default: 8)
        The number of texts to plot.
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
    alpha: `float`, optional (default: 0.9)
        The alpha blending value, between 0 (transparent) and 1 (opaque)
        for returned points.
    alpha_bg: `float`, optional (default: 0.3)
        The alpha blending value, between 0 (transparent) and 1 (opaque)
        for background points
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
    fig_name: `str`, optional (default: 'plot_query.pdf')
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

    if dict_palette is None:
        dict_palette = dict()

    query_output = adata.uns['query']['output']
    nn = query_output.index.tolist()  # nearest neighbors
    query_params = adata.uns['query']['params']
    query_obsm = query_params['obsm']
    query_layer = query_params['layer']
    entity = query_params['entity']
    use_radius = query_params['use_radius']
    r = query_params['r']
    if (obsm == query_obsm) and (layer == query_layer):
        pin = query_params['pin']
    else:
        if entity is not None:
            if obsm is not None:
                pin = adata[entity, :].obsm[obsm].copy()
            elif layer is not None:
                pin = adata[entity, :].layers[layer].copy()
            else:
                pin = adata[entity, :].X.copy()
        else:
            pin = None

    if(sum(list(map(lambda x: x is not None,
                    [layer, obsm]))) == 2):
        raise ValueError("Only one of `layer` and `obsm` can be used")
    if obsm is not None:
        X = adata.obsm[obsm].copy()
        X_nn = adata[nn, :].obsm[obsm].copy()
    elif layer is not None:
        X = adata.layers[layer].copy()
        X_nn = adata[nn, :].layers[layer].copy()
    else:
        X = adata.X.copy()
        X_nn = adata[nn, :].X.copy()
    df_plot = pd.DataFrame(index=adata.obs.index,
                           data=X[:, [comp1-1, comp2-1]],
                           columns=[f'Dim {comp1}', f'Dim {comp2}'])
    df_plot_nn = pd.DataFrame(index=adata[nn, :].obs.index,
                              data=X_nn[:, [comp1-1, comp2-1]],
                              columns=[f'Dim {comp1}', f'Dim {comp2}'])
    if show_texts:
        if texts is None:
            texts = nn[:n_texts]
    if color is None:
        list_ax = _scatterplot2d(df_plot,
                                 x=f'Dim {comp1}',
                                 y=f'Dim {comp2}',
                                 drawing_order=drawing_order,
                                 size=size,
                                 fig_size=fig_size,
                                 alpha=alpha_bg,
                                 pad=pad,
                                 w_pad=w_pad,
                                 h_pad=h_pad,
                                 save_fig=False,
                                 copy=True,
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
        list_ax = _scatterplot2d(df_plot,
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
                                 alpha=alpha_bg,
                                 pad=pad,
                                 w_pad=w_pad,
                                 h_pad=h_pad,
                                 save_fig=False,
                                 copy=True,
                                 **kwargs)
    for ax in list_ax:
        ax.scatter(df_plot_nn[f'Dim {comp1}'],
                   df_plot_nn[f'Dim {comp2}'],
                   s=size,
                   color='#AE6C68',
                   alpha=alpha,
                   lw=0)
        if pin is not None:
            ax.scatter(pin[:, 0],
                       pin[:, 1],
                       s=20*size,
                       marker='+',
                       color='#B33831')
            if use_radius:
                circle = plt.Circle((pin[:, 0],
                                     pin[:, 1]),
                                    radius=r,
                                    color='#B33831',
                                    fill=False)
                ax.add_artist(circle)
        if show_texts:
            plt_texts = [ax.text(df_plot_nn[f'Dim {comp1}'][t],
                                 df_plot_nn[f'Dim {comp2}'][t],
                                 t,
                                 fontdict={'family': 'serif',
                                           'color': 'black',
                                           'weight': 'normal',
                                           'size': text_size})
                         for t in texts]
            adjust_text(plt_texts,
                        ax=ax,
                        expand_text=text_expand,
                        expand_points=text_expand,
                        expand_objects=text_expand,
                        arrowprops=dict(arrowstyle='->', color='black'))
    if save_fig:
        fig = plt.gcf()
        if(not os.path.exists(fig_path)):
            os.makedirs(fig_path)
        fig.savefig(os.path.join(fig_path, fig_name),
                    pad_inches=1,
                    bbox_inches='tight')
        plt.close(fig)
