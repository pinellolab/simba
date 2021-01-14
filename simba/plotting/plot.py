"""Plot"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from adjustText import adjust_text


# import sys
# sys.path.insert(0, '/Users/huidong/Projects/Github/simba/simba/')
# from _settings import settings
from .._settings import settings


def violin(adata,
           list_obs=None,
           list_var=None,
           jitter=0.4,
           size=1,
           log=False,
           pad=1.08,
           w_pad=None,
           h_pad=3,
           fig_size=(4,4),
           fig_ncol=3,
           save_fig=False,
           fig_path=None,
           fig_name='plot_violin.pdf',
           **kwargs):
    """Violin plot
    """
    if(fig_size is None):
        fig_size = mpl.rcParams['figure.figsize']
    if(save_fig is None):
        save_fig = settings.save_fig
    if(fig_path is None):
        fig_path = os.path.join(settings.workdir, 'figures')
    if(list_obs is None):
        list_obs = []
    if(list_var is None):
        list_var = []
    for obs in list_obs:
        if(obs not in adata.obs_keys()):
            raise ValueError(f"could not find {obs} in `adata.obs_keys()`")
    for var in list_var:
        if(var not in adata.var_keys()):
            raise ValueError(f"could not find {var} in `adata.var_keys()`")        

    if(len(list_obs)>0):
        df_plot = adata.obs[list_obs].copy()
        if(log):
            df_plot = pd.DataFrame(data=np.log1p(df_plot.values),
                                   index=df_plot.index,
                                   columns=df_plot.columns)
        fig_nrow = int(np.ceil(len(list_obs)/fig_ncol))
        fig = plt.figure(figsize=(fig_size[0]*fig_ncol*1.05, fig_size[1]*fig_nrow))
        for i,obs in enumerate(list_obs):
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
            ax_i.locator_params(axis='y',nbins=6)
            ax_i.tick_params(axis="y",pad=-2)
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
    if(len(list_var)>0):
        df_plot = adata.var[list_var].copy()
        if(log):
            df_plot = pd.DataFrame(data=np.log1p(df_plot.values),
                                   index=df_plot.index,
                                   columns=df_plot.columns)
        fig_nrow = int(np.ceil(len(list_obs)/fig_ncol))
        fig = plt.figure(figsize=(fig_size[0]*fig_ncol*1.05, fig_size[1]*fig_nrow))
        for i,var in enumerate(list_var):
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
            ax_i.locator_params(axis='y',nbins=6)
            ax_i.tick_params(axis="y",pad=-2)
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
        size=1,
        log=False,
        pad=1.08,
        w_pad=None,
        h_pad=3,
        fig_size=(4,4),
        fig_ncol=3,
        save_fig=False,
        fig_path=None,
        fig_name='plot_violin.pdf',
        **kwargs):
    """Violin plot
    """
    if(fig_size is None):
        fig_size = mpl.rcParams['figure.figsize']
    if(save_fig is None):
        save_fig = settings.save_fig
    if(fig_path is None):
        fig_path = os.path.join(settings.workdir, 'figures')
    if(list_obs is None):
        list_obs = []
    if(list_var is None):
        list_var = []
    for obs in list_obs:
        if(obs not in adata.obs_keys()):
            raise ValueError(f"could not find {obs} in `adata.obs_keys()`")
    for var in list_var:
        if(var not in adata.var_keys()):
            raise ValueError(f"could not find {var} in `adata.var_keys()`")        

    if(len(list_obs)>0):
        df_plot = adata.obs[list_obs].copy()
        if(log):
            df_plot = pd.DataFrame(data=np.log1p(df_plot.values),
                                   index=df_plot.index,
                                   columns=df_plot.columns)
        fig_nrow = int(np.ceil(len(list_obs)/fig_ncol))
        fig = plt.figure(figsize=(fig_size[0]*fig_ncol*1.05, fig_size[1]*fig_nrow))
        for i,obs in enumerate(list_obs):
            ax_i = fig.add_subplot(fig_nrow, fig_ncol, i+1)
            sns.histplot(ax=ax_i,
                         x=obs,
                         data=df_plot,
                         kde=kde,
                         **kwargs)
            ax_i.locator_params(axis='y',nbins=6)
            ax_i.tick_params(axis="y",pad=-2)
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
    if(len(list_var)>0):
        df_plot = adata.var[list_var].copy()
        if(log):
            df_plot = pd.DataFrame(data=np.log1p(df_plot.values),
                                   index=df_plot.index,
                                   columns=df_plot.columns)
        fig_nrow = int(np.ceil(len(list_obs)/fig_ncol))
        fig = plt.figure(figsize=(fig_size[0]*fig_ncol*1.05, fig_size[1]*fig_nrow))
        for i,var in enumerate(list_var):
            ax_i = fig.add_subplot(fig_nrow, fig_ncol, i+1)
            sns.histplot(ax=ax_i,
                         x=var,
                         data=df_plot,
                         kde=kde,
                         **kwargs)
            ax_i.locator_params(axis='y',nbins=6)
            ax_i.tick_params(axis="y",pad=-2)
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
                       fig_name='qc.pdf',
                       pad=1.08,
                       w_pad=None,
                       h_pad=None):
    """Plot the variance ratio.
    """
    if(fig_size is None):
        fig_size = mpl.rcParams['figure.figsize']
    if(save_fig is None):
        save_fig = settings.save_fig
    if(fig_path is None):
        fig_path = os.path.join(settings.workdir, 'figures')

    n_components = len(adata.uns['pca']['variance_ratio'])

    fig = plt.figure(figsize=fig_size)
    if(log):
        plt.plot(range(n_components),
                 np.log(adata.uns['pca']['variance_ratio']))
    else:
        plt.plot(range(n_components),
                 adata.uns['pca']['variance_ratio'])
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
                 show_cutoff=True,
                 fig_size=None,
                 fig_ncol=3,
                 save_fig=None,
                 fig_path=None,
                 fig_name='qc.pdf',
                 pad=1.08,
                 w_pad=None,
                 h_pad=None):
    """Plot features that contribute to the top PCs.
    """
    if(fig_size is None):
        fig_size = mpl.rcParams['figure.figsize']
    if(save_fig is None):
        save_fig = settings.save_fig
    if(fig_path is None):
        fig_path = os.path.join(settings.workdir, 'figures')

    n_pcs = adata.uns['pca']['n_pcs']
    n_features = adata.varm['PCs'].shape[0]

    fig_nrow = int(np.ceil(n_pcs/fig_ncol))
    fig = plt.figure(figsize=(fig_size[0]*fig_ncol*1.05, fig_size[1]*fig_nrow))

    for i in range(n_pcs):
        ax_i = fig.add_subplot(fig_nrow, fig_ncol, i+1)
        if(log):
            ax_i.scatter(range(n_features),
                         np.log(np.sort(
                             np.abs(adata.varm['PCs'][:, i],))[::-1]))
        else:
            ax_i.scatter(range(n_features),
                         np.sort(
                             np.abs(adata.varm['PCs'][:, i],))[::-1])
        n_ft_selected_i = len(adata.uns['pca']['features'][f'pc_{i}'])
        if(show_cutoff):
            print(f'#features selected from PC {i} is: {n_ft_selected_i}')
            ax_i.axvline(n_ft_selected_i, ls='--', c='red')
        ax_i.set_xlabel('Feautures')
        ax_i.set_ylabel('Loadings')
        ax_i.locator_params(axis='x', nbins=5)
        ax_i.locator_params(axis='y', nbins=5)
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
                    fig_size=(4, 4),
                    save_fig=None,
                    fig_path=None,
                    fig_name='plot_variable_genes.pdf',
                    pad=1.08,
                    w_pad=None,
                    h_pad=None):
    """Plot highly variable genes.
    """
    if(fig_size is None):
        fig_size = mpl.rcParams['figure.figsize']
    if(save_fig is None):
        save_fig = settings.save_fig
    if(fig_path is None):
        fig_path = os.path.join(settings.workdir, 'figures')

    means = adata.var['means']
    variances_norm = adata.var['variances_norm']
    mask = adata.var['highly_variable']
    genes = adata.var_names

    fig, ax = plt.subplots(figsize=fig_size)
    ax.scatter(means[~mask],
                variances_norm[~mask],
                s=size,
                c='#1F2433')
    ax.scatter(means[mask],
                variances_norm[mask],
                s=size,
                c='#ce3746')
    ax.set_xscale(value='log')

    if(show_texts):
        ids = variances_norm.values.argsort()[-n_texts:][::-1]
        texts = [plt.text(means[i], variances_norm[i], genes[i],
                    fontdict={'family': 'serif','color': 'black','weight': 'normal','size': text_size,}) 
                for i in ids]
        adjust_text(texts,arrowprops=dict(arrowstyle='-', color='black'))

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