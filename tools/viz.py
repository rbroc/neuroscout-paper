from pathlib import Path
from nilearn import plotting as niplt
from pyns import Neuroscout
from .base import (find_image, flatten_collection, 
                    _extract_regressors, compute_metrics)
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
api = Neuroscout()

RESULTS_DIR = Path('/media/hyperdrive/neuroscout-cli/output/')


def plot_contrast(contrast, analysis, results_dir=None, title=None,
                  cut_coords=range(-25, 55, 12), display_mode='z',
                  target='t', threshold=1.96, axes=None, **kwargs):
    """Plot contrast image for a single contrast image"""
    hash_id = analysis.hash_id
    dataset = api.datasets.get(analysis.dataset_id)['name']
    if title is None:
        title = f"{dataset}-{analysis.name}-{contrast}"

    img = find_image(contrast, hash_id, results_dir=results_dir, target=target)
    if img:
        p = niplt.plot_stat_map(
            img,
            display_mode=display_mode, 
            cut_coords=cut_coords,
            threshold=threshold,
            axes=axes,
            **kwargs)
        p.title(title, x=-.0, y=1.1, color='black', bgcolor='white', fontsize=12)
    else:
        print(f"No image for {title}")

def _get_con_names(analysis, contrasts=None):
    """ Get available contrast names at dataset level """
    last = analysis.model['Steps'][-1]
    con_names = []
    if 'DummyContrasts' in last:
        first = analysis.model['Steps'][0]
        con_names += [c['Name'] for c in first['Contrasts']]
        if first['DummyContrasts']:
            if 'Conditions' not in first['DummyContrasts']:
                con_names += first['DummyContrasts']['Conditions'] \
                    if 'Conditions' in first['DummyContrasts'] \
                    else first['Model']['X']
            else:
                con_names += first['DummyContrasts']['Conditions']
        con_names = [(c, last['DummyContrasts']['Type']) for c in con_names]
    if 'Contrasts' in last:
        con_names += [(c['Name'], c['Type']) for c in last['Contrasts']]
    if contrasts is not None:
        con_names = [c for c in con_names if c[0] in contrasts]
    return con_names

def _truncate_string(si, max_char):
    if len(si) > max_char:
        middle_point = int(len(si) / 2)
        diff = int((len(si) - max_char) / 2)
        end = middle_point - diff
        si = si[:end]+"..."+si[-end:]
    return si


def plot_individual_analyses(analyses, contrasts=None, models=None, datasets=None,  
                             display_plot=True, **kwargs):
    """ Plot various analyses with the same set of contrasts from different
        datasets.
    Args:
        analyses: flattened list of analyses
        contrasts: contrast subset
        models: model subset
        datasets: dataset subset
    """
    if models:
        analyses = [a for a in analyses if a[0][0] in models]
    if datasets:
        analyses = [a for a in analyses if a[0][1] in datasets]
    for analysis in analyses:
        cons = _get_con_names(analysis[1], contrasts=contrasts)
        for name, c_type in cons:
            plot_contrast(name, analysis[1], target=c_type, **kwargs)


def plot_contrast_by_analysis(analyses, contrasts=None, models=None, 
                              datasets=None, outpath=None,
                              out_format='.png', figsize=(10,10), 
                              title_fontsize=16, **kwargs):
    ''' Groups same contrast for same models (different datasets) in one plot
    Args:
        analyses: flattened analysis dictionary
        contrasts: subset of contrasts
        models: subset of models
        datasets: subset of available datasets
        outpath: output path for images (if not defined, images are not saved)
        out_format: output format (defaults to png)
        figsize: figure size
        title_fontsize: fontsize of plot
    '''
    if models:
        analyses = [a for a in analyses if a[0][0] in models]
    if datasets:
        analyses = [a for a in analyses if a[0][1] in datasets]
        
    unique_model_names = list(set([a[0][0] for a in analyses]))
    for m in unique_model_names:
        model_list = []
        m_analyses = [a for a in analyses if a[0][0] == m]
        
        for analysis in m_analyses:
            con_names = _get_con_names(analysis[1], contrasts=contrasts)
            unique_contrasts = list(set(con_names))
            for c, type in con_names:
                model_list.append((c, analysis))
                
        for c, c_type in unique_contrasts:
            analysis_list = [el[1] for el in model_list if el[0] == c]
            f, axarr = plt.subplots(nrows=len(analysis_list), ncols=1, figsize=figsize)
            if len(analysis_list) > 1:
                [axi.set_axis_off() for axi in axarr.ravel()]
            
            for idx, analysis in enumerate(analysis_list):
                title = analysis[0][1]
                ax = axarr[idx]
                plot_contrast(c, analysis[1], target=c_type, axes=ax, title=title, **kwargs)
            f.suptitle(f'{c} [model: {_truncate_string(m, 35)}]', y=.92, fontsize=title_fontsize)
            
            if outpath is not None:
                outfile = f'{c}_{m}{out_format}'
                outfile = Path(str(outpath)) / outfile
                plt.savefig(outfile)
            plt.show()


def plot_contrast_by_dataset(analyses, contrasts=None, models=None, 
                             datasets=None, outpath=None,
                             out_format='.png', figsize=(10,10), 
                             title_fontsize=16, **kwargs):
    ''' Groups same contrast for same dataset (different models) in one plot
    Args:
        analyses: flattened analysis dictionary
        contrasts: subset of contrasts
        models: subset of models
        datasets: subset of available datasets
        outpath: output path for images (if not defined, images are not saved)
        out_format: output format (defaults to png)
        figsize: figure size
        title_fontsize: fontsize of plot
    '''

    if models:
        analyses = [a for a in analyses if a[0][0] in models]
    if datasets:
        analyses = [a for a in analyses if a[0][1] in datasets]
    
    unique_datasets = list(set(['_'.join(a[0][1:3]) for a in analyses]))
    for d in unique_datasets:
        model_list = []
        d_analyses = [a for a in analyses 
                     if (a[0][2] == d.split('_')[1]) and (a[0][1] == d.split('_')[0])]
        
        # Create analysis list step
        for analysis in d_analyses:
            first = analysis[1].model['Steps'][0]
            unique_contrasts = list(set(_get_con_names(analysis[1], contrasts=contrasts)))
            for c, _ in unique_contrasts:
                model_list.append((c, analysis))

        # Create contrasts and plot   
        for c, c_type in unique_contrasts:
            analysis_list = [el[1] for el in model_list if el[0] == c]
            if len(analysis_list) == 1:
                analysis = analysis_list[0]
                plot_contrast(c, analysis[1], target=c_type, title=f'model: {_truncate_string(analysis[0][0], 70)}', **kwargs)
                plt.suptitle(f'Contrast: {c} [{analysis[0][1]}]', y=1.2, fontsize=title_fontsize)
            else:
                f, axarr = plt.subplots(nrows=len(analysis_list), ncols=1, figsize=figsize)
                [axi.set_axis_off() for axi in axarr.ravel()]
                for idx, analysis in enumerate(analysis_list):
                    ax = axarr[idx]
                    plot_contrast(c, analysis[1], axes=ax, title=f'model: {_truncate_string(analysis[0][0], 70)}', **kwargs)
                f.suptitle(f'Contrast: {c} [{analysis[0][1]}]', y=.92, fontsize=title_fontsize)
            
            if outpath is not None:
                outfile = f'{c}_{d}{out_format}'
                outfile = Path(str(outpath)) / outfile
                plt.savefig(outfile)
            plt.show()


def plot_analysis_by_dataset(analyses, contrasts=None, models=None, 
                             datasets=None, outpath=None,
                             out_format='.png', figsize=(10,10), 
                             title_fontsize=16, **kwargs):
    ''' Groups different constrasts for same dataset for same model in one plot.
    Args:
        analyses: flattened analysis dictionary
        contrasts: subset of contrasts
        models: subset of models
        datasets: subset of available datasets
        outpath: output path for images (if not defined, images are not saved)
        out_format: output format (defaults to png)
        figsize: figure size
        title_fontsize: fontsize of plot
    '''
    if models:
        analyses = [a for a in analyses if a[0][0] in models]
    if datasets:
        analyses = [a for a in analyses if a[0][1] in datasets]

    unique_model_names = list(set([a[0][0] for a in analyses]))
    unique_datasets = list(set(['_'.join(a[0][1:3]) for a in analyses]))
    
    for m in unique_model_names:
        for d in unique_datasets:
            a_list  = [a for a in analyses if a[0] == (m, d.split('_')[0], d.split('_')[1])]
            if len(a_list) == 1:
                analysis = a_list[0]
                cons = _get_con_names(analysis[1], contrasts=contrasts)
                if len(cons) == 1:
                    title = f'{d}, model: {_truncate_string(analysis[0][0], 70)}'
                    plot_contrast(cons[0][0], analysis[1], 
                                  title=cons[0], target=cons[0][1], **kwargs)
                    plt.suptitle(title, y=.92, fontsize=title_fontsize)
                else:
                    f, axarr = plt.subplots(nrows=len(cons), ncols=1, figsize=figsize)
                    [axi.set_axis_off() for axi in axarr.ravel()]
                    for idx, (c, c_type) in enumerate(cons):
                        title = f'{d}, model: {_truncate_string(analysis[0][0], 70)}'
                        ax = axarr[idx]
                        plot_contrast(c, analysis[1], target=c_type, axes=ax, title=c, **kwargs)
                    f.suptitle(title, y=.92, fontsize=title_fontsize)
                    
                if outpath is not None:
                    outfile = f'{analysis[0][0]}_{analysis[0][1]}_{d}{out_format}'
                    outfile = Path(str(outpath)) / outfile
                    plt.savefig(outfile)
                plt.show()


def plot_analysis_grid(analyses, models=None, datasets=None, contrasts=None,
                       outpath=None, out_format='.png',
                       figsize=(15,15), title_fontsize=16, **kwargs):
    ''' Plot images from same contrast and different analyses on same row,
        with one row per dataset. Useful for incremental analyses.
    Args 
        analyses: flattened analysis dictionary
        contrasts: subset of contrasts 
        models: subset of models
        datasets: subset of available datasets
        outpath: output path for images (if not defined, images are not saved)
        out_format: output format (defaults to png)
        figsize: figure size
        title_fontsize: fontsize of plot
    '''
    if models:
        analyses = [a for a in analyses if a[0][0] in models]
    if datasets:
        analyses = [a for a in analyses if a[0][1] in datasets]

    unique_model_names = list(set([a[0][0] for a in analyses]))
    unique_model_names.sort(key = len)
    unique_datasets = list(set(['_'.join(a[0][1:3]) for a in analyses]))

    model_list = []
    for analysis in analyses:
        cons = _get_con_names(analysis[1], contrasts=contrasts)
        for c, _ in cons:
            model_list.append((c, analysis))
                
    unique_contrasts = list(set([c for c, a in model_list]))
    
    for c in unique_contrasts:
        width = 0
        height = len(unique_datasets)
        for d in unique_datasets:
            ds_list = [el[1] for el in model_list 
                       if (el[0] == c) and (el[1][0][1] == d.split('_')[0]) and 
                       (el[1][0][2] == d.split('_')[1])]
            width = len(ds_list) if len(ds_list) > width else width
        
        f, axarr = plt.subplots(nrows=height, ncols=width, figsize=figsize)
        [axi.set_axis_off() for axi in axarr.ravel()]

        for m_idx, m in enumerate(unique_model_names):
            for d_idx, d in enumerate(unique_datasets):
                ds_list = [el[1] for el in model_list 
                           if (el[0] == c) and 
                           (el[1][0] == [m, d.split('_')[0], d.split('_')[1]])]
                if m_idx==0:
                    title = d.split('_')[0]
                else:
                    title = m[len(unique_model_names[m_idx-1]):]
                    if len(title) > 40:
                        title = title[:40] + '...'
                if (width > 1):
                    ax = axarr[d_idx, m_idx]
                else:
                    ax = axarr[d_idx]
                if len(ds_list) > 0:
                    analysis = ds_list[0]
                    plot_contrast(c, analysis[1], axes=ax, title=title, **kwargs)
        f.suptitle(c, y=.92, fontsize=title_fontsize)

        if outpath is not None:
            outfile = f'{analysis[0][0]}_{analysis[0][1]}_{d}{out_format}'
            outfile = Path(str(outpath)) / outfile
            plt.savefig(outfile)
        plt.show()


def plot_regressor(plot_type, df=None, model_dict=None, models=None, 
                   datasets=None, predictors=None, save_dir=None, 
                   save_format=None, color='.2', kde=True, rug=False, hist=True,
                   return_df=False, split_by='col', **kwargs):
    """ Plot distribution of predictor values for a set of models and datasets
    Args:
        plot_type: type of plot. one of 'distribution', 'timeseries'
        df: precomputed regressor values from base._extract_regressors
        model_dict: hierarchical dictionary of models, if df not provided
        models (optional): model subset list
        datasets (optional): datasets subset list
        predictors (optional): predictors subset list
        save_dir: output directory for plots (if None, does not save)
        save_format output format for plots (defaults to png)
        return_df (bool): whether to return the regressor df
        split_by: 'col', 'row' or 'hue'.
        color, kde, rug: arguments for seaborn distplot
        kwargs: kwargs for FacetGrid
        
    """
    # Load and subset data df
    if df is None:
        df = _extract_regressors(model_dict, models, datasets, predictors)
    else:
        if models:
            df = df[df['model'].isin(models)]
        if predictors:
            df = df[df['predictor'].isin(predictors)]
        if datasets:
            df = df[df['dataset'].isin(datasets)]

    df['ds_task'] = df['dataset'] + '_' + df['task']
    arg_dict = dict(zip(['col','row','hue'], [None, None, None]))
    arg_dict[split_by] = 'ds_task'

    if plot_type == 'distribution':
        color = None if split_by == 'hue' else color

    for m in df.model.unique():
        for p in m.split('+'): # TODO: Separate predictors in other way
            if p in df.predictor.unique():
                data_sub = df[(df['predictor']==p) & (df['model']==m)]
                g = sns.FacetGrid(data_sub, **arg_dict, **kwargs)
                if plot_type == 'distribution':
                    g = g.map(sns.distplot, 'values', 
                              color=color, kde=kde, rug=rug, hist=hist)
                elif plot_type == 'timeseries':
                    g = g.map_dataframe(sns.lineplot, x='scan', y='values')
                if split_by == 'col':
                    g.set_titles('{col_name}') 
                elif split_by == 'row':
                    g.set_titles('{row_name}')
                else:
                    g.add_legend()

                g.fig.suptitle(f'{p} [model: {_truncate_string(m, 70)}]', y=1.03)
                plt.show()
                if save_dir is not None:
                    save_format = save_format or '.png'
                    plot_id = p + m + save_format
                    save_path = Path(save_dir) / plot_id 
                    plt.savefig(save_path)
    if return_df:
        return df

def plot_metrics(agg_df, metrics, datasets=None, models=None, predictors=None,
                save_dir=None, save_format=None, sns_function='barplot', **kwargs):
    """ Plot summary metrics for predictors across different datasets in barplot
    Args:
        agg_df: aggregate metrics dataframe (from base.compute_metrics)
        metrics: column names for metrics to plot
        datasets: dataset subset
        models: model subset
        predictors: predictor subset
        save_dir: output directory (if None, does not save)
        save_format output format (defaults to png)
        kwargs: further arguments for seaborn plot
    """
    if models:
        agg_df = agg_df[agg_df['model'].isin(models)]
    if predictors:
        agg_df = agg_df[agg_df['predictor'].isin(predictors)]
    if datasets:
        agg_df = agg_df[agg_df['dataset'].isin(datasets)]      
    long_df = agg_df.melt(id_vars=['model', 'predictor', 'dataset', 'task'], 
                          value_vars=metrics)
    long_df['ds_task'] = long_df['dataset'] + '_' + long_df['task']
    if type(sns_function) == str:
        sns_function = getattr(sns, sns_function)
    for m in long_df.model.unique():
        for p in m.split('+'):
            if p in long_df.predictor.unique():
                sns.FacetGrid(long_df[(long_df['predictor'] == p) & (long_df['model'] == m)], 
                    col='variable', **kwargs) \
                    .map(sns_function, 'ds_task', 'value', 
                         order=long_df['ds_task'].unique()) \
                    .set_xticklabels(rotation=90) \
                    .set_xlabels('') \
                    .set_titles('{col_name}') \
                    .fig.suptitle(f'{p} [model: {_truncate_string(m, 50)}]', y=1.03)
                plt.show()
                if save_dir is not None:
                    save_format = save_format or '.png'
                    plot_id = p + m + save_format
                    save_path = Path(save_dir) / plot_id 
                    plt.savefig(save_path)

