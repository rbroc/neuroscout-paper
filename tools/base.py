import nibabel as nib
from pathlib import Path
from pyns import Neuroscout
import json
from copy import deepcopy
import collections
import pandas as pd
import numpy as np
api = Neuroscout()

RESULTS_DIR = Path('/media/hyperdrive/neuroscout-cli/output/')
DOWNLOAD_DIR = Path('/media/hyperdrive/neuroscout-cli/neurovault_dl/')

def snake_to_camel(string):
    words = string.split('_')
    return words[0] + ''.join(word.title() for word in words[1:])


def find_image(con_name, hash_id, results_dir=None, load=True, target='t'):
    """ Given a base result_dirs, contrast_name, and hash_id, find stat_file"""
    if results_dir is None:
        results_dir = RESULTS_DIR

    name = snake_to_camel(con_name)
    images = results_dir / f"neuroscout-{hash_id}" / 'fitlins'

    ses_dirs = [a for a in images.glob('ses*') if a.is_dir()]
    if ses_dirs:  # If session, look for stat files in session folder
        images = images / ses_dirs[0]

    img_f = list((images).glob(
        f'*{name}_*stat-{target}_statmap.nii.gz'))
    if len(img_f) == 1:
        path = str(img_f[0])
        if load:
            return nib.load(path)
        else:
            return path
    else:
        return None

def reverse(di, check_status=False):
    rev = {}
    for pred in di.keys():
        for ds, dn in di[pred].items():
            if (not check_status) or (not dn.get_uploads()):
                if ds in rev:
                    rev[ds] = ' '.join([rev[ds], dn.hash_id])
                else:
                    rev[ds] = dn.hash_id
    return rev
    

def flatten_collection(d, parent_key=None):
    """ Flattens nested dictionary by collecting parent keys in list and
     pairing as tuple. """
    
    sort = False
    if parent_key is None:
        parent_key = []
        sort = True
    
    items = []
    for k, v in d.items():
        new_key = parent_key + [k]
        if isinstance(v, collections.MutableMapping):
            flat = flatten_collection(v, new_key)
            if flat:
                items += flat
        else:
            items.append((new_key, v))
            
    
    if sort:
        items = sorted(items, key=lambda x: x[0])
    return items


def _change_leaves(di, mode='load'):
    for k, v in di.items():
        if isinstance(v, dict):
            _change_leaves(v, mode=mode)
        else:
            if mode == 'load':
                di[k] = api.analyses.get_analysis(v)
            else:
                di[k] = v.hash_id


def load_collection(fname):
    """ Loads json and reads back in analysis object using hash_id """
    with open(fname) as f:
        c_dict = json.load(f)

    _change_leaves(c_dict)
    return c_dict


def dump_collection(di, fname):
    """ Dump dictionary to json, with hash_ids as leaves """
    di = deepcopy(di)
    _change_leaves(di, mode='dump')
    with open(fname, 'w') as f:
        json.dump(di, f)


def _extract_regressors(model_dict, models=None, datasets=None, predictors=None):
    """ Plot distributions of variables in all models
    Args:
        model_dict: hierarchical dictionary of models
        models (list): model subset
        datasets: dataset subset
        predictors: predictor subset
    """
    all_pred = pd.DataFrame()
    for model_name, ds_dict in model_dict.items():
        pred_df = pd.DataFrame()
        for ds, task_dict in ds_dict.items():
            if datasets and ds not in datasets:
                continue
            for task, model in task_dict.items():
                dmat = pd.read_csv(model.get_design_matrix()[0], ',')
                for pred in model_name.split('+'):
                    if (predictors and pred not in predictors) or models and (model_name not in models):
                        continue
                    pred_df['values'] = dmat[pred]
                    pred_df['predictor'] = pred
                    pred_df['dataset'] = ds
                    pred_df['task'] = task
                    pred_df['model'] = model_name
                    all_pred = pd.concat([all_pred, pred_df], axis=0)
    all_pred['scan'] = all_pred.groupby(['predictor', 'dataset', 'task', 'model']).cumcount()
    return all_pred


def compute_metrics(model_dict=None, df=None, aggfunc=[np.mean, np.max, np.min, np.std],
                              models=None, datasets=None, predictors=None, return_df=False):
    """ Compute summary metrics for regressors
    Args:
        model_dict: hierarchical dictionary of models
        df: precomputed regressor values from base._extract_regressors
        aggfunc: list of summary functions passed to pd.pivot_table (e.g. np.mean)
        models (optional): model subset
        datasets (optional): dataset subset
        predictors (optional): predictor subset
        return_df (bool): whether to return regressor df if computed within the function
    """
    if df is None:
        df = _extract_regressors(model_dict, models, datasets, predictors)
    else:
        if models:
            df = df[df['model'].isin(models)]
        if predictors:
            df = df[df['predictor'].isin(predictors)]
        if datasets:
            df = df[df['dataset'].isin(datasets)]   
    agg_df = df.pivot_table(values='values', 
                            index=['model', 'predictor', 'dataset', 'task'], 
                            aggfunc=aggfunc).reset_index()
    agg_df.columns = agg_df.columns.droplevel(1)
    if return_df:
        return df, agg_df
    else:
        return agg_df


def rename_collection_keys(collection_path, dictionary=None):
    ''' Renames keys of json collection. If dictionary is not provided, 
    replaces current key with Analysis.name. If dictionary is provided, 
    Replaces keys with corresponding value in the dictionary '''
    models = load_collection(collection_path)
    new_dict = {}
    for n, ds in models.items():
        pop_keys = []
        for ds_name, d in ds.items():
            if d == {}:
                pop_keys.append(ds_name)
            else:
                if dictionary is None:
                    analysis_name = flatten_collection(d)[0][1].name
                else:
                    try:
                        analysis_name = dictionary[n]
                    except:
                        analysis_name = n
        if pop_keys:
            for k in pop_keys:
                ds.pop(k)
        new_dict[analysis_name] = ds
        analysis_name = None
    dump_collection(new_dict, collection_path)