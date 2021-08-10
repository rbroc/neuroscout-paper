import warnings
import time
from .base import flatten_collection
from pyns import Neuroscout
api = Neuroscout()

ALL_DATASETS = {} 
for dataset in api.datasets.get():
    ALL_DATASETS[dataset['name']] = {
        'id':  dataset['id'],
        'tasks': [task['name'] for task in dataset['tasks']]}


def _get_datasets(datasets):
    if datasets is None:
        datasets = ALL_DATASETS
    elif isinstance(datasets, list):
        datasets = {d:v for d,v in ALL_DATASETS.items() if d in datasets}
        
    return datasets


def create_incremental_models(predictors, confounds, include_single_pred=False, 
                              datasets=None, transformations=None):
    """ Create incremental models for all the datasets in Neuroscout given a list of predictors 
    Args:
        predictors: list of predictor names, or list of lists if steps add multiple at once
        confounds: list of confound names
        include_single_pred (bool): whether to create models including first predictor only.
        datasets: dictionary with dataset names as keys, dataset id as values
        transformation: see PyBids specification
    """
    datasets = _get_datasets(datasets)
        
    # Listfy list of lists
    predictors = [p if isinstance(p, list) else [p] for p in predictors]        

    models = {}
    for i in range(len(predictors)):
        if i == 0 and not include_single_pred:
            continue

        preds = [item for sublist in predictors[:i+1] for item in sublist] 

        model_name = '+'.join(preds)
        models[model_name] = {}
        
        if transformations:
            subset_trans = [t for t in transformations if set(preds) >= set(t["Input"])]
        else:
            subset_trans = None
            
        for ds_name, ds_values in datasets.items(): 
            d_tasks = ds_values['tasks'] 
            models[model_name][ds_name] = {}
            for task_id in d_tasks:
                try:
                    models[model_name][ds_name][task_id] = api.analyses.create_analysis(
                        name=model_name, 
                        dataset_name=ds_name, 
                        predictor_names=confounds + preds,
                        task=task_id,
                        transformations=subset_trans,
                        hrf_variables=preds,
                        dummy_contrasts='hrf') 
                except ValueError as e:
                    print(f"Dataset: {ds_name}, Predictors: {preds}", e)
    return models


def create_single_models(predictors, confounds=None, control=None, datasets=None, transformations=None):
    """ Create single predictor models for all the datasets in Neuroscout given a list of predictors
        Args:
            predictors: list of predictor names
            confounds: list of confound names
            control: list of control variables that need to be convolved
            include_single_pred (bool): whether to create models including first predictor only.
            datasets: dictionary with dataset names as keys, dataset id as values
            transformation: see PyBids specification
    """
    models = {}
    if confounds is None:
        confounds = []
 
    datasets = _get_datasets(datasets)

    for predictor in predictors:
        models[predictor] = {}
        preds = [predictor] if control is None else [predictor] + control
        for ds_name, ds_values in datasets.items(): 
            models[predictor][ds_name] = {}
            if transformations:
                subset_transformations = [t for t in transformations if predictor in t["Input"]]
            else:
                subset_transformations = None
            for task_name in ds_values['tasks']:
                try:
                    models[predictor][ds_name][task_name] = api.analyses.create_analysis(
                        name=predictor, 
                        dataset_name=ds_name, 
                        predictor_names=confounds + preds,
                        tasks=task_name,
                        transformations=subset_transformations,
                        hrf_variables=preds,
                        dummy_contrasts='hrf')
                except ValueError as e:
                    print(f"Dataset: {ds_name}, Predictors: {predictor}", e)
    return models


def create_set_models(predictors, confounds, name, transformations=None, datasets=None):
    """ Create same model with set of predictors + confounds for all datasets """
    models = {}
    if datasets is None:
        datasets = ALL_DATASETS

    for ds_name, ds_values in datasets.items(): 
        models[ds_name] = {}
        for task_id in ds_values['tasks'] : #loop over tasks
            try: # create an analysis
                models[ds_name][task_id] = api.analyses.create_analysis(
                    name=name, 
                    dataset_name=ds_name, 
                    predictor_names=confounds + predictors,
                    task=task_id,
                    transformations=transformations,
                    hrf_variables=predictors,
                    dummy_contrasts='hrf')
            except ValueError as e: # print out error if build fails
                print(f"Dataset: {ds_name}, Predictors: {str(predictors)}", e)
    return models
