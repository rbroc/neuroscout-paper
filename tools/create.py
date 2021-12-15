from pyns import Neuroscout
from utils import _get_datasets
from collections import defaultdict
api = Neuroscout()


# Note: This and the following function create multiple analysis sets for multiple predictor
# In a future refactor, consider making all functions only operate on a single predictor, 
# and users can combine them as they wish
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

    models = defaultdict(list)
    for i in range(len(predictors)):
        if i == 0 and not include_single_pred:
            continue

        preds = [item for sublist in predictors[:i+1] for item in sublist] 
        model_name = '+'.join(preds)
        
        # Modify transformations to those that apply to predictor
        if transformations:
            subset_transformations = [t for t in transformations if set(preds) >= set(t["Input"])]
        else:
            subset_transformations = None

        for ds_name, ds_values in datasets.items(): 
            cf = confounds if ds_name != 'NaturalisticNeuroimagingDatabase' else []
            for task_name in ds_values['tasks'] :
                try:
                    analysis = api.analyses.create_analysis(
                        name=model_name, 
                        dataset_name=ds_name, 
                        predictor_names=cf + preds,
                        tasks=task_name,
                        transformations=subset_transformations,
                        hrf_variables=preds,
                        dummy_contrasts='hrf') 
                    models[model_name].append(
                        {'analysis': analysis, 'hash_id': analysis.hash_id, 'dataset': ds_name, 'task': task_name}
                    )
                except ValueError as e:
                    print(f"Dataset: {ds_name}, Task: {task_name}, Predictors: {preds}", e)
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
    models = defaultdict(list)
    if confounds is None:
        confounds = []
 
    datasets = _get_datasets(datasets)

    for predictor in predictors:
        preds = [predictor] if control is None else [predictor] + control
        for ds_name, ds_values in datasets.items(): 
            cf = confounds if ds_name != 'NaturalisticNeuroimagingDatabase' else []

            if transformations:
                subset_transformations = [t for t in transformations if predictor in t["Input"]]
            else:
                subset_transformations = None
            for task_name in ds_values['tasks']:
                try:
                    analysis = api.analyses.create_analysis(
                        name=predictor, 
                        dataset_name=ds_name, 
                        predictor_names=cf + preds,
                        tasks=task_name,
                        transformations=subset_transformations,
                        hrf_variables=preds,
                        dummy_contrasts='hrf')

                    models[predictor].append(
                        {'analysis': analysis, 'hash_id': analysis.hash_id, 'dataset': ds_name, 'task': task_name}
                    )
                except ValueError as e:
                    print(f"Dataset: {ds_name}, Task: {task_name}, Predictors: {preds}", e)
    return models


def create_set_models(predictors, confounds, name, transformations=None, datasets=None):
    """ Create same model with set of predictors + confounds for all datasets """
    models = []
    datasets = _get_datasets(datasets)

    for ds_name, ds_values in datasets.items(): 
        cf = confounds if ds_name != 'NaturalisticNeuroimagingDatabase' else []
        for task_name in ds_values['tasks']:
            try:
                analysis = api.analyses.create_analysis(
                    name=name, 
                    dataset_name=ds_name, 
                    predictor_names=cf + predictors,
                    tasks=task_name,
                    transformations=transformations,
                    hrf_variables=predictors,
                    dummy_contrasts='hrf')
                models.append(
                    {'analysis': analysis, 'hash_id': analysis.hash_id, 'dataset': ds_name, 'task': task_name}
                )
            except ValueError as e:
                    print(f"Dataset: {ds_name}, Task: {task_name}, Predictors: {predictors}", e)
    return models
