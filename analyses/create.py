from pyns import Neuroscout
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
        datasets: list of datasets from Neuroscout API (can be filtered to only apply to subset)
        transformation: see PyBids specification
    """        
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


def create_single_models(predictors, confounds, datasets, transformations=None):
    """ Create single predictor models for all the datasets in Neuroscout given a list of predictors
        Args:
            predictors: list of predictor names
            confounds: list of confound names
            include_single_pred (bool): whether to create models including first predictor only.
            datasets: list of datasets from Neuroscout API (can be filtered to only apply to subset)
            transformation: see PyBids specification
    """
    models = defaultdict(list)
    if confounds is None:
        confounds = []
 
    for predictor in predictors:
        # Remove transformations that don't apply to current predictor
        subset_transformations = None
        if transformations:
            subset_transformations = [t for t in transformations if predictor in t["Input"]]

        for dataset in datasets: 
            # NNDb images are de-trended and don't require confounds
            cf = confounds if dataset['name'] != 'NaturalisticNeuroimagingDatabase' else []

            for task in dataset['tasks']:
                try:
                    analysis = api.analyses.create_analysis(
                        name=predictor, # Analysis name will be set to predictor name
                        dataset_name=dataset['name'], 
                        tasks=task['name'],
                        predictor_names=cf + [predictor], # All predictors that will go into model
                        hrf_variables=[predictor], # Variables to convolve w/ HRF (only single predictor)
                        transformations=subset_transformations,
                        dummy_contrasts='hrf')
                        
                    models[predictor].append(
                        {'analysis': analysis, 'hash_id': analysis.hash_id, 'dataset': dataset['name'], 'task': task['name']}
                    )
                except ValueError as e:
                    print(f"Skipped: Dataset: {dataset['name']}, Task: {task['name']}, Predictors: {predictor}", e)
    return models


def create_set_models(predictors, confounds, name, datasets, transformations=None):
    """ Create same model with set of predictors + confounds for all datasets """
    models = []
    for dataset in datasets: 
        # NNDb images are de-trended and don't require confounds
        cf = confounds if dataset['name'] != 'NaturalisticNeuroimagingDatabase' else []
        for task in dataset['tasks']:
            try:
                analysis = api.analyses.create_analysis(
                    name=name, 
                    dataset_name=dataset['name'], 
                    predictor_names=cf + predictors,
                    tasks=task['name'],
                    transformations=transformations,
                    hrf_variables=predictors,
                    dummy_contrasts='hrf')
                models.append(
                        {'analysis': analysis, 'hash_id': analysis.hash_id, 'dataset': dataset['name'], 'task': task['name']}
                )
            except ValueError as e:
                    print(f"Skipped: Dataset: {dataset['name']}, Task: {task['name']}, Predictors: {predictors}", e)
    return models
