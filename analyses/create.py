from pyns import Neuroscout
api = Neuroscout()

def create_models(name, predictors, confounds, datasets, transformations=None):
    """ Create predictor models for all the datasets for a list of predictors
        Args:
            predictors: list of predictor names
            confounds: list of confound names
            include_single_pred (bool): whether to create models including first predictor only.
            datasets: list of datasets from Neuroscout API (can be filtered to only apply to subset)
            transformation: see PyBids specification
    """    
    models = []
    # Remove transformations that don't apply to current predictor
    subset_transformations = None
    if transformations:
        subset_transformations = [t for t in transformations if t["Input"] in predictors]
        
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
