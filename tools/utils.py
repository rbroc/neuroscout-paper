from pyns import Neuroscout
import json

api = Neuroscout()


def load_collection(fname):
    """ Loads json collection and instantiates Analysis objects """
    with open(fname) as f:
        c_dict = json.load(f)

    for set_name, analyses in c_dict:
        for an in analyses:
            an['analysis'] = api.analyses.get_analysis(an['hash_id'])
    return c_dict


def dump_collection(c_dict, fname):
    """ Dump dictionary to json, removing analysis object """
    for set_name, analyses in c_dict:
        for an in analyses:
           an.pop('analysis')

    with open(fname, 'w') as f:
        json.dump(di, f)


ALL_DATASETS = {} 
for dataset in api.datasets.get():
    ALL_DATASETS[dataset['name']] = {
        'id':  dataset['id'],
        'tasks': [task['name'] for task in dataset['tasks']]}


def _get_datasets(datasets):
    """ Given a list of dataset names, get dataset IDs """
    if datasets is None:
        datasets = ALL_DATASETS
    elif isinstance(datasets, list):
        datasets = {d:v for d,v in ALL_DATASETS.items() if d in datasets}
        
    return datasets