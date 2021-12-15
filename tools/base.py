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
