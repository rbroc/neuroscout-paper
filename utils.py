from pyns import Neuroscout
from copy import deepcopy
from nilearn._utils.glm import z_score
import scipy.stats as sps

import json

api = Neuroscout()


def load_collection(fname):
    """ Loads json collection and instantiates Analysis objects """
    with open(fname) as f:
        c_dict = json.load(f)

    for set_name, analyses in c_dict.items():
        for an in analyses:
            an['analysis'] = api.analyses.get_analysis(an['hash_id'])
    return c_dict


def dump_collection(c_dict, fname):
    """ Dump dictionary to json, removing analysis object """
    for set_name, analyses in c_dict.items():
        for an in analyses:
            an.pop('analysis')

    with open(fname, 'w') as f:
        json.dump(c_dict, f)


def t_to_z(t_map, dof):
    # From nilearn.glm
    old_shape = t_map.get_fdata().shape
    data = t_map.get_fdata().flatten()
    new_img = deepcopy(t_map)
    baseline = 0
    p_value = sps.t.sf(data, dof)
    
    one_minus_pvalues = sps.t.cdf(data, dof)
        
    # Avoid inf values kindly supplied by scipy.
    z_scores = z_score(p_value, one_minus_pvalue=one_minus_pvalues)
    
    z_scores = z_scores.reshape(old_shape)
    
    new_img.get_fdata()[:] = z_scores
    
    return new_img