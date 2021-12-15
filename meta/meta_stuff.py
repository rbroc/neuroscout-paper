from pathlib import Path
import re
from collections import defaultdict
import json
import warnings
import os

warnings.filterwarnings("ignore")
from itertools import chain

import pymare as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

from nimare.dataset import Dataset
from nimare.io import convert_neurovault_to_dataset
from nimare.meta import ibma
import nibabel as nb
from pyns import Neuroscout
from nilearn.datasets import load_mni152_template
from nilearn.image import resample_to_img
from nilearn import masking
import nilearn.plotting as plotting


### Add download option to create_dataset and figure out how to go from nested structure to create dataset
def create_dataset(
    neuroscout_ids,
    name=None,
    **collection_kwargs
    ):
    """Download maps from NeuroVault and save them locally,
    along with relevant metadata from NeuroScout."""

    if collection_kwargs is None:
        collection_kwargs = {}

    ns = Neuroscout()
    nv_colls = {}

    skipped = []
    for id_ in neuroscout_ids:
        analysis = ns.analyses.get_analysis(id_)
        uploads = analysis.get_uploads(**collection_kwargs)

        if not uploads:
            skipped.append(id_)
        else:
            nv_colls[analysis.hash_id] = collections[-1]["collection_id"]
            
    if skipped:
        print(f"Skipped ids: {' '.join(skipped)} due to no matching collections")

    mask_img = masking.compute_gray_matter_mask(load_mni152_template())

    # Assuming the first analysis is prototypical, use that to pull out contrasts in model
    analysis = ns.analyses.get_analysis(neuroscout_ids[0])
    dummy_conditions = analysis.model['Steps'][0]['DummyContrasts']['Conditions']
    contrasts = {snake_to_camel(con): snake_to_camel(con) for con in dummy_conditions}
    if name is None:
        name = analysis.name

    dset = convert_neurovault_to_dataset(
        nv_colls,
        contrasts,
        img_dir=str(Path("./images").absolute()),
        mask=mask_img,
    )

    # Map neuroscout meta-data to dataset 

    #  Make directories if needed
    ds_dir = Path("./datasets")
    ds_dir.mkdir(exist_ok=True)

    # Add option to load at the beginning of this function
    dset.save(ds_dir / f"{name}_dset.pkl")

    return dset


def snake_to_camel(string):
    words = string.split('_')
    return words[0] + ''.join(word.title() for word in words[1:])


def analyze_dataset(
    name,
    dataset,
    plot=True,
    stat="z",
    analysis_kwargs=None,
    plot_kwargs=None,
    save_maps=True,
    ):

    if analysis_kwargs is None:
        analysis_kwargs = {}

    if plot_kwargs is None:
        plot_kwargs = {}


    ## Need to add slicing to run only on each effect at the time no?
    
    fxn = "effect_name" ### Need to see if I can get this from contrasts

    meta_result = ibma.DerSimonianLaird(resample=True).fit(dataset)

    eff_img = meta_result.get_map(stat)
    
    print(
        (
            f"meta-analyzing {len(meta_result.estimator.inputs_['id'])}"
        )
    )

    if save_maps:
        ma_dir = Path("./ma-maps")
        ma_dir.mkdir(exist_ok=True)
        filename = name[:200] + "-" + fxn
        ma_path = Path(".") / "ma-maps" / (filename + ".nii.gz")
        eff_img.to_filename(ma_path)

    if plot:
        defaults = dict(
            threshold=3.3,
            vmax=None,
            display_mode="z",
            cut_coords=[-24, -12, 0, 12, 24, 36, 48, 60],
        )
        defaults.update(plot_kwargs)

    return eff_img
