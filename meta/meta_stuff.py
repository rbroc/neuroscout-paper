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


def create_dset(
    effect,
    neuroscout_ids,
    map_types=["variance", "univariate-beta map"],
    name=None,
    overwrite=False,
    camel_case=True,
    **collection_kwargs
):
    """Download maps from NeuroVault and save them locally,
    along with relevant metadata from NeuroScout."""
    
    if collection_kwargs is None:
        collection_kwargs = {}

    #  Make directories if needed
    analysis_dir = Path("./analyses")
    analysis_dir.mkdir(exist_ok=True)
    ds_dir = Path("./datasets")
    ds_dir.mkdir(exist_ok=True)

    ns = Neuroscout()
    nv_colls = {}

    skipped = []
    for id_ in neuroscout_ids:

        filename = analysis_dir / f"{id_}.json"
        if filename.exists() and not overwrite:
            analysis = json.load(open(filename, "r"))
        else:
            analysis = ns.analyses.get_full(id_)
            json.dump(analysis, open(filename, "w"))

        if not analysis.get("neurovault_collections"):
            continue
           
        collections = []
        for a in analysis["neurovault_collections"]:
            for k, val in collection_kwargs.items():
                if a[k] != val:
                    break
            else:
                collections.append(a)

        if not collections:
            skipped.append(id_)
        else:
            nv_colls[analysis["hash_id"]] = collections[-1]["collection_id"]
            
    if skipped:
        print(f"Skipped ids: {' '.join(skipped)} due to no matching collections")

    mask_img = masking.compute_gray_matter_mask(load_mni152_template())

    if camel_case:
        effect_name_mapping = {snake_to_camel(effect): snake_to_camel(effect)}
    else:
        effect_name_mapping = {effect: effect}

    dset = convert_neurovault_to_dataset(
        nv_colls,
        effect_name_mapping,
        img_dir=str(Path("./images").absolute()),
        mask=mask_img,
    )

    dset.save(ds_dir / f"{effect}_dset.pkl")

    return dset


def snake_to_camel(text):
    """Convert snake_case to camelCase--needed when downloading some NV images."""
    cc = "".join(x.capitalize() or "_" for x in text.split("_"))
    cc = re.sub("([\.\-])(\w)", lambda x: x.group(1) + x.group(2).capitalize(), cc)
    return cc[0].lower() + cc[1:]


def process_set(
    name,
    effects=None,
    json_data=None,
    analyze=True,
    analysis_method="nimare",
    download=True,
    plot=True,
    camel_case=True,
    overwrite=False,
    stat="z",
    exclude=None,
    analysis_kwargs=None,
    plot_kwargs=None,
    save_maps=True,
    collection_kwargs=None
):

    if analysis_kwargs is None:
        analysis_kwargs = {}

    if plot_kwargs is None:
        plot_kwargs = {}

    # Retrieve data if needed
    if json_data is not None:
        if isinstance(json_data, str):
            json_data = requests.get(json_data).json()

        # Build dict of effects --> analysis IDs
        effects = defaultdict(list)
        hash_lookup = {}
        for c, dsets in json_data.items():
            hash_lookup[c] = {}
            for ds, children in dsets.items():
                effects[c].extend(children.values())
                for movie, hash_ in children.items():
                    hash_lookup[c][hash_] = movie

    # Download maps
    dsets = {}
    if download:
        for effect, ids in effects.items():
            print(f"Downloading maps for effect '{effect}'...")
            dsets[effect] = create_dset(
                effect, ids, name=name, overwrite=overwrite, camel_case=camel_case, **collection_kwargs
            )

    else:
        dsets = {effect: Dataset.load(f"./datasets/{effect}_dset.pkl") for effect in effects}

    if analyze:
        fx_names = sorted(list(effects.keys()))
        if exclude is not None:
            analysis_dir = Path("./analyses")
            def filter_analyses(ids):
                filtered_ids = []
                for id_ in ids:
                    a_id = re.search("study-(\w{5})-.*", str(id_)).group(1)
                    filename = analysis_dir / f"{a_id}.json"
                    # Ugh
                    if not filename.exists():
                        filename = analysis_dir / f"{a_id} (Case Conflict).json"
                    analysis = json.load(open(filename, "r"))
                    if analysis["model"]["Input"]["Task"][0] not in exclude:
                        filtered_ids.append(id_)
                return filtered_ids

            dsets = {
                effect: dset.slice(filter_analyses(dset.ids)) for effect, dset in dsets.items()
            }

        meta_results = {
            effect: ibma.DerSimonianLaird(resample=True).fit(dset) for effect, dset in dsets.items()
        }
        eff_imgs = [meta_results[fx].get_map(stat) for fx in sorted(list(meta_results.keys()))]
        for fx_name in fx_names:
            print(
                (
                    f"meta-analyzing {len(meta_results[fx_name].estimator.inputs_['id'])}"
                    f" studies for effect: {fx_name}"
                )
            )

        if save_maps:
            ma_dir = Path("./ma-maps")
            ma_dir.mkdir(exist_ok=True)
            for i, fxn in enumerate(fx_names):
                ma_path = Path(".") / "ma-maps" / (name + "-" + fxn + ".nii.gz")
                eff_imgs[i].to_filename(ma_path)

        if plot:
            # Big ol' plot
            n_eff = len(fx_names)
            fig, axes = plt.subplots(n_eff, 1, figsize=(14, n_eff * 2.5))
            if n_eff == 1:
                axes = [axes]
            defaults = dict(
                threshold=3.3,
                vmax=None,
                display_mode="z",
                cut_coords=[-24, -12, 0, 12, 24, 36, 48, 60],
            )
            defaults.update(plot_kwargs)
            for i, img in enumerate(eff_imgs):
                plotting.plot_stat_map(img, axes=axes[i], title=fx_names[i], **defaults)

    return eff_imgs
