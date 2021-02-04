from pathlib import Path
import re
from collections import defaultdict
import json
import warnings
warnings.filterwarnings('ignore')
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


def create_dset(effect, neuroscout_ids,
                map_types=['variance', 'univariate-beta map'],
                name=None, overwrite=False, camel_case=True):
    """Download maps from NeuroVault and save them locally,
    along with relevant metadata from NeuroScout."""

    metadata = {
        'effect': effect,
        'neuroscout_ids': neuroscout_ids,
        'analyses': {}
    }

    #  Make directories if needed
    analysis_dir = Path('./analyses')
    analysis_dir.mkdir(exist_ok=True)
    ds_dir = Path('./datasets')
    ds_dir.mkdir(exist_ok=True)
    # ma_dir = Path('./ma-maps')
    # ma_dir.mkdir(exist_ok=True)

    ns = Neuroscout()
    nv_colls = {}

    for id_ in neuroscout_ids:
        
        filename = analysis_dir / f'{id_}.json'
        if filename.exists() and not overwrite:
            analysis = json.load(open(filename, 'r'))
        else:
            analysis = ns.analyses.get_full(id_)
            json.dump(analysis, open(filename, 'w'))

        if not analysis.get('neurovault_collections'):
            continue

        nv_colls[analysis['hash_id']] = analysis['neurovault_collections'][-1]['collection_id']
    
    mask_img = masking.compute_gray_matter_mask(load_mni152_template())

    dset = convert_neurovault_to_dataset(
            nv_colls,
            snake_to_camel(effect),
            img_dir=Path('./images').absolute(),
            mask=mask_img,
    )

    dset.save(ds_dir / f"{effect}_dset.pkl")

    return dset


def snake_to_camel(text):
    """Convert snake_case to camelCase--needed when downloading some NV images."""
    cc = ''.join(x.capitalize() or '_' for x in text.split('_'))
    cc = re.sub('([\.\-])(\w)', lambda x: x.group(1) + x.group(2).capitalize(), cc)
    return cc[0].lower() + cc[1:]


def meta_analyze(effect, ids=None, name=None, method='DL', camel_case=True, measure=None, exclude=None,
                 stat='z'):
    # Read images into arrays
    eff_name = snake_to_camel(effect) if camel_case else effect
    beta_maps = sorted(list(Path('.').glob(f'images2/*{eff_name}*effect*')))
    var_maps = sorted(list(Path('.').glob(f'images2/*{eff_name}*variance*')))

    # Optionally filter candidate images on analysis IDs
    if ids is not None:
        patt = f"analysis-({'|'.join(ids)})"
        beta_maps = [bm for bm in beta_maps if re.search(patt, str(bm))]
        var_maps = [vm for vm in var_maps if re.search(patt, str(vm))]

    if exclude is not None:
        analysis_dir = Path('./analyses')
        def filter_analyses(f):
            a_id = re.search('analysis-(\w{5})', str(f)).group(1)
            filename = analysis_dir / f'{a_id}.json'
            # Ugh
            if not filename.exists():
                filename = analysis_dir / f'{a_id} (Case Conflict).json'
            analysis = json.load(open(filename, 'r'))
            return analysis['model']['Input']['Task'] not in exclude
        beta_maps = [f for f in beta_maps if filter_analyses(f)]

    # drop analyses that don't have both maps
    maps = defaultdict(list)
    [maps[str(fn.name)[9:14]].append(fn) for fn in (beta_maps + var_maps)]
    maps = list(chain(*[m for m in maps.values() if len(m) == 2]))
    beta_maps = [m for m in maps if 'effect' in str(m)]
    var_maps = [m for m in maps if 'variance' in str(m)]
    kept_ids = [str(m.name)[9:14] for m in beta_maps]

    template = load_mni152_template()

    # Resample to standard MNI152 template
    beta_imgs = [resample_to_img(nb.load(img), template, clip=True, interpolation='linear')
                 for img in beta_maps]
    var_imgs = [resample_to_img(nb.load(img), template, clip=True, interpolation='linear')
                for img in var_maps]

    print(f"Meta-analyzing {len(beta_imgs)} studies for effect '{effect}'.")
    # Mask images and set up arrays for pymare
    mask_img = masking.compute_gray_matter_mask(template)
    m = masking.apply_mask(beta_imgs, mask_img)
    v = masking.apply_mask(var_imgs, mask_img)

    # Get sample sizes
    prefix = f'{name}_' if name is not None else ''
    md_path = Path('.') / 'metadata' / f'{prefix}{effect}.json'
    effect_md = json.load(open(md_path, 'r'))
    n = []
    for f in beta_maps:
        match = re.search('analysis-(.*?)_', str(f))
        n_subs = effect_md['analyses'][match.group(1)]['n_subjects']
        n.append(n_subs)
    n = np.array(n)[:, None]

    if measure is not None:
        # pymare's converters expect sds, not sampling variances
        sd = np.sqrt(v * n)

        # Convert means and variances to Hedges' g
        esc = pm.OneSampleEffectSizeConverter(m=m, sd=sd, n=n)
        dataset = esc.to_dataset(measure=measure)
    else:
        dataset = pm.Dataset(y=m, v=v, n=n)

    # Run meta-regression
    results = pm.meta_regression(data=dataset, method=method)
    
    if stat == 'z':
        stat_maps = results.get_fe_stats()['z']
    else:
        stats = results.get_heterogeneity_stats()
        stat_maps = stats.get(stat)
    
    # Unmask and return as nibabel image
    vox_z = np.nan_to_num(stat_maps.ravel())
    ma_img = masking.unmask(vox_z,  mask_img)

    return ma_img


def process_set(name, effects=None, json_data=None, analyze=True, download=True,
                plot=True, camel_case=True, overwrite=False, stat='z',
                exclude=None, analysis_kwargs=None, plot_kwargs=None, save_maps=True):

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
        for c, dsets in json_data.items():
            for ds, children in dsets.items():
                effects[c].extend(children.values())

    # Download maps
    dsets = {}
    if download:
        for effect, ids in effects.items():
            print(f"Downloading maps for effect '{effect}'...")
            dsets[effect] = create_dset(effect, ids, name=name, overwrite=overwrite, camel_case=camel_case)

    else:
        dsets = {effect: Dataset.load(f'./datasets/{effect}_dset.pkl') for effect in effects}

    if analyze:
        if exclude is not None:
            analysis_dir = Path('./analyses')
            def filter_analyses(ids):
                filtered_ids = []
                for id_ in ids:
                    a_id = re.search('study-(\w{5})-.*', str(id_)).group(1)
                    filename = analysis_dir / f'{a_id}.json'
                    # Ugh
                    if not filename.exists():
                        filename = analysis_dir / f'{a_id} (Case Conflict).json'
                    analysis = json.load(open(filename, 'r'))
                    if analysis['model']['Input']['Task'] not in exclude:
                        filtered_ids.append(id_)
                return filtered_ids
            dsets = {
                effect: dset.slice(filter_analyses(dset.ids)) for effect, dset in dsets.items()
            }

        meta_results = {
            effect: ibma.DerSimonianLaird().fit(dset) for effect, dset in dsets.items()
        }
        eff_imgs = [meta_results[fx].get_map(stat) for fx in sorted(list(meta_results.keys()))]
        fx_names = sorted(list(effects.keys()))
        # Meta-analyze all the things
        eff_imgs2 = [meta_analyze(fxn, effects[fxn], name=name, camel_case=camel_case,
                                 stat=stat, exclude=exclude, **analysis_kwargs)
                    for fxn in fx_names]

        if save_maps:
            ma_dir = Path('./ma-maps')
            ma_dir.mkdir(exist_ok=True)
            for i, fxn in enumerate(fx_names):
                ma_path = Path('.') / 'ma-maps' / (name + '-' + fxn + ".nii.gz")
                eff_imgs[i].to_filename(ma_path)

        if plot:
            # Big ol' plot
            n_eff = len(fx_names)
            fig, axes = plt.subplots(n_eff, 1, figsize=(14, n_eff*2.5))
            if n_eff == 1:
                axes = [axes]
            defaults = dict(threshold=3.3, vmax=None, display_mode='z', cut_coords=[-24, -12, 0, 12, 24, 36, 48, 60])
            defaults.update(plot_kwargs)
            for i, img in enumerate(eff_imgs):
                plotting.plot_stat_map(img, axes=axes[i], title=fx_names[i], **defaults)

    return eff_imgs
