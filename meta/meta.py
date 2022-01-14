from pathlib import Path
import warnings
from nimare.dataset import Dataset
from nimare.io import convert_neurovault_to_dataset
from nimare.meta import ibma
from pyns import Neuroscout
from pyns.models.utils import snake_to_camel
from nilearn.datasets import load_mni152_template
from nilearn import masking
from nilearn import plotting as niplt
import pandas as pd

warnings.filterwarnings("ignore")

def create_dataset(neuroscout_ids, img_dir=None, **collection_kwargs):
    """Download maps from NeuroVault and save them locally,
    along with relevant metadata from NeuroScout.
    Args:
        neuroscout_ids: list of neuroscout_ids
        img_dir: optional directory to cached downloaded images,
        collection_kwargs: filters to use for selecting NeuroVault collection
    Returns:
        dataset: NiMare dataset object
    """
    if collection_kwargs is None:
        collection_kwargs = {}

    ns = Neuroscout()
    nv_colls = {}
    annotations = []
    skipped = []
    for id_ in neuroscout_ids:
        analysis = ns.analyses.get_analysis(id_)
        upload = analysis.get_uploads(**collection_kwargs, select='latest')

        if not upload:
            skipped.append(id_)
        else:
            nv_colls[analysis.hash_id] = upload[0]['collection_id']
            task_name = analysis.model['Input']['Task']
            n_subjects = len(analysis.model['Input']['Subject'])
            if len(task_name) == 1:
                task_name = task_name[0]
            dataset_name = ns.datasets.get(analysis.dataset_id)['name']
            annotations.append({
                'study_id': f'study-{analysis.hash_id}',
                'dataset': dataset_name, 
                'task': task_name,
                'n_subjects': n_subjects
                }
            )   
            
    if skipped:
        print(f"Skipped ids: {' '.join(skipped)} due to no matching collections")

    if not nv_colls:
        raise ValueError("No matching collections for any analyses")

    mask_img = masking.compute_gray_matter_mask(load_mni152_template())

    # Assuming the first analysis is prototypical, use that to pull out contrasts in model
    dummy_conditions = analysis.model['Steps'][0]['DummyContrasts']['Conditions']
    contrasts = {snake_to_camel(con): snake_to_camel(con) for con in dummy_conditions}

    dset = convert_neurovault_to_dataset(
        nv_colls,
        contrasts,
        img_dir=img_dir,
        mask=mask_img,
    )

    # add dataset and task annotations to dataset
    annotations =  pd.DataFrame(annotations)
    dset.annotations = dset.annotations.merge(annotations)

    # Map neuroscout meta-data to dataset 
    return dset


def _slice_meta_plot_save(
    dataset, analysis_name, contrast_name, exclude_datasets=None, exclude_tasks=None,
    save_maps=True, plot=True, stat='z', **plot_kwargs):
    """ Run meta-analysis on dataset on a specific contrast """

    # Slice
    dataset = dataset.slice(
       ids=dataset.images.id[dataset.images.contrast_id == contrast_name])

    if exclude_datasets:
        dataset = dataset.slice(
            ids=dataset.annotations.id[ \
                dataset.annotations.dataset.isin(exclude_datasets) == False])

    if exclude_tasks:
        dataset = dataset.slice(
            ids=dataset.annotations.id[ \
                dataset.annotations.task.isin(exclude_tasks) == False])

    # Run
    meta_result = ibma.DerSimonianLaird(resample=True).fit(dataset)
    eff_img = meta_result.get_map(stat)
    print(
        f"meta-analyzing {len(meta_result.estimator.inputs_['id'])} for {contrast_name}")

    if save_maps:
        ma_dir = Path("./ma-maps")
        ma_dir.mkdir(exist_ok=True)
        filename = analysis_name[:200] + "-" + contrast_name
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
        niplt.plot_stat_map(eff_img, title=contrast_name, **defaults)

    return eff_img


def analyze_collection(
    neuroscout_ids,
    analysis_name=None,
    contrasts=None,
    exclude_datasets=None,
    exclude_tasks=None,
    force_recreate=False,
    plot=True,
    stat="z",
    analysis_kwargs=None,
    collection_kwargs=None,
    plot_kwargs=None,
    save_maps=True,
    ):

    if analysis_kwargs is None:
        analysis_kwargs = {}

    if collection_kwargs is None:
        collection_kwargs = {}

    if plot_kwargs is None:
        plot_kwargs = {}

    if isinstance(neuroscout_ids[0], dict):
        neuroscout_ids = [a['hash_id'] for a in neuroscout_ids]

    # Make directories if needed
    ds_dir = Path("./datasets")
    ds_dir.mkdir(exist_ok=True)

    # Default to first analysis name in Neuroscout
    if analysis_name is None:
        analysis_name = Neuroscout().analyses.get(neuroscout_ids[0])['name']

    save_path = ds_dir / f"{analysis_name}_dset.pkl"

    if save_path.exists() and not force_recreate:
        dataset = Dataset.load(str(save_path))
    else:
        dataset = create_dataset(neuroscout_ids, 
            img_dir=str(Path("./images").absolute()),
            **collection_kwargs)
        dataset.save(save_path)

    # If no contrast name is specified, ignore or extract from dataset?
    if not contrasts:
        contrasts = dataset.images.contrast_id.unique()
        if len(dataset.images.contrast_id.unique()) > 1:
            warnings.warn("More than one contrast found")

    eff_imgs = []
    for contrast_name in contrasts:
        eff_imgs.append(
            _slice_meta_plot_save(
                dataset, analysis_name, contrast_name, save_maps=save_maps,
                exclude_datasets=exclude_datasets, exclude_tasks=exclude_tasks,
                plot=plot, stat=stat, **plot_kwargs)
        )    

    return eff_imgs
