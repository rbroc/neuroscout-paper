""" Tools for use in drakh """
from pyns.models.utils import snake_to_camel
import nibabel as nib

RESULTS_DIR = Path('/media/hyperdrive/neuroscout-cli/output/')
DOWNLOAD_DIR = Path('/media/hyperdrive/neuroscout-cli/neurovault_dl/')


def find_image(con_name, hash_id, results_dir=None, load=True, target='t'):
    """ Given a base result_dirs, contrast_name, and hash_id, find stat_file"""
    if results_dir is None:
        results_dir = RESULTS_DIR

    name = snake_to_camel(con_name)
    images = results_dir / f"neuroscout-{hash_id}" / 'fitlins'

    ses_dirs = [a for a in images.glob('ses*') if a.is_dir()]
    if ses_dirs:  # If session, look for stat files in session folder
        images = images / ses_dirs[0]

    img_f = list((images).glob(
        f'*{name}_*stat-{target}_statmap.nii.gz'))
    if len(img_f) == 1:
        path = str(img_f[0])
        if load:
            return nib.load(path)
        else:
            return path
    else:
        return None