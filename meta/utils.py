from nilearn import datasets, surface, plotting
from nilearn.plotting.surf_plotting import _colorbar_from_array, get_cmap
from matplotlib import colorbar, colors
from matplotlib import pyplot as plt
import numpy as np

fsaverage = datasets.fetch_surf_fsaverage()

def plot_surfaces(imgs, 
                  img_names, 
                  select=None, 
                  vmax=10, 
                  threshold=3.29,
                  cmap='cold_hot',
                  figsize=None,
                  title_fontsize=16,
                  title_kwargs={},
                  surface_kwargs={},
                  vol_to_surf_kwargs={}):
    '''
        Plot surface maps for several contrasts in one figure
    Args:
        imgs (list): meta-analytic maps (outputs from process_set)
        img_names (list): labels for each image
        select (list): subset of labels for which images should be plotted
    '''
    if select is None:
        select = img_names
    idxs = [np.where(np.array(img_names)==f)[0][0] for f in select]
    sub_imgs = [imgs[i] for i in idxs]
    if figsize is  None:
        figsize = (12,2.5*len(idxs))
    fig, ax = plt.subplots(ncols=4, nrows=len(idxs), 
                           figsize=figsize, 
                           subplot_kw={'projection':'3d'})
    
    for iidx, simg in enumerate(sub_imgs):
        ref = surface.vol_to_surf(simg, 
                                  fsaverage.pial_left)
        cm = _colorbar_from_array(ref, 
                                  vmax=vmax, 
                                  threshold=threshold, 
                                  kwargs={},
                                  cmap=get_cmap(cmap))
        for aix in range(4):
            s = 'left' if aix % 2 == 0 else 'right'
            v = 'lateral' if aix < 2 else 'medial'
            mesh = getattr(fsaverage, f'infl_{s}')
            bg = getattr(fsaverage, f'sulc_{s}')
            pial = getattr(fsaverage, f'pial_{s}')
            texture = surface.vol_to_surf(sub_imgs[iidx], pial, 
                                          **vol_to_surf_kwargs)
            plotting.plot_surf_stat_map(mesh,
                                        texture,
                                        colorbar=False,
                                        hemi=s,
                                        view=v,
                                        threshold=threshold,
                                        cmap=cmap,
                                        bg_map=bg,
                                        axes=ax[iidx,aix],
                                        figure=fig,
                                        vmax=vmax, 
                                        **surface_kwargs)
        cax = ax[iidx,3].inset_axes([1,.2,.05,.6])
        fig.colorbar(cm, ax=ax[iidx,3], cax=cax)
        ax[iidx, 0].set_title(label=select[iidx], 
                              loc='left', 
                              color='white', 
                              backgroundcolor='black',
                              fontdict={'fontsize': title_fontsize},
                              **title_kwargs)
    plt.tight_layout()
    plotting.show()
        