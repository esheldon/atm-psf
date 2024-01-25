FLUX_NAME = 'base_PsfFlux_instFlux'


def select_stars(sources):
    """
    Select stars and construct a list of PsfCandidate

    Parameters
    ----------
    sources: SourceCatalog
        From running atm_psf.measure.detect_and_measure

    Returns
    -------
    array of bool, True for the kept candidates
    """
    import numpy as np

    from lsst.meas.algorithms.objectSizeStarSelector import (
        ObjectSizeStarSelectorConfig,
        ObjectSizeStarSelectorTask,
    )

    config = ObjectSizeStarSelectorConfig()
    config.sourceFluxField = FLUX_NAME
    # config.doSignalToNoiseLimit = True
    # config.fluxMin = 0
    # config.fluxMax = 0

    # remove parents and blends
    selected = (
        (sources['deblend_nChild'] == 0)
        & (sources['deblend_parentNPeaks'] == 0)
    )
    task = ObjectSizeStarSelectorTask(config=config)
    res = task.selectSources(sources[selected])
    print('    selected', res.selected.sum())

    w, = np.where(selected)
    selected[w[~res.selected]] = False
    print('    kept', selected.sum(), 'after blending cuts')
    # import IPython; IPython.embed()
    return selected


def plot_sizemag(sources, keep, show=False):
    # import numpy as np
    import matplotlib.pyplot as mplt

    flux = sources[FLUX_NAME]
    T = sources['base_SdssShape_xx'] + sources['base_SdssShape_yy']

    fig, ax = mplt.subplots()
    ax.set(
        # xlabel='mag',
        xlabel='psf inst flux',
        ylabel='T',
        ylim=[0, 15],
    )
    ax.set_xscale('log')

    ax.scatter(flux, T, s=1)
    ax.scatter(flux[keep], T[keep], s=1)

    if show:
        mplt.show()

    return fig, ax
