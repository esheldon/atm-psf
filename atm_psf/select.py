FLUX_NAME = 'base_PsfFlux_instFlux'


def make_psf_candidates(sources, exposure):
    """
    Select stars and construct a list of PsfCandidate

    Parameters
    ----------
    sources: SourceCatalog
        From running atm_psf.measure.detect_and_measure
    exposure: lsst.afw.image.ExposureF
        The exposure object.  This is needed for the psf candidates
        to construct postage stamp images

    Returns
    -------
    list of lsst.meas.algorithms.PsfCandidateF
    """
    from lsst.meas.algorithms.makePsfCandidates import MakePsfCandidatesTask

    keep = select_stars(sources)

    task = MakePsfCandidatesTask()
    res = task.makePsfCandidates(sources[keep], exposure)
    return res.psfCandidates


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

    from lsst.meas.algorithms.objectSizeStarSelector import (
        ObjectSizeStarSelectorConfig,
        ObjectSizeStarSelectorTask,
    )

    config = ObjectSizeStarSelectorConfig()
    config.sourceFluxField = FLUX_NAME
    # config.doSignalToNoiseLimit = True
    # config.fluxMin = 0
    # config.fluxMax = 0

    task = ObjectSizeStarSelectorTask(config=config)
    res = task.selectSources(sources)
    return res.selected


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
