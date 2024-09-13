FLUX_NAME = 'base_PsfFlux_instFlux'


def select_stars(sources, plot_dir=None):
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

    if plot_dir is not None:
        plot_sizemag(sources=sources, keep=selected, plot_dir=plot_dir)

    return selected


def plot_sizemag(sources, keep, plot_dir):
    import matplotlib.pyplot as mplt
    import numpy as np
    from esutil.ostools import makedirs_fromfile

    flux = sources[FLUX_NAME]
    # T = sources['base_SdssShape_xx'] + sources['base_SdssShape_yy']
    T = (
        sources['ext_shapeHSM_HsmSourceMoments_xx']
        + sources['ext_shapeHSM_HsmSourceMoments_yy']
    )
    fwhm = np.sqrt(T/2) * 2.3548200450309493 * 0.2

    fwhm_mean = fwhm[keep].mean()
    fwhm_std = fwhm[keep].std()
    ymin = fwhm_mean - 10 * fwhm_std
    ymax = fwhm_mean + 20 * fwhm_std

    fig, ax = mplt.subplots()
    ax.set(
        xlabel='psf inst flux',
        ylabel='FWHM [arcsec]',
        # ylim=[0, 15],
        ylim=[ymin, ymax],
    )
    ax.set_xscale('log')

    ax.scatter(flux, fwhm, s=1)
    ax.scatter(flux[keep], fwhm[keep], s=1)
    ax.axhline(fwhm_mean)

    fname = f'{plot_dir}/sizemag.png'
    makedirs_fromfile(fname)
    fig.savefig(fname, dpi=150)

    return fig, ax
