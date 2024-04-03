import lsst.pex.config as pexConfig
from lsst.meas.algorithms.psfDeterminer import BasePsfDeterminerTask
from lsst.meas.extensions.piff.piffPsfDeterminer import (
    _validateGalsimInterpolant
)


def run_piff(psf_candidates, reserved, exposure, show=False):
    """
    Run PIFF on the image using the input PSF candidates

    Parameters
    ----------
    psf_candidates: of PsfCandidateF
        list of lsst.meas.algorithms.PsfCandidateF from running
        atm_psf.select.make_psf_candidates
    exposure: lsst.afw.image.ExposureF
        The exposure object
    """
    import numpy as np
    from lsst.meas.extensions.piff.piffPsfDeterminer import (
        # PiffPsfDeterminerConfig,
        getGoodPixels, computeWeight,
        CelestialWcsWrapper, UVWcsWrapper,
    )
    from lsst.meas.extensions.piff.piffPsf import PiffPsf

    from lsst.afw.cameraGeom import PIXELS, FIELD_ANGLE
    import galsim
    import piff
    import pprint

    # config = PiffPsfDeterminerConfig(
    #     useCoordinates='sky',
    #     # spatialOrder=3,
    #     # interpolant='Lanczos(5)',
    # )
    config = MyPiffPsfDeterminerConfig(
        useCoordinates='sky',
    )
    pprint.pprint(config.toDict())

    stampSize = config.stampSize
    psize = psf_candidates[0].getWidth()
    print('psize:', psize)
    if stampSize > psize:
        raise ValueError(
            f'stampSize {stampSize} is larger than the PSF candidate '
            f'size {psize}'
        )
        # stampSize = psf_candidates[0].getWidth()

    scale = exposure.getWcs().getPixelScale().asArcseconds()
    match config.useCoordinates:
        case 'field':
            detector = exposure.getDetector()
            pix_to_field = detector.getTransform(PIXELS, FIELD_ANGLE)
            gswcs = UVWcsWrapper(pix_to_field)
            pointing = None
        case 'sky':
            gswcs = CelestialWcsWrapper(exposure.getWcs())
            skyOrigin = exposure.getWcs().getSkyOrigin()
            ra = skyOrigin.getLongitude().asDegrees()
            dec = skyOrigin.getLatitude().asDegrees()
            pointing = galsim.CelestialCoord(
                ra*galsim.degrees,
                dec*galsim.degrees
            )
        case 'pixel':
            gswcs = galsim.PixelScale(scale)
            pointing = None

    stars = []
    ncand = len(psf_candidates)
    not_kept = np.zeros(ncand, dtype=bool)
    indices = np.arange(ncand)
    for ind, candidate, is_reserve in zip(indices, psf_candidates, reserved):
        cmi = candidate.getMaskedImage(stampSize, stampSize)
        good = getGoodPixels(cmi, config.zeroWeightMaskBits)
        fracGood = np.sum(good)/good.size
        if fracGood < config.minimumUnmaskedFraction:
            print(
                'skipping:',
                ind,
                f'{fracGood} < {config.minimumUnmaskedFraction}',
            )
            not_kept[ind] = True
            continue

        weight = computeWeight(cmi, config.maxSNR, good)

        bbox = cmi.getBBox()
        bds = galsim.BoundsI(
            galsim.PositionI(*bbox.getMin()),
            galsim.PositionI(*bbox.getMax())
        )
        gsImage = galsim.Image(bds, wcs=gswcs, dtype=float)
        gsImage.array[:] = cmi.image.array
        gsWeight = galsim.Image(bds, wcs=gswcs, dtype=float)
        gsWeight.array[:] = weight

        source = candidate.getSource()
        image_pos = galsim.PositionD(source.getX(), source.getY())

        properties = {'is_reserve': is_reserve}
        data = piff.StarData(
            gsImage,
            image_pos,
            weight=gsWeight,
            pointing=pointing,
            properties=properties,
        )

        star = piff.Star(data, None)
        assert star.is_reserve == is_reserve
        stars.append(star)

    piffConfig = {
        'type': "Simple",
        'model': {
            'type': 'PixelGrid',
            'scale': scale * config.samplingSize,
            'size': config.modelSize,
            'interp': config.interpolant
        },
        'interp': {
            'type': 'BasisPolynomial',
            'order': config.spatialOrder
        },
        'outliers': {
            'type': 'Chisq',
            'nsigma': config.outlierNSigma,
            'max_remove': config.outlierMaxRemove
        }
    }

    piffResult = piff.PSF.process(piffConfig)

    wcs = {0: gswcs}

    piffResult.fit(stars, wcs, pointing)

    # stats plots
    if show:
        run_stats(config, piffResult, stars)

    # this is indicative of old piff where stars could be dropped
    assert len(stars) == len(piffResult.stars)

    for i, star in enumerate(piffResult.stars):
        if star.is_flagged and not star.is_reserve:
            not_kept[i] = True

    if show:
        plot_stats(piffResult.stars, show=show)

    drawSize = 2*np.floor(0.5*stampSize/config.samplingSize) + 1

    meta = {}
    meta["spatialFitChi2"] = piffResult.chisq
    meta["numAvailStars"] = len(stars)
    meta["numGoodStars"] = len(piffResult.stars)
    meta["avgX"] = np.mean([p.x for p in piffResult.stars])
    meta["avgY"] = np.mean([p.y for p in piffResult.stars])

    if not config.debugStarData:
        for star in piffResult.stars:
            # Remove large data objects from the stars
            del star.fit.params
            del star.fit.params_var
            del star.fit.A
            del star.fit.b
            del star.data.image
            del star.data.weight
            del star.data.orig_weight

    ppsf = PiffPsf(drawSize, drawSize, piffResult)
    return ppsf, meta, not_kept


class MyPiffPsfDeterminerConfig(BasePsfDeterminerTask.ConfigClass):
    spatialOrder = pexConfig.Field[int](
        doc="specify spatial order for PSF kernel creation",
        default=2,
    )
    samplingSize = pexConfig.Field[float](
        doc="Resolution of the internal PSF model relative to the pixel size; "
        "e.g. 0.5 is equal to 2x oversampling",
        default=1,
    )
    modelSize = pexConfig.Field[int](
        doc="Internal model size for PIFF",
        default=25,
    )
    outlierNSigma = pexConfig.Field[float](
        doc="n sigma for chisq outlier rejection",
        default=4.0
    )
    outlierMaxRemove = pexConfig.Field[float](
        doc="Max fraction of stars to remove as outliers each iteration",
        default=0.05
    )
    maxSNR = pexConfig.Field[float](
        doc="Rescale the weight of bright stars such that their SNR is less "
            "than this value.",
        default=200.0
    )
    zeroWeightMaskBits = pexConfig.ListField[str](
        doc="List of mask bits for which to set pixel weights to zero.",
        default=['BAD', 'CR', 'INTRP', 'SAT', 'SUSPECT', 'NO_DATA']
    )
    minimumUnmaskedFraction = pexConfig.Field[float](
        doc="Minimum fraction of unmasked pixels required to use star.",
        default=0.5
    )
    interpolant = pexConfig.Field[str](
        doc="GalSim interpolant name for Piff to use. "
            "Options include 'Lanczos(N)', where N is an integer, along with "
            "galsim.Cubic, galsim.Delta, galsim.Linear, galsim.Nearest, "
            "galsim.Quintic, and galsim.SincInterpolant.",
        check=_validateGalsimInterpolant,
        default="Lanczos(11)",
    )
    debugStarData = pexConfig.Field[bool](
        doc="Include star images used for fitting in PSF model object.",
        default=False
    )
    useCoordinates = pexConfig.ChoiceField[str](
        doc="Which spatial coordinates to regress against in PSF modeling.",
        allowed=dict(
            pixel='Regress against pixel coordinates',
            field='Regress against field angles',
            sky='Regress against RA/Dec'
        ),
        default='pixel'
    )

    def setDefaults(self):
        super().setDefaults()
        self.modelSize = 25
        self.stampSize = 35


def run_stats(config, piffResult, stars):
    import os
    import piff

    stats_dir = 'plots'
    if not os.path.exists(stats_dir):
        try:
            os.makedirs(stats_dir)
        except Exception:
            pass

    star_file = f'{stats_dir}/starsfile.fits'
    stats_config = [
        {
            'type': 'StarImages',
            'file_name': star_file,
            'nplot': 0,  # all stars
            'adjust_stars': True,
        },
    ]

    stats_list = piff.Stats.process(stats_config)

    for istat, stat_obj in enumerate(stats_list):
        stat_obj.compute(piffResult, stars)
        fig, ax = stat_obj.plot()
        fname = f'stat-{config.useCoordinates}-{istat:02d}.png'
        fname = f'{stats_dir}/{fname}'
        print('writing:', fname)
        fig.savefig(fname, dpi=150)


def plot_stats(stars, show=False):
    import matplotlib.pyplot as mplt
    import numpy as np

    alpha = 0.5

    fig, axs = mplt.subplots(nrows=2, ncols=2)

    off = np.array(
        [np.sqrt(star.center[0]**2 + star.center[1]**2)
         for star in stars if not star.is_flagged]
    )
    uoff = np.array(
        [star.center[0] for star in stars if not star.is_flagged]
    )
    voff = np.array(
        [star.center[1] for star in stars if not star.is_flagged]
    )
    chisq = np.array(
        [star.fit.chisq for star in stars if not star.is_flagged]
    )

    foff = np.array(
        [np.sqrt(star.center[0]**2 + star.center[1]**2)
         for star in stars
         if star.is_flagged and not star.is_reserve]
    )
    fuoff = np.array(
        [star.center[0] for star in stars
         if star.is_flagged and not star.is_reserve]
    )
    fvoff = np.array(
        [star.center[1] for star in stars
         if star.is_flagged and not star.is_reserve]
    )
    fchisq = np.array(
        [star.fit.chisq for star in stars
         if star.is_flagged and not star.is_reserve]
    )

    ax = axs[0, 0]
    ax.set(xlabel=r'$\Delta u$ [arcsec]')

    bins = np.linspace(-10, 10, 50)
    ax.hist(uoff, bins=bins, label='unflagged', alpha=alpha)
    ax.hist(fuoff, bins=bins, label='flagged', alpha=alpha)
    ax.legend()

    ax = axs[0, 1]
    ax.set(xlabel=r'$\Delta v$ [arcsec]')

    ax.hist(voff, bins=bins, alpha=alpha)
    ax.hist(fvoff, bins=bins, alpha=alpha)

    ax = axs[1, 0]
    ax.set(xlabel=r'log$_{10}|$offset$|$ [arcsec]')

    bins = np.linspace(np.log10(0.001), np.log10(12), 50)
    ax.hist(np.log10(off), bins=bins, alpha=alpha)
    ax.hist(np.log10(foff), bins=bins, alpha=alpha)

    ax = axs[1, 1]
    ax.set(xlabel=r'log$_{10}[ \chi^2 ]$')

    max_chisq = chisq.max()
    min_chisq = chisq.min()
    if fchisq.size > 0:
        max_chisq = max(max_chisq, fchisq.max())
        min_chisq = min(min_chisq, fchisq.min())

    bins = np.linspace(np.log10(min_chisq), np.log10(max_chisq), 50)
    ax.hist(np.log10(chisq), bins=bins, alpha=alpha)
    ax.hist(np.log10(fchisq), bins=bins, alpha=alpha)

    # axs[1, 1].axis('off')

    if show:
        mplt.show()

    return fig, ax


def run_piff_old(psf_candidates, exposure):
    """
    Run PIFF on the image using the input PSF candidates

    Parameters
    ----------
    psf_candidates: of PsfCandidateF
        list of lsst.meas.algorithms.PsfCandidateF from running
        atm_psf.select.make_psf_candidates
    exposure: lsst.afw.image.ExposureF
        The exposure object
    """
    from lsst.meas.extensions.piff.piffPsfDeterminer import (
        PiffPsfDeterminerConfig,
        PiffPsfDeterminerTask,
    )

    config = PiffPsfDeterminerConfig()
    task = PiffPsfDeterminerTask(config=config)

    piff_psf, _ = task.determinePsf(
        exposure=exposure, psfCandidateList=psf_candidates,
    )
    return piff_psf


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
    from lsst.meas.algorithms.makePsfCandidates import (
        MakePsfCandidatesTask,
        MakePsfCandidatesConfig,
    )

    # this needs to be at least as big as stampSize in the piff wrapper config
    config = MakePsfCandidatesConfig(
        kernelSize=35,
    )
    task = MakePsfCandidatesTask(config)
    res = task.makePsfCandidates(sources, exposure)
    return res.psfCandidates


def split_candidates(rng, star_select, reserve_frac):
    """
    Split the candidates into training and validation samples

    Parameters
    ----------
    rng: numpy random state
        e.g. numpy.random.RandomState
    star_select: bool array
        Should have True if the object was selected as a star.  This array
        should be the lenght of the entire SourceCatalog
    reserve_frac: float
        Fraction to reserve

    Returns
    -------
    reserved.  Bool arrays, same length as star_select
    """

    reserved = star_select.copy()
    reserved[:] = False

    for i in range(star_select.size):
        if not star_select[i]:
            continue

        r = rng.uniform()
        if r < reserve_frac:
            reserved[i] = True

    return reserved
