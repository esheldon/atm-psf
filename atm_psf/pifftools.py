

def run_piff(
    psf_candidates, reserved, exposure, spatial_order=2, plot_dir=None,
):
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
        PiffPsfDeterminerConfig,
        getGoodPixels, computeWeight,
        CelestialWcsWrapper, UVWcsWrapper,
    )
    from lsst.meas.extensions.piff.piffPsf import PiffPsf

    from lsst.afw.cameraGeom import PIXELS, FIELD_ANGLE
    import galsim
    import piff
    import pprint

    stack_config = PiffPsfDeterminerConfig(
        modelSize=25,
        stampSize=35,
        # useCoordinates='sky',
        spatialOrder=spatial_order,
        # interpolant='Lanczos(5)',
    )
    pprint.pprint(stack_config.toDict())

    stampSize = stack_config.stampSize
    psize = psf_candidates[0].getWidth()

    if stampSize > psize:
        raise ValueError(
            f'stampSize {stampSize} is larger than the PSF candidate '
            f'size {psize}'
        )
        # stampSize = psf_candidates[0].getWidth()

    scale = exposure.getWcs().getPixelScale().asArcseconds()
    match stack_config.useCoordinates:
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
        good = getGoodPixels(cmi, stack_config.zeroWeightMaskBits)
        fracGood = np.sum(good)/good.size
        if fracGood < stack_config.minimumUnmaskedFraction:
            print(
                'skipping:',
                ind,
                f'{fracGood} < {stack_config.minimumUnmaskedFraction}',
            )
            not_kept[ind] = True
            continue

        weight = computeWeight(cmi, stack_config.maxSNR, good)

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
            'scale': scale * stack_config.samplingSize,
            'size': stack_config.modelSize,
            'interp': stack_config.interpolant
        },
        'interp': {
            'type': 'BasisPolynomial',
            'order': stack_config.spatialOrder
        },
        'outliers': {
            'type': 'Chisq',
            'nsigma': stack_config.outlierNSigma,
            'max_remove': stack_config.outlierMaxRemove
        }
    }

    piffResult = piff.PSF.process(piffConfig)

    wcs = {0: gswcs}

    piffResult.fit(stars, wcs, pointing)

    if plot_dir is not None:
        run_stats(stack_config, piffResult, stars, stats_dir=plot_dir)

    # this is indicative of old piff where stars could be dropped
    assert len(stars) == len(piffResult.stars)

    for i, star in enumerate(piffResult.stars):
        if star.is_flagged and not star.is_reserve:
            not_kept[i] = True

    if plot_dir is not None:
        plot_stats(piffResult.stars, plot_dir=plot_dir)

    drawSize = 2*np.floor(0.5*stampSize/stack_config.samplingSize) + 1

    meta = {}
    meta["spatialFitChi2"] = piffResult.chisq
    meta["numAvailStars"] = len(stars)
    meta["numGoodStars"] = len(piffResult.stars)
    meta["avgX"] = np.mean([p.x for p in piffResult.stars])
    meta["avgY"] = np.mean([p.y for p in piffResult.stars])

    if not stack_config.debugStarData:
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


def run_stats(config, piffResult, stars, stats_dir):
    import piff
    from esutil.ostools import makedirs_fromfile

    star_file = f'{stats_dir}/starsfile.fits'
    makedirs_fromfile(star_file)

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


def plot_stats(stars, plot_dir):
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

    fname = f'{plot_dir}/stats-plot.png'
    fig.savefig(fname, dpi=150)

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


def get_piff_config(config=None):
    output = {
        'nstars_min': 50,
        'spatial_order': 2,
    }
    if config is not None:
        output.update(config)
    return output
