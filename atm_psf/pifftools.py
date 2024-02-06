

def run_piff(psf_candidates, reserved, exposure):
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

    config = PiffPsfDeterminerConfig()

    if config.stampSize:
        stampSize = config.stampSize
        if stampSize > psf_candidates[0].getWidth():
            print('stampSize is larger than the PSF candidate '
                  'size.  Using candidate size.')
            stampSize = psf_candidates[0].getWidth()
    else:  # TODO: Only the if block should stay after DM-36311
        stampSize = psf_candidates[0].getWidth()

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
            'size': stampSize,
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

    # this is indicative of old piff where stars could be dropped
    assert len(stars) == len(piffResult.stars)

    for i, star in enumerate(piffResult.stars):
        if star.is_flagged:
            not_kept[i] = True

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
    from lsst.meas.algorithms.makePsfCandidates import MakePsfCandidatesTask

    task = MakePsfCandidatesTask()
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
