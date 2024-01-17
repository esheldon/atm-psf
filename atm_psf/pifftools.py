def run_piff(psf_candidates, exposure):
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
    training, reserved.  Bool arrays, same length as star_select
    """

    training = star_select.copy()
    reserved = star_select.copy()

    for i in range(star_select.size):
        if not star_select[i]:
            continue

        r = rng.uniform()
        if r < reserve_frac:
            training[i] = False
        else:
            reserved[i] = False

    return training, reserved
