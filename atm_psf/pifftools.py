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


def split_candidates(psf_candidates, rng, reserve_frac):
    """
    Split the candidates into training and validation samples

    Parameters
    ----------
    psf_candidates: of PsfCandidateF
        list of lsst.meas.algorithms.PsfCandidateF from running
        atm_psf.select.make_psf_candidates
    rng: numpy random state
        e.g. numpy.random.RandomState
    reserve_frac: float
        Fraction to reserve
    """

    training = []
    reserved = []
    for cand in psf_candidates:
        r = rng.uniform()
        if r < reserve_frac:
            reserved.append(cand)
        else:
            training.append(cand)

    return training, reserved
