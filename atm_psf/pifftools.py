def run_piff(psf_candidates, exposure):
    """
    Run PIFF on the image using the input PSF candidates

    Parameters
    ----------
    psf_candidates:
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
