def save_stack_piff(fname, piff_psf):
    """
    Save a stack PiffPsf to a file.  The object is pickled using the _write()
    method of the PiffPsf.  Note pickling is not stable across python releases

    Parameters
    ----------
    fname: str
        Path for file to write
    piff_psf: lsst.meas.extensions.piff.piffPsf.PiffPsf
        The object to write
    """
    print('saving piff to:', fname)
    with open(fname, 'wb') as fobj:
        s = piff_psf._write()
        fobj.write(s)


def load_stack_piff(fname):
    """
    Load a stack PiffPsf from a file.  The object should have been pickled
    using the _write() method of the PiffPsf.  Note pickling is not stable
    across python releases

    Parameters
    ----------
    fname: str
        Path for file to write

    Returns
    --------
    piff_psf: lsst.meas.extensions.piff.piffPsf.PiffPsf
        The PiffPsf object
    """

    import lsst.meas.extensions.piff.piffPsf

    print('reading piff from:', fname)
    with open(fname, 'rb') as fobj:
        data = fobj.read()
    return lsst.meas.extensions.piff.piffPsf.PiffPsf._read(data)


def save_source_data(fname, data):
    """
    save stars and training/reserve samples

    Parameters
    ----------
    fname: str
        Path for file to write
    data: dict
        Should have entries
            sources: SourceCatalog
                The result of detection and measurement, including all objects
                not just star candidates
            star_select: array
                Bool array, True if the object was a psf candidate
            reserved: list of PsfCandidateF
                Bool array, True if the object was reserved for validation
            other: metadata from piff run
              such as spatialFitChi2, numAvailStars, numGoodStars, avgX, avgY
    """
    import pickle

    print('saving sources data to:', fname)
    with open(fname, 'wb') as fobj:
        s = pickle.dumps(data)
        fobj.write(s)


def load_source_data(fname):
    """
    save stars and training/reserve samples

    Parameters
    ----------
    fname: str
        Path for file to write

    Returns
    -------
    data: dict
        Should have entries
            sources: SourceCatalog
                The result of detection and measurement, including all objects
                not just star candidates
            star_select: array
                Bool array, True if the object was a psf candidate
            reserved: list of PsfCandidateF
                Bool array, True if the object was reserved for validation
            other: metadata from piff run
              such as spatialFitChi2, numAvailStars, numGoodStars, avgX, avgY
    """
    import pickle

    print('loading sources and candidates from:', fname)
    with open(fname, 'rb') as fobj:
        s = fobj.read()
        data = pickle.loads(s)

    return data
