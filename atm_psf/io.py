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


def save_catalogs(fname, sources, training, reserved):
    """
    save stars and training/reserve samples

    Parameters
    ----------
    fname: str
        Path for file to write
    sources: SourceCatalog
        The result of detection and measurement, including all objects
        not just star candidates
    training: list of PsfCandidateF
        The training sample
    reserved: list of PsfCandidateF
        The reserve sample
    """
    import pickle

    print('saving stars to:', fname)
    with open(fname, 'wb') as fobj:
        output = (sources, training, reserved)
        s = pickle.dumps(output)
        fobj.write(s)
