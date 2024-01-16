def save_stack_piff(fname, piff_psf):
    with open(fname, 'wb') as fobj:
        s = piff_psf._write()
        fobj.write(s)


def load_stack_piff(fname):
    import lsst.meas.extensions.piff.piffPsf
    with open(fname, 'rb') as fobj:
        data = fobj.read()
    return lsst.meas.extensions.piff.piffPsf.PiffPsf._read(data)
