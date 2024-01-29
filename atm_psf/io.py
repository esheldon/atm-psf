NONPOS_T = 2**0
PSF_NONPOS_T = 2**1


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
    import esutil as eu
    import pickle
    eu.ostools.makedirs_fromfile(fname)

    with open(fname, 'wb') as fobj:
        if piff_psf is None:
            s = pickle.dumps(None)
        else:
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
            seed: int
                Seed used in processing
            image_file, truth_file: str
                The input files
            additional entries:
               metadata from piff run, such as spatialFitChi2, numAvailStars,
               numGoodStars, avgX, avgY
    """
    import pickle
    import esutil as eu

    eu.ostools.makedirs_fromfile(fname)

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
            seed: int
                Seed used in processing
            image_file, truth_file: str
                The input files
            additional entries
              metadata from piff runsuch as spatialFitChi2, numAvailStars,
              numGoodStars, avgX, avgY
    """
    import pickle

    with open(fname, 'rb') as fobj:
        s = fobj.read()
        data = pickle.loads(s)

    return data


_name_map = {
    'ra': 'coord_ra',
    'dec': 'coord_dec',
    'x': 'x',
    'y': 'y',
}


def make_star_struct(n):
    import numpy as np
    dtype = [
        ('flags', 'i2'),
        ('star_select', bool),
        ('reserved', bool),
        ('ra', 'f8'),
        ('dec', 'f8'),
        ('x', 'f8'),
        ('y', 'f8'),
        ('psf_flux', 'f4'),
        ('psf_flux_err', 'f4'),
        ('e1', 'f8'),
        ('e2', 'f8'),
        ('T', 'f8'),
        ('psfrec_flags', 'i2'),
        ('psfrec_e1', 'f8'),
        ('psfrec_e2', 'f8'),
        ('psfrec_T', 'f8'),
    ]
    return np.zeros(n, dtype=dtype)


def load_sources_many(flist, nstars_min=50, fwhm_min=0.55, airmass_max=None):
    """
    load star data from multiple files.  See load_sources for details.

    Parameters
    ----------
    flist: [str]
        List of paths to source data

    Returns
    -------
    sources: array
        Array with e1, e2, T added as well as flags for processing.
        Flags are set based on HSM flags and if T is <= 0
    """
    import numpy as np
    import esutil as eu
    from .util import T_to_fwhm

    dlist = []
    for fname in flist:
        st, alldata = load_sources(fname, get_all=True)
        if airmass_max is not None:
            airmass = alldata['airmass']
            if airmass > airmass_max:
                print(f'    skipping airmass {airmass} > {airmass_max}')
                continue

        ss = st['star_select']
        nstars = ss.sum()
        if nstars < nstars_min:
            print(f'    skipping nstars {nstars} < {nstars_min}')
            continue

        res = st['reserved']
        fwhms = T_to_fwhm(st['psfrec_T'][res])
        fwhm = np.median(fwhms)
        if fwhm < fwhm_min:
            print(f'    skipping fwhm {fwhm:.3f} < {fwhm_min:.3f}')
            continue

        dlist.append(st)

    print(f'kept {len(dlist)}/{len(flist)}')
    return eu.numpy_util.combine_arrlist(dlist)


def load_sources(fname, get_all=False):
    """
    load source data with calculated shapes

    Parameters
    ----------
    fname: str
        Path to source data

    Returns
    -------
    sources: array
        Array with e1, e2, T added as well as flags for processing.
        Flags are set based on HSM flags and if T is <= 0
    """
    print('loading sources from:', fname)
    data = load_source_data(fname)

    sources = data['sources']

    st = make_star_struct(len(sources))

    st['star_select'] = data['star_select']

    if 'reserved' in data:
        st['reserved'] = data['reserved']

        st['ra'] = sources['coord_ra']
        st['dec'] = sources['coord_dec']
        st['psf_flux'] = sources['base_PsfFlux_instFlux']
        st['psf_flux_err'] = sources['base_PsfFlux_instFluxErr']

        st['x'] = sources['ext_shapeHSM_HsmSourceMoments_x']
        st['y'] = sources['ext_shapeHSM_HsmSourceMoments_y']

        _oflags = sources['ext_shapeHSM_HsmSourceMoments_flag']
        xx = sources['ext_shapeHSM_HsmSourceMoments_xx']
        xy = sources['ext_shapeHSM_HsmSourceMoments_xy']
        yy = sources['ext_shapeHSM_HsmSourceMoments_yy']

        _pflags = sources['ext_shapeHSM_HsmPsfMoments_flag']
        pxx = sources['ext_shapeHSM_HsmPsfMoments_xx']
        pxy = sources['ext_shapeHSM_HsmPsfMoments_xy']
        pyy = sources['ext_shapeHSM_HsmPsfMoments_yy']

        e1, e2, T, obj_flags = get_e1e2T(
            flags=_oflags, xx=xx, xy=xy, yy=yy,
        )
        psfrec_e1, psfrec_e2, psfrec_T, psfrec_flags = get_e1e2T(
            flags=_pflags, xx=pxx, xy=pxy, yy=pyy,
        )

        ngood = ((obj_flags == 0) & (psfrec_flags == 0)).sum()

        print(f'keeping: {ngood}/{len(sources)}')
        st['flags'] = obj_flags
        st['e1'] = e1
        st['e2'] = e2
        st['T'] = T

        st['psfrec_flags'] = psfrec_flags
        st['psfrec_e1'] = psfrec_e1
        st['psfrec_e2'] = psfrec_e2
        st['psfrec_T'] = psfrec_T
    else:
        st['flags'] = 2**10

    if get_all:
        return st, data
    else:
        return st


def get_e1e2T(xx, xy, yy, flags):
    """
    get e1, e2 and T from input moments

    Parameters
    ----------
    xx: array
        <x**2>
    xy: array
        <xy>
    yy: array
        <y**2>
    flags: array
        HSM flags

    Returns
    -------
    e1, e2, T, flags
        e1 and e2 will be nan if flags are nonzero.  Flags are set
        for input flags non zero or T <= 0
    """
    import numpy as np

    Tflags = np.zeros(xx.size, dtype='i2') + NONPOS_T
    e1 = xx * 0 + np.nan
    e2 = xx * 0 + np.nan

    T = xx + yy

    w, = np.where((flags == 0) & (T > 0))
    if w.size > 0:
        e1[w] = (xx[w] - yy[w]) / T[w]
        e2[w] = 2 * xy[w] / T[w]
        Tflags[w] = 0

    return e1, e2, T, Tflags


def load_instcat_paths(fname):
    with open(fname) as fobj:
        paths = [line.strip() for line in fobj]
    return paths


def load_opsim_info(fname, filter=None):
    import fitsio
    import numpy as np

    data = fitsio.read(fname)
    if filter is not None:
        w, = np.where(data['filter'] == filter)
        data = data[w]
    return data


def load_config(fname):
    import yaml
    with open(fname) as fobj:
        data = yaml.safe_load(fobj)
    return data
