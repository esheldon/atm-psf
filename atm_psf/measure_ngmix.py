import logging
LOG = logging.getLogger('measure_ngmix')

HMOM_NAMES = [
    'M00',  # MF
    'M11',  # MT
    'M22',  # r^4
    'M13',  # r^2 * 2 * u * v
    'M31',  # r^2 * (u^ - v^)
]


def ngmix_measure(exp, sources, stamp_size, rng):
    """
    run measurements on the input exposure, given the input measurement task,
    list of sources

    Parameters
    ----------
    exp: lsst.afw.image.Exposure
        The exposure on which to detect and measure
    sources: list of sources
        From a detection task
    stamp_size: int
        Size for postage stamps
    rng: numpy random number generator
        e.g. np.random.RandomState

    Returns
    -------
    ndarray with results or None
    """
    import esutil as eu
    from metadetect.fitting import get_admom_runner
    from metadetect.procflags import NO_ATTEMPT

    if len(sources) == 0:
        return None

    results = []
    runner = get_admom_runner(rng)

    for i, source in enumerate(sources):

        if source.get('deblend_nChild') != 0:
            psf_am_res = {'flags': NO_ATTEMPT}
            obj_am_res = {'flags': NO_ATTEMPT}
        else:
            psf_am_res, obj_am_res = _do_meas(
                exp=exp, source=source, stamp_size=stamp_size,
                runner=runner,
            )

        res = get_ngmix_output(res=obj_am_res, pres=psf_am_res)

        results.append(res)

    results = eu.numpy_util.combine_arrlist(results)

    return results


def _do_meas(exp, source, stamp_size, runner):
    import ngmix

    from metadetect.lsst.measure import (
        AllZeroWeightError, CentroidFailError,
    )
    from metadetect.procflags import (
        EDGE_HIT, ZERO_WEIGHTS, CENTROID_FAILURE,
        OBJ_FAILURE,
        PSF_FAILURE
    )
    from lsst.pex.exceptions import LengthError

    flags = 0
    try:
        obs = _get_stamp_obs(
            exp=exp, source=source, stamp_size=stamp_size,
        )
    except LengthError as err:
        # This is raised when a bbox hits an edge
        LOG.debug('%s', err)
        flags |= EDGE_HIT
    except AllZeroWeightError as err:
        # failure creating some observation due to zero weights
        LOG.info('%s', err)
        flags |= ZERO_WEIGHTS
    except CentroidFailError as err:
        # failure in the center finding
        LOG.info(str(err))
        flags |= CENTROID_FAILURE

    if flags == 0:
        try:
            psf_am_res = runner.go(obs.psf)
            if psf_am_res['flags'] == 0:
                _add_higher_order_moms(res=psf_am_res, obs=obs.psf)
        except ngmix.GMixRangeError:
            psf_am_res = {'flags': PSF_FAILURE}

        try:
            obj_am_res = runner.go(obs)
            if obj_am_res['flags'] == 0:
                _add_higher_order_moms(res=obj_am_res, obs=obs)
        except ngmix.GMixRangeError:
            obj_am_res = {'flags': OBJ_FAILURE}

    else:
        psf_am_res = {'flags': flags}
        obj_am_res = {'flags': flags}

    return psf_am_res, obj_am_res


def _add_higher_order_moms(res, obs):
    gmix = res.get_gmix()
    moms = gmix.get_weighted_moments(obs, with_higher_order=True)
    for name in HMOM_NAMES:
        res[name] = moms[name]


def get_ngmix_output_struct():
    import numpy as np
    dtype = [
        ('am_flags', 'i4'),
        ('am_s2n', 'f4'),
        ('am_e1', 'f8'),
        ('am_e1_err', 'f4'),
        ('am_e2', 'f8'),
        ('am_e2_err', 'f4'),
        ('am_T', 'f4'),
        ('am_T_err', 'f4'),
        ('am_rho4', 'f4'),
        ('am_rho4_err', 'f4'),
    ]
    for name in HMOM_NAMES:
        dtype += [(name, 'f4')]

    dtype += [
        ('am_psf_flags', 'i4'),
        ('am_psf_e1', 'f8'),
        ('am_psf_e2', 'f8'),
        ('am_psf_T', 'f4'),
        ('am_psf_rho4', 'f4'),
    ]
    for name in HMOM_NAMES:
        dtype += [('psf_' + name, 'f4')]

    st = np.zeros(1, dtype=dtype)

    for name in st.dtype.names:
        if 'flags' not in name:
            st[name] = np.nan

    return st


def get_ngmix_output(res, pres):
    st = get_ngmix_output_struct()

    st['am_flags'] = res['flags']
    st['am_psf_flags'] = pres['flags']

    if res['flags'] == 0:
        st['am_s2n'] = res['s2n']
        st['am_e1'] = res['e'][0]
        st['am_e2'] = res['e'][1]
        st['am_e1_err'] = res['e_err'][0]
        st['am_e2_err'] = res['e_err'][1]
        st['am_T'] = res['T']
        st['am_T_err'] = res['T_err']
        st['am_rho4'] = res['rho4']
        st['am_rho4_err'] = res['rho4_err']

        for name in HMOM_NAMES:
            st[name] = res[name]

    if pres['flags'] == 0:
        st['am_psf_e1'] = pres['e'][0]
        st['am_psf_e2'] = pres['e'][1]
        st['am_psf_T'] = pres['T']
        st['am_psf_rho4'] = pres['rho4']

        for name in HMOM_NAMES:
            st['psf_' + name] = pres[name]

    if pres['flags'] != 0:
        pass

    return st


def _get_stamp_obs(exp, source, stamp_size, clip=False):
    """
    Get a postage stamp MultibandExposure

    Parameters
    ----------
    mbexp: lsst.afw.image.MultibandExposure
        The exposures
    source: lsst.afw.table.SourceRecord
        The source for which to get the stamp
    stamp_size: int
        If sent, a bounding box is created with about this size rather than
        using the footprint bounding box. Typically the returned size is
        stamp_size + 1
    clip: bool, optional
        If set to True, clip the bbox to fit into the exposure.

        If clip is False and the bbox does not fit, a
        lsst.pex.exceptions.LengthError is raised

        Only relevant if stamp_size is sent.  Default False

    Returns
    -------
    lsst.afw.image.ExposureF
    """
    from metadetect.lsst.measure import _get_bbox, extract_obs

    bbox = _get_bbox(exp, source, stamp_size, clip=clip)

    subexp = exp[bbox]
    obs = extract_obs(exp=subexp, source=source)
    return obs
