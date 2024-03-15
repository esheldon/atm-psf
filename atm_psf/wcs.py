def header_to_wcs(hdr):
    """
    convert a header to a WCS

    Note this will not capture tree rings
    """
    from lsst.daf.base import PropertyList
    from lsst.afw.geom import makeSkyWcs

    prop = PropertyList()
    for key in hdr:
        if key[:3] == 'GS_':
            continue
        prop.set(key, hdr[key])

    return makeSkyWcs(prop)


def fit_gs_wcs(orig_gs_wcs, truth):
    """
    fit galsim WCS using input ra, dec
    """
    import numpy as np
    import galsim
    w, = np.where(np.isfinite(truth['x']))

    return galsim.FittedSIPWCS(
        truth['x'][w],
        truth['y'][w],
        np.radians(truth['ra'][w]),
        np.radians(truth['dec'][w]),
        center=orig_gs_wcs.center,
        order=3,
    )


def gs_wcs_to_dm_wcs(gs_wcs, bounds):
    header = {}
    gs_wcs.writeToFitsHeader(header, bounds)
    return header_to_wcs(header)
