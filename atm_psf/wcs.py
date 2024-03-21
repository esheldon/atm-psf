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


def fit_gs_wcs(
    orig_gs_wcs, truth, order=3, nsig=4, maxiter=20, get_indices=False,
):
    """
    fit galsim WCS using input ra, dec
    """
    import numpy as np
    import galsim

    # flux cut is to work around bug where faint objects
    # did not have DCR effect
    w, = np.where(
        np.isfinite(truth['x'])
        & (truth['phot_flux'] > 100)
    )

    print(f'starting with {w.size} stars')

    x, y = truth['x'], truth['y']
    ra, dec = truth['ra'], truth['dec']

    ok = False
    for iiter in range(maxiter):
        wcs = galsim.FittedSIPWCS(
            x[w],
            y[w],
            np.radians(ra[w]),
            np.radians(dec[w]),
            center=orig_gs_wcs.center,
            order=order,
        )
        wgood, x_std, y_std = get_wcs_non_outliers(
            wcs=wcs,
            data=truth[w],
            nsig=nsig,
        )

        if wgood.size == w.size:
            print('    Did not remove any outlier')
            ok = True
            break
        else:
            nd = w.size - wgood.size
            print(f'    removed {nd} on iter {iiter},'
                  f' std: {x_std:.3g}, {y_std:.3g}')
            w = w[wgood]

    if not ok:
        raise RuntimeError(f'did not converge after {iiter+1} iterations')

    print(f'    kept {w.size}')

    if get_indices:
        return wcs, w
    else:
        return wcs


def get_wcs_non_outliers(wcs, data, nsig=3):
    import galsim
    import numpy as np

    px, py = wcs.radecToxy(
        ra=data['ra'],
        dec=data['dec'],
        units=galsim.degrees,
    )
    xdiff = data['x'] - px
    ydiff = data['y'] - py
    x_std = xdiff.std()
    y_std = ydiff.std()
    # std = min(x_std, y_std)
    # std = max(x_std, y_std)

    xrdiff = np.abs(xdiff) / x_std
    yrdiff = np.abs(ydiff) / y_std
    # xrdiff = np.abs(xdiff) / std
    # yrdiff = np.abs(ydiff) / std
    wgood, = np.where((xrdiff < nsig) & (yrdiff < nsig))
    return wgood, x_std, y_std


def gs_wcs_to_dm_wcs(gs_wcs, bounds):
    header = {}
    gs_wcs.writeToFitsHeader(header, bounds)
    return header_to_wcs(header)
