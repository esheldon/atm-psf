def header_to_wcs(hdr):
    """
    convert a header to a WCS

    Note this will not capture tree rings
    """
    from lsst.daf.base import PropertyList
    from lsst.afw.geom import makeSkyWcs

    prop = PropertyList()
    for key in hdr:
        prop.set(key, hdr[key])

    return makeSkyWcs(prop)
