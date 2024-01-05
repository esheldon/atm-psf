def fits_to_exposure(fname):
    import fitsio
    from lsst.afw.cameraGeom.testUtils import DetectorWrapper
    import lsst.afw.image as afw_image
    from .wcs import header_to_wcs

    with fitsio.FITS(fname) as fits:
        hdr = fits[0].read_header()
        image = fits[0].read()

    wcs = header_to_wcs(hdr)

    ny, nx = image.shape
    masked_image = afw_image.MaskedImageF(nx, ny)
    masked_image.image.array[:, :] = image
    # variance is TBD
    # masked_image.variance.array[:, :] = variance.array
    # masked_image.mask.array[:, :] = bmask.array

    exp = afw_image.ExposureF(masked_image)

    filter_label = afw_image.FilterLabel(
        band=hdr['filter'],
        physical=hdr['filter'],
    )
    exp.setFilterLabel(filter_label)

    # exp.setPsf(dm_psf)

    exp.setWcs(wcs)

    detector = DetectorWrapper(hdr['DET_NAME']).detector
    exp.setDetector(detector)

    return exp
