# pixels for edge
EDGE = 50


def fits_to_exposure(fname, truth, rng, fwhm=0.8):
    """
    load an exposure from an eimage fits file.  They sky is subtracted
    if truth is sent

    Parameters
    ----------
    fname: str
        Path to fits file
    truth: str
        Path to truth file, from which sky is extracted.  It is assumed that
        the sky noise is poisson, so sky var = sky_level

    Returns
    -------
    afw.image.ExposureF
    """
    import fitsio
    from lsst.afw.cameraGeom.testUtils import DetectorWrapper
    import lsst.afw.image as afw_image
    from .wcs import header_to_wcs

    print('loading:', fname)
    with fitsio.FITS(fname) as fits:
        hdr = fits[0].read_header()
        image = fits[0].read()

    if truth is not None:
        print('loading:', truth)
        truth_data = fitsio.read(truth)

    wcs = header_to_wcs(hdr)

    ny, nx = image.shape
    masked_image = afw_image.MaskedImageF(nx, ny)
    masked_image.image.array[:, :] = image

    if truth is not None:
        sky = int(truth_data['sky_level'][0] * 0.2**2)

        print('sky:', sky)
        masked_image.image.array[:, :] -= sky
        masked_image.variance.array[:, :] = sky
        print('stats after subtraction:')
        print_image_stats(masked_image.image.array)

    EDGEFLAG = masked_image.mask.getMaskPlane('EDGE')
    masked_image.mask.array[:EDGE, :] = EDGEFLAG
    masked_image.mask.array[ny-EDGE, :] = EDGEFLAG
    masked_image.mask.array[:, :EDGE] = EDGEFLAG
    masked_image.mask.array[:, nx-EDGE] = EDGEFLAG

    exp = afw_image.ExposureF(masked_image)

    filter_label = afw_image.FilterLabel(
        band=hdr['filter'],
        physical=hdr['filter'],
    )
    try:
        exp.setFilterLabel(filter_label)
    except AttributeError:
        exp.setFilter(filter_label)

    psf = make_fixed_psf(fwhm=fwhm, rng=rng)
    exp.setPsf(psf)

    exp.setWcs(wcs)

    detector = DetectorWrapper(hdr['DET_NAME']).detector
    exp.setDetector(detector)

    return exp


def make_fixed_psf(fwhm, rng):
    """
    make a KernelPsf(FixedKernel()) for a gaussian with the input fwhm
    """
    import galsim
    from lsst.meas.algorithms import KernelPsf
    from lsst.afw.math import FixedKernel
    import lsst.afw.image as afw_image

    g = galsim.Gaussian(fwhm=fwhm)
    psf_image = g.drawImage(scale=0.2, nx=25, ny=25).array

    noise = psf_image.max() / 1000
    psf_image += rng.normal(scale=noise, size=psf_image.shape)

    psf_image = psf_image.astype(float)

    return KernelPsf(
        FixedKernel(afw_image.ImageD(psf_image))
    )


def print_image_stats(image):
    import esutil as eu
    import numpy as np

    mn, sig, err = eu.stat.sigma_clip(image.ravel(), get_err=True)

    print('    median:', np.median(image))
    print(f'    mean: {mn:3f} +/- {err:3f}')
