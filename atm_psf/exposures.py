# pixels for edge
EDGE = 50


def fits_to_exposure(fname, rng, fwhm=0.8):
    """
    load an exposure from an eimage fits file.

    Parameters
    ----------
    fname: str
        Path to fits file
    rng: np.random.default_rng
        The random number generator, used to add noise to the gaussian PSF
        image
    fwhm: float
        Initial PSF fwhm for detection, default 0.8

    Returns
    -------
    afw.image.ExposureF
    """
    import fitsio
    from lsst.afw.cameraGeom.testUtils import DetectorWrapper
    import lsst.afw.image as afw_image
    from .wcs import gs_wcs_to_dm_wcs

    print('loading:', fname)
    with fitsio.FITS(fname) as fits:
        image = fits['image'].read()
        sky_image = fits['sky'].read()
        image -= sky_image

        hdr = fits['truth'].read_header()

    print('image stats after subtraction:')
    print_image_stats(image)

    gs_wcs, bounds = load_galsim_info(fname)

    # gs_wcs = orig_gs_wcs
    wcs = gs_wcs_to_dm_wcs(gs_wcs, bounds)

    ny, nx = image.shape
    masked_image = afw_image.MaskedImageF(nx, ny)
    masked_image.image.array[:, :] = image
    masked_image.variance.array[:, :] = sky_image

    EDGEFLAG = masked_image.mask.getMaskPlane('EDGE')
    masked_image.mask.array[:EDGE, :] = EDGEFLAG
    masked_image.mask.array[ny-EDGE:, :] = EDGEFLAG
    masked_image.mask.array[:, :EDGE] = EDGEFLAG
    masked_image.mask.array[:, nx-EDGE:] = EDGEFLAG

    exp = afw_image.ExposureF(masked_image)

    filter_label = afw_image.FilterLabel(
        band=hdr['band'],
        physical=hdr['band'],
    )
    try:
        exp.setFilterLabel(filter_label)
    except AttributeError:
        exp.setFilter(filter_label)

    psf = make_fixed_psf(fwhm=fwhm, rng=rng)
    exp.setPsf(psf)

    exp.setWcs(wcs)

    detector = DetectorWrapper(hdr['det_name']).detector
    exp.setDetector(detector)

    return exp, hdr


def load_galsim_info(fname):
    import galsim
    try:
        gsim = galsim.fits.read(fname)
    except OSError:
        gsim = galsim.fits.read(fname, hdu=1)

    return gsim.wcs, gsim.bounds


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


def plot_flux_fwhm(flux, fwhm, show=False, figax=None):
    import matplotlib.pyplot as mplt

    if figax is None:
        fig, ax = mplt.subplots()

        ax.set_xscale('log')
        ax.set(
            xlim=[700, 1.e6],
            ylim=[0, 1.5],
            xlabel='flux',
            ylabel='FWHM [arcsec]',
        )
    else:
        fig, ax = figax

    ax.scatter(flux, fwhm, alpha=0.5)
    if show:
        mplt.show()

    return fig, ax


def view_image(im):
    import matplotlib.pyplot as mplt

    fig, ax = mplt.subplots()
    ax.imshow(im)
    mplt.show()
