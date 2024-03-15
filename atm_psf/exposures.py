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
    import galsim
    from lsst.afw.cameraGeom.testUtils import DetectorWrapper
    import lsst.afw.image as afw_image
    from .wcs import fit_gs_wcs, gs_wcs_to_dm_wcs
    from metadetect.lsst.skysub import iterate_detection_and_skysub
    import numpy as np

    print('loading:', fname)
    with fitsio.FITS(fname) as fits:
        hdr = fits[0].read_header()
        image = fits[0].read()

    gsim = galsim.fits.read(fname)
    orig_gs_wcs = gsim.wcs

    assert truth is not None, 'need truth for wcs'
    print('loading:', truth)
    truth_data = fitsio.read(truth)

    gs_wcs = fit_gs_wcs(orig_gs_wcs=orig_gs_wcs, truth=truth_data)
    wcs = gs_wcs_to_dm_wcs(gs_wcs, gsim.bounds)

    ny, nx = image.shape
    masked_image = afw_image.MaskedImageF(nx, ny)
    masked_image.image.array[:, :] = image

    if True and truth is not None:
        sky = int(truth_data['sky_level'][0] * 0.2**2)

        print('cat sky:', sky)
        print('subtracting cat sky:', sky)
        masked_image.image.array[:, :] -= sky
        masked_image.variance.array[:, :] = sky
        print('image stats after subtraction:')
        print_image_stats(masked_image.image.array)

    if False:
        import sep
        import sxdes

        bkg_model = sep.Background(masked_image.image.array)
        bkg = bkg_model.back()

        print('bkg median:', np.median(bkg))
        masked_image.image.array[:, :] -= bkg
        masked_image.variance.array[:, :] = bkg_model.globalrms**2
        print('image stats after subtraction:')
        print_image_stats(masked_image.image.array)
        # objects = sep.extract(
        #     masked_image.image.array, 1.0, err=bkg_model.globalrms,
        # )
        objects, seg = sxdes.run_sep(
            masked_image.image.array, noise=bkg_model.globalrms,
        )
        print('sky:', int(truth_data['sky_level'][0] * 0.2**2))
        print('globalrms**2:', bkg_model.globalrms**2)
        # import IPython; IPython.embed()
        # _fwhm = objects['flux_radius'] * 2 * 0.2,
        # plot_flux_fwhm(objects['flux_auto'], _fwhm, show=True)
        # stop

    EDGEFLAG = masked_image.mask.getMaskPlane('EDGE')
    masked_image.mask.array[:EDGE, :] = EDGEFLAG
    masked_image.mask.array[ny-EDGE:, :] = EDGEFLAG
    masked_image.mask.array[:, :EDGE] = EDGEFLAG
    masked_image.mask.array[:, nx-EDGE:] = EDGEFLAG
    # view_image(masked_image.mask.array)

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

    if True:
        print('doing iterative detection/sky subtraction')
        iterate_detection_and_skysub(
            exposure=exp,
            thresh=5,
        )

        print('image stats after second subtraction:')
        print('BGMEAN:', exp.getMetadata()['BGMEAN'])
        print_image_stats(exp.image.array)

        # sources = result.sources
        val = 2 ** exp.mask.getMaskPlaneDict()['DETECTED']
        exp.mask.array[:, :] &= ~val
        # view_image(exp.mask.array)
        # import IPython; IPython.embed()

        if False:
            from .measure import DetectMeasurer
            detmeas = DetectMeasurer(exposure=exp, rng=rng)
            detmeas.detect()
            detmeas.measure()
            sources = detmeas.sources

            Ts = (
                sources['base_SdssShape_xx'] + sources['base_SdssShape_yy']
            )
            _fwhm = np.sqrt(Ts/2) * 2.3548200450309493 * 0.2
            figax = plot_flux_fwhm(
                sources['base_PsfFlux_instFlux'],
                _fwhm,
                show=False,
            )

            Th = (
                sources['ext_shapeHSM_HsmSourceMoments_xx']
                + sources['ext_shapeHSM_HsmSourceMoments_yy']
            )

            _fwhm = np.sqrt(Th/2) * 2.3548200450309493 * 0.2
            plot_flux_fwhm(
                sources['base_PsfFlux_instFlux'],
                _fwhm,
                figax=figax,
                show=False,
            )
            _flux = objects['flux_auto']
            _fwhm = objects['flux_radius'] * 2 * 0.2,
            plot_flux_fwhm(
                _flux,
                _fwhm,
                figax=figax,
                show=True,
            )

    return exp, hdr


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
