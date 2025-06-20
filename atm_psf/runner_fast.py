def run_fast_sim(
    rng,
    config,
    instcat,
    ccds,
    outdir,
    use_existing=False,
    selector=lambda d, i: True,
):
    import os
    import numpy as np
    import logging
    import galsim
    import imsim
    from . import io
    from pprint import pformat
    from .logging import setup_logging
    import montauk

    setup_logging('info')
    logger = logging.getLogger('runner_fast.run_sim_fast')

    logger.info('sim config:')
    logger.info('\n' + pformat(config))

    gs_rng = galsim.BaseDeviate(rng.integers(0, 2**60))

    logger.info(f'loading opsim data from {instcat}')
    obsdata = montauk.opsim_data.load_obsdata_from_instcat(instcat)

    psf = None

    # coord system is centered on boresight
    big_wcs = make_tan_wcs(
        world_origin=obsdata['boresight'],
        image_origin=galsim.PositionD(0, 0),
    )

    sky_model = imsim.SkyModel(
        exptime=obsdata['exptime'],
        mjd=obsdata['mjd'],
        bandpass=obsdata['bandpass'],
    )

    for iccd, ccd in enumerate(ccds):
        logger.info('-' * 70)
        logger.info(f'ccd: {ccd} {iccd+1}/{len(ccds)}')

        # e.g. simdata-00355204-0-i-R14_S00-det063.fits
        fname = io.get_sim_output_fname(
            obsid=obsdata['obshistid'],
            ccd=ccd,
            band=obsdata['band'],
        )

        fname = os.path.join(outdir, fname)

        if use_existing and os.path.exists(fname):
            logger.info(f'using existing file {fname}')
            continue

        dm_detector = montauk.camera.make_dm_detector(ccd)

        wcs, icrf_to_field = montauk.wcs.make_batoid_wcs(
            obsdata=obsdata, dm_detector=dm_detector,
        )
        logger.info(f'loading objects from {instcat}')
        cat = imsim.instcat.InstCatalog(file_name=instcat, wcs=wcs)

        nobj = cat.getNObjects()

        logger.info(f'matched {nobj} objects to CCD area')
        if nobj == 0:
            continue

        if psf is None:
            logger.info('creating PS psf')
            psf = make_fov_psf(rng)

        truth = make_truth(nobj)

        bbox = dm_detector.getBBox()
        image = galsim.ImageF(bbox.width, bbox.height, wcs=wcs)
        pixel_scale = montauk.wcs.get_pixel_scale(wcs=wcs, bbox=bbox)

        sky_image = montauk.sky.make_sky_image(
            sky_model=sky_model,
            wcs=wcs,
            nx=bbox.width,
            ny=bbox.height,
            logger=logger,
        )

        med_noise_var = np.median(sky_image.array)

        nskipped_low_flux = 0
        nskipped_select = 0
        nskipped_bounds = 0

        for iobj in range(nobj):
            obj_coord = cat.world_pos[iobj]
            image_pos = cat.image_pos[iobj]

            # this is pixel coordinates, but relative to boresight. getPSF
            # takes relpos and adds back in half the "big" image size
            big_image_relpos = big_wcs.toImage(obj_coord)

            truth['x'][iobj] = image_pos.x
            truth['y'][iobj] = image_pos.y
            truth['ra'][iobj] = obj_coord.ra.deg
            truth['dec'][iobj] = obj_coord.dec.deg

            obj = cat.getObj(
                index=iobj, rng=gs_rng, exptime=obsdata['exptime']
            )

            if not selector(cat, iobj):
                nskipped_select += 1
                continue

            flux = obj.calculateFlux(obsdata['bandpass'])
            truth['flux'][iobj] = flux

            if flux <= 0:  # pragma: no cover
                nskipped_low_flux += 1
                continue

            # psf_at_pos = psf.getPSF(relpos=big_image_relpos)
            psf_at_pos = psf.getPSF(big_image_relpos, ccd_id=ccd)

            stamp_size = montauk.stamps.get_stamp_size(
                obj=obj, flux=flux, noise_var=med_noise_var, obsdata=obsdata,
                pixel_scale=pixel_scale,
            )
            local_wcs = wcs.local(image_pos=image_pos)

            prof = galsim.Convolve(obj, psf_at_pos)

            stamp = prof.drawImage(
                bandpass=obsdata['bandpass'],
                nx=stamp_size, ny=stamp_size,
                center=image_pos,
                wcs=local_wcs,
                method='fft',
            )

            # Some pixels can end up negative from FFT numerics.  Just set them
            # to 0.
            stamp.array[stamp.array < 0] = 0.

            # stamp.addNoise(galsim.PoissonNoise(rng=gs_rng))

            bounds = stamp.bounds & image.bounds
            if not bounds.isDefined():  # pragma: no cover
                nskipped_bounds += 1
                continue

            image[bounds] += stamp[bounds]

            truth['skipped'][iobj] = False

        # we have variance for sky and signal
        sky_plus_signal = image.array + sky_image.array

        # replace image with poisson deviate, including sky and signal
        image.array[:, :] = rng.poisson(lam=sky_plus_signal)

        logging.info(f'writing to: {fname}')

        extra = {'det_name': dm_detector.getName()}

        nskipped = (
            nskipped_select + nskipped_low_flux + nskipped_bounds
        )
        logger.info(f'skipped {nskipped}/{nobj}')
        logger.info(f'skipped {nskipped_low_flux}/{nobj} low flux')
        logger.info(f'skipped {nskipped_select}/{nobj} selector')
        logger.info(f'skipped {nskipped_bounds}/{nobj} bounds')

        io.save_sim_data(
            fname=fname,
            image=image,
            var=sky_plus_signal,
            sky_image=sky_image,
            truth=truth,
            obsdata=obsdata,
            extra=extra,
        )


def make_truth(nobj, with_realized_pos=False):
    """
    Make the truth for run_sim.  The catalog will have fields
        ('skipped', bool),
        ('ra', 'f8'),
        ('dec', 'f8'),
        ('x', 'f4'),
        ('y', 'f4'),
        ('nominal_flux', 'f4'),
        ('realized_flux', 'f4'),

    Parameters
    ----------
    nobj: int
        The number of rows

    Returns
    -------
    array
    """
    import numpy as np

    dtype = [
        ('skipped', bool),
        ('ra', 'f8'),
        ('dec', 'f8'),
        ('x', 'f4'),
        ('y', 'f4'),
        ('xfull', 'f4'),
        ('yfull', 'f4'),
        ('flux', 'f4'),
    ]

    st = np.zeros(nobj, dtype=dtype)
    st['skipped'] = True
    st['flux'] = np.nan

    return st


def make_tan_wcs(image_origin, world_origin, theta=None, scale=0.2):
    """
    make and return a wcs object

    Parameters
    ----------
    image_origin: galsim.PositionD
        Image origin position
    world_origin: galsim.CelestialCoord
        Origin on the sky
    scale: float
        Pixel scale
    theta: float, optional
        Rotation angle in radians

    Returns
    -------
    A galsim wcs object, currently a TanWCS
    """
    import numpy as np
    import galsim

    mat = np.array(
        [[scale, 0.0],
         [0.0, scale]],
    )
    if theta is not None:
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        rot = np.array(
            [[costheta, -sintheta],
             [sintheta, costheta]],
        )
        mat = np.dot(mat, rot)

    return galsim.TanWCS(
        affine=galsim.AffineTransform(
            mat[0, 0], mat[0, 1], mat[1, 0], mat[1, 1],
            origin=image_origin,
        ),
        world_origin=world_origin,
        units=galsim.arcsec,
    )


def make_fov_psf(rng):
    import simsim

    config = simsim.config.get_config({'sim_type': 'image'})
    return simsim.fov_psf.FOVPSF(
        rng=rng,
        config=config,
    )

    # covers 4 degree radius disk, 8 degree across
    # But how to get this all in the same coordinate system?
    # full_dim = int(4.0 * 3600 / 0.2 * 2)
    # return PowerSpectrumPSF(
    #     rng=rng,
    #     im_width=full_dim,
    #     buff=100,
    # )


class PowerSpectrumPSF(object):
    """Produce a spatially varying Moffat PSF according to the power spectrum
    given by Heymans et al. (2012).

    Parameters
    ----------
    rng : np.random.RandomState
        An RNG instance.
    im_width : float
        The width of the image in pixels.
    buff : int
        An extra buffer of pixels for things near the edge.
    scale : float
        The pixel scale of the image, default 0.2
    grid_spacing: float
        Spacing in arcsec of grid for look up table, default 10
    trunc : float
        The truncation scale in for the shape/magnification power
        spectrum used to generate the PSF variation.  Default 1
    variation_factor : float, optional
        This factor is used internally to scale the overall variance in the
        PSF shape power spectra and the change in the PSF size across the
        image. Setting this factor greater than 1 results in more variation
        and less than 1 results in less variation.
    median_seeing : float, optional
        The approximate median seeing for the PSF.

    Methods
    -------
    getPSF(pos)
        Get a PSF model at a given position.
    """
    def __init__(self, *,
                 rng,
                 im_width,
                 buff,
                 scale=0.2,
                 grid_spacing=10.0,
                 trunc=1,
                 variation_factor=1,
                 median_seeing=0.8):

        import numpy as np
        import galsim

        self._rng = rng
        self._im_cen = (im_width - 1)/2
        self._scale = scale
        self._grid_spacing = grid_spacing
        self._tot_width = im_width + 2 * buff
        self._buff = buff
        self._variation_factor = variation_factor
        self._median_seeing = median_seeing

        # set the power spectrum and PSF params
        # Heymans et al, 2012 found L0 ~= 3 arcmin, given as 180 arcsec here.
        def _pf(k):
            return (k**2 + (1./180)**2)**(-11./6.) * np.exp(-(k*trunc)**2)

        self._ps = galsim.PowerSpectrum(
            e_power_function=_pf,
            b_power_function=_pf,
        )

        width_arcsec = self._tot_width * self._scale
        self._ngrid = int(width_arcsec / self._grid_spacing)

        # ngrid = 128
        # grid_spacing = max(self._tot_width * self._scale / ngrid, 1)
        # self.ngrid = ngrid
        if hasattr(self._rng, 'randint'):
            seed = self._rng.randint(1, 2**30)
        else:
            seed = self._rng.integers(1, 2**30)

        self._ps.buildGrid(
            grid_spacing=self._grid_spacing,
            ngrid=self._ngrid,
            get_convergence=True,
            variance=(0.01 * variation_factor)**2,
            rng=galsim.BaseDeviate(seed),
        )

        # cache the galsim LookupTable2D objects by hand to speed computations
        g1_grid, g2_grid, mu_grid = galsim.lensing_ps.theoryToObserved(
            self._ps.im_g1.array, self._ps.im_g2.array,
            self._ps.im_kappa.array)

        # lut is short for lookup table
        self._lut_g1 = galsim.table.LookupTable2D(
            self._ps.x_grid,
            self._ps.y_grid, g1_grid.T,
            edge_mode='wrap',
            interpolant=galsim.Lanczos(5)
        )
        self._lut_g2 = galsim.table.LookupTable2D(
            self._ps.x_grid,
            self._ps.y_grid, g2_grid.T,
            edge_mode='wrap',
            interpolant=galsim.Lanczos(5)
        )
        self._lut_mu = galsim.table.LookupTable2D(
            self._ps.x_grid,
            self._ps.y_grid, mu_grid.T - 1,
            edge_mode='wrap',
            interpolant=galsim.Lanczos(5)
        )

        self._g1_mean = self._rng.normal() * 0.01 * variation_factor
        self._g2_mean = self._rng.normal() * 0.01 * variation_factor

        def _getlogmnsigma(mean, sigma):
            logmean = np.log(mean) - 0.5*np.log(1 + sigma**2/mean**2)
            logvar = np.log(1 + sigma**2/mean**2)
            logsigma = np.sqrt(logvar)
            return logmean, logsigma

        lm, ls = _getlogmnsigma(self._median_seeing, 0.1)
        self._fwhm_central = np.exp(self._rng.normal() * ls + lm)

    def _get_lensing(self, pos):
        import galsim

        pos_x, pos_y = galsim.utilities._convertPositions(
            pos, galsim.arcsec, '_get_lensing',
        )
        return (
            self._lut_g1(pos_x, pos_y),
            self._lut_g2(pos_x, pos_y),
            self._lut_mu(pos_x, pos_y)+1
        )

    def _get_atm(self, x, y):
        import numpy as np
        import galsim

        xs = (x + 1 - self._im_cen) * self._scale
        ys = (y + 1 - self._im_cen) * self._scale
        g1, g2, mu = self._get_lensing((xs, ys))

        if g1*g1 + g2*g2 >= 1.0:
            norm = np.sqrt(g1*g1 + g2*g2) / 0.5
            g1 /= norm
            g2 /= norm

        fwhm = self._fwhm_central / np.power(mu, 0.75)

        psf = galsim.Moffat(
            beta=2.5,
            fwhm=fwhm
        ).shear(
            g1=g1 + self._g1_mean, g2=g2 + self._g2_mean
        )

        return psf

    def getPSF(self, pos=None, relpos=None):  # noqa: N802
        """
        Get a PSF model at a given position.  Send one of
        pos= or relpos=

        Parameters
        ----------
        pos : galsim.PositionD
            The position at which to compute the PSF. In zero-indexed
            pixel coordinates.
        relpos : galsim.PositionD
            The position at which to compute the PSF, relative to
            the center

        Returns
        -------
        psf : galsim.GSObject
            A representation of the PSF as a galism object.
        """

        if pos is not None:
            psf = self._get_atm(pos.x, pos.y)
        elif relpos is not None:
            psf = self._get_atm(
                x=relpos.x + self._im_cen,
                y=relpos.y + self._im_cen,
            )
        else:
            raise ValueError('send pos= or relpos=')

        return psf.withFlux(1.0)
