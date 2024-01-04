# import os
# import pickle
import numpy as np
import galsim
# from tqdm import trange

SCALE = 0.20
LAMBDAS = dict(u=365.49, g=480.03, r=622.20, i=754.06, z=868.21, y=991.66)
# gives median FWHM 0.8 when convolved by fixed gaussian with FWHM=0.35
# for "optical psf"
FUDGE_FACTOR = 0.30
NPIX = 4096

ALTRANGE = [20, 80]


class HebertPSFWithOptical(object):
    def __init__(self, alt, az, band, rng, optical_psf, wcs=None, fast=False):
        self.hebert_psf = HebertPSF(
            alt=alt, az=az, band=band, rng=rng, wcs=wcs,
            fast=fast,
        )
        self.optical_psf = optical_psf

    def _get_psf_focal_plane(self, u, v):
        return self.hebert_psf._get_psf_focal_plane(u, v)

    def get_psf(self, x, y):
        u, v = self.hebert_psf.get_focal_plane_coords(x, y)
        hpsf = self._get_psf_focal_plane(u, v)
        return self._convolve_optical(hpsf, x, y)

    def _convolve_optical(self, psf, x, y):
        if isinstance(self.optical_psf, galsim.GSObject):
            opsf = self.optical_psf
        else:
            opsf = self.optical_psf.get_psf(x, y)
        return galsim.Convolve(psf, opsf)


class HebertPSF(object):
    def __init__(self, alt, az, band, rng, wcs=None, fast=False):
        self.alt = alt
        self.az = az
        self.rng = rng
        self.lam = LAMBDAS[band]
        self.wcs = wcs
        self.fast = fast

        kwargs = get_atm_kwargs(
            alt=self.alt,
            az=self.az,
            rng=self.rng,
            lam=self.lam,
            fast=self.fast,
        )
        print('creating atm')
        self.atm = galsim.Atmosphere(**kwargs)

        print('creating Aperture')
        self.aper = galsim.Aperture(
            diam=8.36, obscuration=0.61, lam=self.lam, screen_list=self.atm,
        )

        r0 = kwargs['r0_500'] * (self.lam/500)**1.2
        kcrit = 0.2
        kmax = kcrit / r0

        # set up Galsim atmosphere object, can take a few minutes (generating
        # large arrays)
        print('instiantiating atm')
        self.atm.instantiate(kmax=kmax, check='phot')
        print('done')

    def get_psf(self, x, y):
        u, v = self.get_focal_plane_coords(x, y)

        return self._get_psf_focal_plane(u, v)

    def _get_psf_focal_plane(self, u, v):
        return self.atm.makePSF(
            self.lam, aper=self.aper, exptime=30.0, theta=(u, v),
        )

    def get_focal_plane_coords(self, x, y):
        if self.wcs is None:
            raise ValueError(
                'must send wcs to convert x,y to focal plane coords'
            )

        pos = galsim.PositionD(x, y)
        world_pos = self.wcs.toWorld(pos)
        world_origin = self.wcs.center
        # these are Angle objects
        u, v = world_origin.project(world_pos)
        return u, v


def get_atm_kwargs(alt, az, rng, lam, fast=False):
    """Get all atmospheric setup parameters."""
    # psf-weather-station params
    speeds, directions, altitudes, weights = get_psfws_params(
        alt, az, nlayers=6, rng=rng,
    )

    # associated r0 at 500nm for these turbulence weights
    r0_500 = FUDGE_FACTOR * (2.914 * (500e-9)**(-2) * np.sum(weights))**(-3./5)

    # Draw L0 from truncated log normal, broadcast to list of layers
    nrng = galsim.GaussianDeviate(rng)
    L0 = 0
    while L0 < 10.0 or L0 > 100:
        L0 = np.exp(nrng() * 0.6 + np.log(25.0))
    L0 = [L0] * len(speeds)

    if fast:
        screen_size = 25
    else:
        screen_size = set_screen_size(speeds)

    return dict(
        r0_500=r0_500,
        L0=L0,
        speed=speeds,
        direction=directions,
        altitude=altitudes,
        r0_weights=weights,
        screen_size=screen_size,
        screen_scale=0.1,
        rng=rng,
    )


def get_psfws_params(alt, az, nlayers, rng):
    """Use psf-weather-station to fetch simulation setup parameters."""
    import psfws

    ws = psfws.ParameterGenerator(seed=rng.raw())
    pt = ws.draw_datapoint()

    params = ws.get_parameters(
        pt, nl=nlayers, skycoord=True, alt=alt, az=az, location='com'
    )

    # place layers 200m above ground
    altitudes = [p - ws.h0 + 0.2 for p in params['h']]
    directions = [i*galsim.degrees for i in params['phi']]

    return params['speed'], directions, altitudes, params['j']


def set_screen_size(speeds):
    vmax = np.max(speeds)
    if vmax > 35:
        screen_size = 1050
    else:
        screen_size = vmax * 30
    return screen_size


def dofit(img, rng):
    import ngmix

    cen = (np.array(img.shape) - 1)/2
    jac = ngmix.DiagonalJacobian(
        row=cen[0], col=cen[1], scale=SCALE,
    )
    psf_obs = ngmix.Observation(
        img,
        weight=img * 0 + 1.0/0.001**2,
        jacobian=jac,
    )
    fitter = ngmix.gaussmom.GaussMom(fwhm=1.2)
    res = fitter.go(obs=psf_obs)
    # fitter = ngmix.admom.AdmomFitter(rng=rng)
    # guesser = ngmix.guessers.GMixPSFGuesser(
    #     rng=rng, ngauss=1, guess_from_moms=True,
    # )
    # runner = ngmix.runners.PSFRunner(fitter=fitter, guesser=guesser, ntry=4)
    # res = runner.go(obs=psf_obs)
    assert res['flags'] == 0
    res['fwhm'] = ngmix.moments.T_to_fwhm(res['T'])
    return res


def get_fwhm(img, show=False, save=False):
    import matplotlib.pyplot as plt
    import espy.images

    r, improf = espy.images.get_profile(img)
    r = r * SCALE
    improf *= 1.0/improf.max()

    s = improf.argsort()
    fwhm = 2 * np.interp(0.5, improf[s], r[s])
    # nrows, ncols = img.shape
    # cen = (np.array(img.shape) - 1)/2
    # rows, cols = np.mgrid[0:nrows, 0:ncols]
    # rows = rows - cen[0]
    # cols = cols - cen[1]
    # r = np.sqrt(rows**2 + cols++2).ravel()
    # imravel = img.ravel()

    if show or save:
        fig, ax = plt.subplots()
        ax.set(
            xlabel='r [arcsec]',
        )
        ax.plot(r, improf, marker='o')
        # ax.plot(improf, r, marker='o')
        # ax.scatter(r, improf)
        ax.axhline(0.5, color='black')
        ax.axvline(fwhm/2, color='black')
        ax.plot(fwhm/2, 0.5, color='red', marker='o')
        # ax.scatter(0.5, fwhm/2)
        ax.text(r.mean() * 1.3, 0.95, f'fwhm: {fwhm:2f}')
        if save:
            import fitsio
            plt.savefig('improf.png')
            output = np.zeros(r.size, dtype=[('r', 'f8'), ('prof', 'f8')])
            output['r'] = r[s]
            output['prof'] = improf[s]
            fitsio.write('improf.fits', output, clobber=True)
        if show:
            plt.show()
        plt.close()

    return fwhm


def get_rand_altaz(rng):
    cosalt_low = np.cos(np.radians(ALTRANGE[0]))
    cosalt_high = np.cos(np.radians(ALTRANGE[1]))

    crng = [cosalt_low, cosalt_high]
    cosalt = rng.uniform(low=min(crng), high=max(crng))
    cosalt = np.clip(cosalt, -1.0, 1.0)
    alt = np.arccos(cosalt)
    alt = np.degrees(alt)

    az = rng.uniform(low=0, high=360)
    return alt, az


def get_rand_radec(rng, num=None):
    if num is None:
        is_scalar = True
        num = 1
    else:
        is_scalar = False

    dec_range = [-60, 3]
    ra = rng.uniform(low=0, high=360, size=num)

    # number [-1,1)
    cosdec_min = np.cos(np.radians(90.0 + dec_range[0]))
    cosdec_max = np.cos(np.radians(90.0 + dec_range[1]))
    crng = [cosdec_min, cosdec_max]

    v = rng.uniform(low=min(crng), high=max(crng), size=num)

    np.clip(v, -1.0, 1.0, v)

    # Now this generates on [0,pi)
    dec = np.arccos(v)

    # convert to degrees
    np.degrees(dec, dec)

    # now in range [-90,90.0)
    dec -= 90.0
    if is_scalar:
        ra = ra[0]
        dec = dec[0]
    return ra, dec


def get_wcs(rng):
    ra, dec = get_rand_radec(rng)
    theta = rng.uniform(low=0, high=2*np.pi)

    image_origin = galsim.PositionD(NPIX/2, NPIX/2)
    world_origin = galsim.CelestialCoord(
        ra=ra * galsim.degrees,
        dec=dec * galsim.degrees,
    )
    mat = np.array(
        [[SCALE, 0.0],
         [0.0, SCALE]],
    )

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


def main(psf_type, seed, ntrial, outfile, fast=False, save=False, show=False):
    import matplotlib.pyplot as plt
    from esutil.stat import print_stats
    import fitsio

    rng = np.random.default_rng(seed)
    gsrng = galsim.BaseDeviate(rng.integers(0, 2**31))

    wcs = get_wcs(rng)
    alt, az = get_rand_altaz(rng)
    print(wcs)
    print('altaz:', alt, az)

    if psf_type == 'hebert':
        optical_psf = galsim.Gaussian(fwhm=0.35)
        hpsf = HebertPSFWithOptical(
            alt=alt, az=az, rng=gsrng, band='i',
            optical_psf=optical_psf,
            wcs=wcs,
            fast=fast,
        )
        # fname = f'hpsf-{oseed}.pkl'
        # if os.path.exists(fname):
        #     print('reading from:', fname)
        #     with open(fname, 'rb') as fobj:
        #         hpsf = pickle.load(fobj)
        # else:
        #     hpsf = HebertPSF(alt=90, az=90, rng=gsrng, band='i')
        #     print('pickling to:', fname)
        #     with open(fname, 'wb') as fobj:
        #         pickle.dump(hpsf, fobj)
    else:
        hpsf = galsim.Gaussian(fwhm=0.8)

    nx, ny = [17] * 2
    n_photons = 1.e6

    data = np.zeros(args.ntrial, dtype=[('fwhm', 'f8')])
    for i in range(args.ntrial):
        x, y = rng.uniform(low=0, high=NPIX-1, size=2)

        if isinstance(hpsf, galsim.GSObject):
            psf = hpsf
        else:
            psf = hpsf.get_psf(x, y)

        # draw PSF
        img = psf.drawImage(
            nx=nx,
            ny=ny,
            scale=SCALE,
            method='phot',
            n_photons=n_photons,
            rng=gsrng,
        ).array

        # import IPython; IPython.embed()
        if show:
            plt.imshow(np.log10(img.clip(min=1.0e-6)), origin='lower')
            plt.show()

        fwhm = get_fwhm(img, show=show, save=save)
        print(f'fwhm: {fwhm}')
        data['fwhm'][i] = fwhm

    print('fwhm stats')
    print_stats(data['fwhm'])
    print('writing:', outfile)
    fitsio.write(outfile, data, clobber=True)


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--ntrial', type=int, default=1)
    parser.add_argument('--fast', action='store_true',
                        help='use screen size 25 to speed up')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--psf', default='hebert')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(
        psf_type=args.psf,
        seed=args.seed,
        ntrial=args.ntrial,
        outfile=args.output,
        save=args.save,
        show=args.show,
        fast=args.fast,
    )
