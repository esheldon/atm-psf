# import os
# import pickle
import numpy as np
import galsim
# from tqdm import trange

LAMBDAS = dict(u=365.49, g=480.03, r=622.20, i=754.06, z=868.21, y=991.66)
# 0.5 with gaussian optical fwhm 0.35 gets overall fwhm=0.53
# 0.25 gets fwhm=1.1
# 0.375 gets fwhm=0.64
# 0.30 gets fwhm=0.65?
FUDGE_FACTOR = 0.30


class HebertPSF(object):
    def __init__(self, alt, az, band, rng):
        self.alt = alt
        self.az = az
        self.rng = rng
        self.lam = LAMBDAS[band]

        kwargs = get_atm_kwargs(
            alt=self.alt,
            az=self.az,
            rng=self.rng,
            lam=self.lam,
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
        thx, thy = self.convert_to_focal_plane_coords(x, y)

        return self.get_psf_focal_plane(thx, thy)

    def get_psf_focal_plane(self, thx, thy):
        theta = (thx*galsim.degrees, thy*galsim.degrees)
        return self.atm.makePSF(
            self.lam, aper=self.aper, exptime=30.0, theta=theta,
        )

        # psfRng = galsim.BaseDeviate(psfSeed)
        # draw PSF
        # img = psf.drawImage(
        #     nx=nx, ny=ny, scale=0.2, method='phot',
        #     n_photons=nPhot, rng=psfRng,
        # )


def get_atm_kwargs(alt, az, rng, lam):
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

    return dict(
        r0_500=r0_500,
        L0=L0,
        speed=speeds,
        direction=directions,
        altitude=altitudes,
        r0_weights=weights,
        screen_size=set_screen_size(speeds),
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
        row=cen[0], col=cen[1], scale=0.2,
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
    r = r * 0.2
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
        ax.scatter(fwhm/2, 0.5, color='red')
        # ax.scatter(0.5, fwhm/2)
        ax.text(1.0, 0.95, f'fwhm: {fwhm:g}')
        if save:
            plt.savefig('tmp.png')
        if show:
            plt.show()
        plt.close()

    return fwhm


def main(seed, outfile):
    import matplotlib.pyplot as plt
    from esutil.stat import print_stats
    import fitsio

    show = False
    save = False
    psf_type = 'hebert'
    # psf_type = 'gauss'

    # optical_psf = galsim.Gaussian(fwhm=0.75)
    # optical_psf = galsim.Gaussian(fwhm=0.01)
    optical_psf = galsim.Gaussian(fwhm=0.35)

    rng = galsim.BaseDeviate(seed)
    urng = galsim.UniformDeviate(rng)
    if psf_type == 'hebert':
        hpsf = HebertPSF(alt=90, az=90, rng=rng, band='i')
        # fname = f'hpsf-{oseed}.pkl'
        # if os.path.exists(fname):
        #     print('reading from:', fname)
        #     with open(fname, 'rb') as fobj:
        #         hpsf = pickle.load(fobj)
        # else:
        #     hpsf = HebertPSF(alt=90, az=90, rng=rng, band='i')
        #     print('pickling to:', fname)
        #     with open(fname, 'wb') as fobj:
        #         pickle.dump(hpsf, fobj)
    else:
        hpsf = galsim.Gaussian(fwhm=0.8)

    nstar = 100
    nx, ny = [17] * 2
    n_photons = 1.e6

    data = np.zeros(nstar, dtype=[('fwhm', 'f8')])
    # for i in trange(nstar):
    for i in range(nstar):
        # print(f'{i+1}/{nstar}')
        # thx, thy = rng.uniform(low=-0.5, high=0.5, size=2)
        thx = urng() - 0.5
        thy = urng() - 0.5

        if isinstance(hpsf, galsim.GSObject):
            psf = hpsf
        else:
            psf = hpsf.get_psf_focal_plane(thx, thy)
            psf = galsim.Convolve(psf, optical_psf)

        # draw PSF
        img = psf.drawImage(
            nx=nx,
            ny=ny,
            scale=0.2,
            method='phot',
            n_photons=n_photons,
            rng=rng,
        ).array

        # import IPython; IPython.embed()
        if show:
            plt.imshow(np.log10(img.clip(min=1.0e-6)), origin='lower')
            plt.show()

        fwhm = get_fwhm(img, show=show, save=save)
        print(f'fwhm: {fwhm}')
        data['fwhm'][i] = fwhm
        # res = dofit(img, rng)
        # print(f'fwhm: {res["fwhm"]}')

    print('fwhm stats')
    print_stats(data['fwhm'])
    print('writing:', outfile)
    fitsio.write(outfile, data, clobber=True)


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args.seed, args.output)
