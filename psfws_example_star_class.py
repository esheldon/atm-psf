import psfws
import numpy as np
import galsim
# from tqdm import trange

LAMBDAS = dict(u=365.49, g=480.03, r=622.20, i=754.06, z=868.21, y=991.66)


class HebertPSF(object):
    def __init__(self, alt, az, band, rng):
        self.alt = alt
        self.az = az
        self.rng = rng
        self.lam = LAMBDAS[band]

        if hasattr(rng, 'integers'):
            seed = rng.integers(0, 2**31)
        else:
            seed = rng.randint(0, 2**31)

        self.gs_rng = galsim.BaseDeviate(seed)

        kwargs = get_atm_kwargs(
            alt=self.alt,
            az=self.az,
            rng=self.rng,
            gs_rng=self.gs_rng,
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


def get_atm_kwargs(alt, az, rng, gs_rng, lam):
    """Get all atmospheric setup parameters."""
    # psf-weather-station params
    speeds, directions, altitudes, weights = get_psfws_params(
        alt, az, nlayers=6
    )

    # associated r0 at 500nm for these turbulence weights
    r0_500 = (2.914 * (500e-9)**(-2) * np.sum(weights))**(-3./5)

    # Draw L0 from truncated log normal, broadcast to list of layers
    L0 = 0
    while L0 < 10.0 or L0 > 100:
        L0 = np.exp(rng.normal() * 0.6 + np.log(25.0))
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
        rng=gs_rng,
    )


def get_psfws_params(alt, az, nlayers):
    """Use psf-weather-station to fetch simulation setup parameters."""
    ws = psfws.ParameterGenerator()
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


def dofit(im, rng):
    import ngmix

    cen = (np.array(im.shape) - 1)/2
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


def get_fwhm(img, show=False):
    import matplotlib.pyplot as plt
    import espy.images

    r, improf = espy.images.get_profile(img)
    r = r * 0.2
    improf *= 1.0/improf.max()
    # nrows, ncols = img.shape
    # cen = (np.array(img.shape) - 1)/2
    # rows, cols = np.mgrid[0:nrows, 0:ncols]
    # rows = rows - cen[0]
    # cols = cols - cen[1]
    # r = np.sqrt(rows**2 + cols++2).ravel()
    # imravel = img.ravel()

    if show:
        plt.scatter(r, improf)
        plt.show()

    fwhm = 1
    return fwhm


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    show = True
    psf_type = 'hebert'
    # psf_type = 'gauss'

    # optical_psf = galsim.Gaussian(fwhm=0.75)
    # optical_psf = galsim.Gaussian(fwhm=0.01)
    optical_psf = galsim.Gaussian(fwhm=0.35)

    rng = np.random.default_rng(1234)
    if psf_type == 'hebert':
        hpsf = HebertPSF(alt=90, az=90, rng=rng, band='i')
    else:
        hpsf = galsim.Gaussian(fwhm=0.8)
        hpsf.gs_rng = galsim.GaussianDeviate(rng.integers(0, 2**31))

    nx, ny = [17] * 2
    nstar = 100
    n_photons = 1.e6

    # for i in trange(nstar):
    for i in range(nstar):
        # print(f'{i+1}/{nstar}')
        thx, thy = rng.uniform(low=-0.5, high=0.5, size=2)

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
            rng=hpsf.gs_rng,
        ).array

        # import IPython; IPython.embed()
        if show:
            plt.imshow(np.log10(img.clip(min=1.0e-6)), origin='lower')
            plt.show()

        fwhm = get_fwhm(img, show=show)
        print(f'fwhm: {fwhm}')
        # res = dofit(img, rng)
        # print(f'fwhm: {res["fwhm"]}')
