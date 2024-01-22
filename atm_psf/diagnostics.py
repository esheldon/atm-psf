def plot_stars(st, pixel_scale=0.2, show=False):
    import matplotlib.pyplot as mplt
    import numpy as np

    fig, axs = mplt.subplots(ncols=2, figsize=(8, 4))

    logic = (st['flags'] == 0) & (st['psfrec_flags'] == 0)
    w, = np.where(logic)
    # slogic = logic & st['star_select']
    wstar, = np.where(st['star_select'][w])

    flux = st['psf_flux'][w]
    fwhm = np.sqrt(st['T'][w] / 2) * 2.3548 * pixel_scale
    Tratio = st['T'][w] / st['psfrec_T'][w]

    flux_lim = [1000, 1.e7]
    fwhm_lim = [0.3, 1.5]
    Tratio_lim = [0.0, 4]
    # fwhm
    axs[0].set(
        xlabel='PSF flux',
        ylabel='FWHM [arcsec]',
        xlim=flux_lim,
        ylim=fwhm_lim,
    )
    axs[0].set_xscale('log')

    ms = 1
    axs[0].scatter(
        flux,
        fwhm,
        s=ms,
        color='black',
        label='all',
    )
    axs[0].scatter(
        flux[wstar],
        fwhm[wstar],
        s=ms,
        color='red',
        label='star cand',
    )
    axs[0].legend()

    # T/Tpsf

    axs[1].set(
        xlabel='PSF flux',
        ylabel=r'T/T$_{\mathrm{psf}}$',
        xlim=flux_lim,
        ylim=Tratio_lim,
    )
    axs[1].set_xscale('log')

    axs[1].scatter(
        flux,
        Tratio,
        s=ms,
        color='black',
        label='all',
    )
    axs[1].scatter(
        flux[wstar],
        Tratio[wstar],
        s=ms,
        color='red',
        label='star cand',
    )

    if show:
        mplt.show()

    return fig, axs
