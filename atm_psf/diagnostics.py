def plot_stars(st, pixel_scale=0.2, show=False):
    import matplotlib.pyplot as mplt
    import numpy as np
    import esutil as eu

    fig, axs = mplt.subplots(nrows=2, ncols=2, figsize=(8, 8))

    logic = (st['flags'] == 0) & (st['psfrec_flags'] == 0)
    w, = np.where(logic)
    # slogic = logic & st['star_select']
    wnostar, = np.where(~st['star_select'][w])
    wstar, = np.where(st['star_select'][w])
    wres, = np.where(st['reserved'][w])

    flux = st['psf_flux'][w]
    fwhm = np.sqrt(st['T'][w] / 2) * 2.3548 * pixel_scale
    T = st['T'][w]
    Tpsf = st['psfrec_T'][w]
    Tratio = T / Tpsf - 1
    e1 = st['e1'][w]
    e1psf = st['psfrec_e1'][w]
    e2 = st['e2'][w]
    e2psf = st['psfrec_e2'][w]

    flux_lim = [1000, 1.e7]
    fwhm_lim = [0.3, 1.2]
    Tratio_lim = [-1, 1]
    Tratio_hist_lim = [-0.1, 0.1]
    # Tdiff_lim = [-0.5, 0.5]
    ediff_lim = [-0.1, 0.1]

    # fwhm
    axs[0, 0].set(
        xlabel='PSF flux',
        ylabel='FWHM [arcsec]',
        xlim=flux_lim,
        ylim=fwhm_lim,
    )
    axs[0, 0].set_xscale('log')

    axs[0, 1].set(
        xlabel='PSF flux',
        ylabel=r'T/T$_{\mathrm{PSF}}$ - 1',
        xlim=flux_lim,
        ylim=Tratio_lim,
    )
    axs[0, 1].set_xscale('log')

    # axs[1, 0].set(
    #     xlabel=r'T$_{\mathrm{PSF}}$ - T$_{\mathrm{star}}$',
    #     xlim=Tdiff_lim,
    # )
    Tratio_label = r'T/T$_{\mathrm{PSF}}$ - 1'
    axs[1, 0].set(
        ylabel=Tratio_label,
        xlim=Tratio_hist_lim,
    )

    axs[1, 1].set(
        xlabel=r'e1$_{\mathrm{PSF}}$ - e1$_{\mathrm{star}}$',
        xlim=ediff_lim,
    )

    # ms = 0.1
    ms = 0.25
    axs[0, 0].scatter(
        flux[wnostar],
        fwhm[wnostar],
        marker='.',
        s=ms,
        c='black',
        label='all',
    )
    axs[0, 0].scatter(
        flux[wstar],
        fwhm[wstar],
        marker='.',
        s=ms,
        c='red',
        label='star cand',
    )

    axs[0, 0].legend()

    # T/Tpsf - 1

    axs[0, 1].scatter(
        flux[wnostar],
        Tratio[wnostar],
        marker='.',
        s=ms,
        color='black',
        label='all',
    )
    axs[0, 1].scatter(
        flux[wstar],
        Tratio[wstar],
        marker='.',
        s=ms,
        color='red',
        label='star cand',
    )
    axs[0, 1].axhline(1, color='darkgreen')

    nbin = 30

    # Tpsf - Tstar
    # Tdiff = Tpsf - T
    #
    # Tdiff_stats = eu.stat.get_stats(Tdiff[wres])
    # Tdmess = r'$\Delta$T = %.3g +/- %.3g' % (
    #     Tdiff_stats['mean'],
    #     Tdiff_stats['err'],
    # )
    # axs[1, 0].hist(
    #     Tdiff[wres],
    #     bins=np.linspace(Tdiff_lim[0], Tdiff_lim[1], nbin),
    # )
    # xloc = 0.1
    # axs[1, 0].text(
    #     xloc, 0.9, Tdmess,
    #     transform=axs[1, 0].transAxes,
    # )

    # T/Tpsf

    Tratio_stats = eu.stat.get_stats(Tratio[wres])
    Tdmess = Tratio_label + ': %.3g +/- %.3g' % (
        Tratio_stats['mean'],
        Tratio_stats['err'],
    )
    axs[1, 0].hist(
        Tratio[wres],
        bins=np.linspace(Tratio_hist_lim[0], Tratio_hist_lim[1], nbin),
    )
    xloc = 0.1
    axs[1, 0].text(
        xloc, 0.9, Tdmess,
        transform=axs[1, 0].transAxes,
    )

    # e diff
    e1diff = e1psf - e1
    e2diff = e2psf - e2
    e1diff_stats = eu.stat.get_stats(e1diff[wres])
    e2diff_stats = eu.stat.get_stats(e2diff[wres])

    e1dmess = r'$\Delta$e$_1$: %.3g +/- %.3g' % (
        e1diff_stats['mean'],
        e1diff_stats['err'],
    )
    e2dmess = r'$\Delta$e$_2$: %.3g +/- %.3g' % (
        e2diff_stats['mean'],
        e2diff_stats['err'],
    )

    axs[1, 1].hist(
        e1diff[wres],
        bins=np.linspace(ediff_lim[0], ediff_lim[1], nbin),
        alpha=0.5,
        label=r'$e_1$',
    )
    axs[1, 1].hist(
        e2diff[wres],
        bins=np.linspace(ediff_lim[0], ediff_lim[1], nbin),
        alpha=0.5,
        label=r'$e_1$',
    )
    axs[1, 1].text(
        xloc, 0.9, e1dmess,
        transform=axs[1, 1].transAxes,
    )
    axs[1, 1].text(
        xloc, 0.85, e2dmess,
        transform=axs[1, 1].transAxes,
    )
    axs[1, 1].legend(loc='lower right')

    if show:
        mplt.show()

    return fig, axs
