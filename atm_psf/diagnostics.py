def plot_stars(st, pixel_scale=0.2, nbin=30, show=False, frac=None):
    import matplotlib.pyplot as mplt
    import numpy as np
    import esutil as eu

    fig, axs = mplt.subplots(nrows=2, ncols=2, figsize=(8, 8))

    logic = (
        (st['flags'] == 0)
        & (st['psfrec_flags'] == 0)
        & (st['psf_flux_err'] > 0)
    )

    w, = np.where(logic)
    sind = np.arange(w.size)
    if frac is not None:
        num = int(frac * w.size)
        rng = np.random.default_rng(seed=st.size)
        sind = rng.choice(sind, size=num)

    s2n = st['psf_flux'][w] / st['psf_flux_err'][w]

    fwhm = np.sqrt(st['T'][w] / 2) * 2.3548 * pixel_scale
    T = st['T'][w]
    Tpsf = st['psfrec_T'][w]
    Tratio = T / Tpsf - 1
    e1 = st['e1'][w]
    e1psf = st['psfrec_e1'][w]
    e2 = st['e2'][w]
    e2psf = st['psfrec_e2'][w]

    # flux_lim = [1000, 1.e7]
    s2n_lim = [3, 50000]
    fwhm_lim = [0.3, 1.2]
    Tratio_lim = [-1, 1]
    Tratio_hist_lim = [-0.1, 0.1]
    ediff_lim = [-0.1, 0.1]

    wnostar, = np.where(~st['star_select'][w[sind]])
    wstar, = np.where(st['star_select'][w[sind]])
    wnostar = sind[wnostar]
    wstar = sind[wstar]

    # fwhm
    axs[0, 0].set(
        xlabel='S/N',
        ylabel='FWHM [arcsec]',
        # xlim=flux_lim,
        xlim=s2n_lim,
        ylim=fwhm_lim,
    )
    axs[0, 0].set_xscale('log')

    Tratio_label = r'T/T$_{\mathrm{PSF}} - 1$'
    axs[0, 1].set(
        xlabel='S/N',
        ylabel=Tratio_label,
        # xlim=flux_lim,
        xlim=s2n_lim,
        ylim=Tratio_lim,
    )
    axs[0, 1].set_xscale('log')

    axs[1, 0].set(
        xlabel=Tratio_label,
        xlim=Tratio_hist_lim,
    )

    axs[1, 1].set(
        xlabel=r'e$_{\mathrm{star}} - $e$_{\mathrm{PSF}}$',
        xlim=ediff_lim,
    )

    # ms = 0.1
    ms = 0.25
    axs[0, 0].scatter(
        s2n[wnostar],
        fwhm[wnostar],
        marker='.',
        s=ms,
        c='black',
    )
    axs[0, 0].scatter(
        s2n[wstar],
        fwhm[wstar],
        marker='.',
        s=ms,
        c='red',
    )

    # Above makes the legend markers too small.  Use larger fake points off
    # plot
    axs[0, 0].scatter(1.e9, -1000, marker='.', c='black', label='all')
    axs[0, 0].scatter(1.e9, -1000, marker='.', c='red', label='star cand')
    axs[0, 0].legend()

    # T/Tpsf - 1

    axs[0, 1].scatter(
        s2n[wnostar],
        Tratio[wnostar],
        marker='.',
        s=ms,
        color='black',
    )
    axs[0, 1].scatter(
        s2n[wstar],
        Tratio[wstar],
        marker='.',
        s=ms,
        color='red',
    )
    axs[0, 1].axhline(0, color='darkgreen', alpha=0.5)

    #
    # only reserved stars used in the following plots
    #

    # T/Tpsf

    wres, = np.where(st['reserved'][w])

    Tratio_stats = eu.stat.get_stats(Tratio[wres])
    Tdmess = Tratio_label + ': %.3g +/- %.3g' % (
        Tratio_stats['mean'],
        Tratio_stats['err'],
    )
    axs[1, 0].hist(
        Tratio[wres],
        bins=np.linspace(Tratio_hist_lim[0], Tratio_hist_lim[1], nbin),
        alpha=0.5,
    )
    xloc = 0.1
    axs[1, 0].text(
        xloc, 0.9, Tdmess,
        transform=axs[1, 0].transAxes,
    )

    # e diff
    e1diff = e1 - e1psf
    e2diff = e2 - e2psf
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
