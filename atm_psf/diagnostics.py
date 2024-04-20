def plot_star_stats(st, pixel_scale=0.2, nbin=30, show=False, frac=None):
    import matplotlib.pyplot as mplt
    import numpy as np
    from .util import T_to_fwhm

    rng = np.random.default_rng(seed=st.size)

    fig, axs = mplt.subplots(nrows=2, ncols=2, figsize=(9, 8))

    logic = get_logic(st)

    w, = np.where(logic)
    sind = np.arange(w.size)
    if frac is not None:
        num = int(frac * w.size)
        sind = rng.choice(sind, size=num)

    s2n = st['psf_flux'][w] / st['psf_flux_err'][w]

    fwhm = T_to_fwhm(st['am_T'][w])
    # fwhm = T_to_fwhm(st['psfrec_T'][w])
    T = st['am_T'][w]
    Tpsf = st['am_psf_T'][w]
    Tratio = T / Tpsf - 1
    e1 = st['am_e1'][w]
    e1psf = st['am_psf_e1'][w]
    e2 = st['am_e2'][w]
    e2psf = st['am_psf_e2'][w]

    wnostar, = np.where(~st['star_select'][w[sind]])
    wstar, = np.where(st['star_select'][w[sind]])
    wnostar = sind[wnostar]
    wstar = sind[wstar]

    # flux_lim = [1000, 1.e7]
    s2n_lim = [3, 50000]
    fwhm_lim = [0.3, 2.0]
    Tratio_lim = [-1, 1]
    # Tratio_hist_lim = [-0.05, 0.05]
    Tratio_hist_lim = [-0.1, 0.1]
    ediff_lim = [-0.07, 0.07]

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

    Tstat = bootstrap(rng, Tratio[wres])
    Tdmess = Tratio_label + ': %.3g +/- %.3g' % (
        Tstat['mean'], Tstat['mean_err'],
    )
    Tskewmess = 'skew: %.3g +/- %.3g' % (
        Tstat['skew'], Tstat['skew_err'],
    )
    # Tskewmess = 'skew: %.3g' % (
    #     Tstat['skew'],
    # )
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
    axs[1, 0].text(
        xloc, 0.85, Tskewmess,
        transform=axs[1, 0].transAxes,
    )

    # e diff
    e1diff = e1 - e1psf
    e2diff = e2 - e2psf
    m1stat = bootstrap(rng, e1diff[wres])
    m2stat = bootstrap(rng, e2diff[wres])

    e1dmess = r'$\Delta$e$_1$: %.3g +/- %.3g' % (
        m1stat['mean'], m1stat['mean_err'],
    )
    e2dmess = r'$\Delta$e$_2$: %.3g +/- %.3g' % (
        m2stat['mean'], m2stat['mean_err'],
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


def plot_star_stats_bys2n(st, pixel_scale=0.2, show=False, frac=None, nbin=5):
    import matplotlib.pyplot as mplt
    import numpy as np
    import esutil as eu

    rng = np.random.default_rng(seed=st.size)

    fig, axs = mplt.subplots(nrows=2, figsize=(7, 7))

    logic = get_logic(st)
    logic &= st['reserved']

    w, = np.where(logic)

    s2n = st['psf_flux'][w] / st['psf_flux_err'][w]
    T = st['am_T'][w]
    Tpsf = st['am_psf_T'][w]
    Tratio = T / Tpsf - 1

    nperbin = int(s2n.size / nbin)
    print(f'nperbin: {nperbin}')

    binner = eu.stat.Binner(s2n)
    binner.dohist(nperbin=nperbin, rev=True)

    s2ns = np.zeros(nbin)
    means = np.zeros(nbin)
    mean_errs = np.zeros(nbin)
    skews = np.zeros(nbin)
    skew_errs = np.zeros(nbin)

    rev = binner['rev']
    for i in range(binner['hist'].size):
        if rev[i] != rev[i+1]:
            wsub = rev[rev[i]:rev[i+1]]

            stats = bootstrap(rng, Tratio[wsub])

            s2ns[i] = s2n[wsub].mean()

            means[i] = stats['mean']
            mean_errs[i] = stats['mean_err']
            skews[i] = stats['skew']
            skew_errs[i] = stats['skew_err']

    s2n_lim = [50, 4000]

    # fwhm
    Tratio_label = r'T/T$_{\mathrm{PSF}} - 1$'
    axs[0].set(
        xlabel='S/N',
        ylabel=Tratio_label,
        xlim=s2n_lim,
        ylim=[-0.00095, 0.00195],
    )
    axs[0].axhline(0, color='black')
    axs[1].set(
        xlabel='S/N',
        ylabel=Tratio_label + ' skew',
        xlim=s2n_lim,
        ylim=[-0.41, 0.11],
    )
    axs[1].axhline(0, color='black')
    axs[0].set_xscale('log')
    axs[1].set_xscale('log')

    axs[0].errorbar(
        s2ns, means, mean_errs,
    )
    axs[1].errorbar(
        s2ns, skews, skew_errs,
    )
    if show:
        mplt.show()

    return fig, axs


def get_logic(st):
    return (
        (st['am_flags'] == 0)
        & (st['am_psf_flags'] == 0)
        & (st['psf_flux_err'] > 0)
    )


def bootstrap(rng, vals, nsamp=1000):
    import numpy as np
    import scipy.stats
    import esutil as eu

    means = np.zeros(nsamp)
    skews = np.zeros(nsamp)
    for i in range(nsamp):
        ind = rng.integers(0, vals.size, size=vals.size)
        means[i] = vals[ind].mean()
        means[i], _, _ind = eu.stat.sigma_clip(vals[ind], get_indices=True)
        skews[i] = scipy.stats.skew(vals[ind[_ind]])

    mn = means.mean()
    err = means.std()
    skew = skews.mean()
    skew_err = skews.std()
    return {
        'mean': mn,
        'mean_err': err,
        'skew': skew,
        'skew_err': skew_err,
    }
