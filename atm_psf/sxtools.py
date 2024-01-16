import numpy as np

KERNEL = np.array(
    [[0.00262617, 0.00624612, 0.01050468, 0.01249224, 0.01050468, 0.00624612, 0.00262617],  # noqa
     [0.00624612, 0.01485586, 0.02498447, 0.02971171, 0.02498447, 0.01485586, 0.00624612],  # noqa
     [0.01050468, 0.02498447, 0.0420187 , 0.04996894, 0.0420187 , 0.02498447, 0.01050468],  # noqa
     [0.01249224, 0.02971171, 0.04996894, 0.05942342, 0.04996894, 0.02971171, 0.01249224],  # noqa
     [0.01050468, 0.02498447, 0.0420187 , 0.04996894, 0.0420187 , 0.02498447, 0.01050468],  # noqa
     [0.00624612, 0.01485586, 0.02498447, 0.02971171, 0.02498447, 0.01485586, 0.00624612],  # noqa
     [0.00262617, 0.00624612, 0.01050468, 0.01249224, 0.01050468, 0.00624612, 0.00262617]]  # noqa
)
SX_CONFIG = {

    'deblend_cont': 0.001,

    'deblend_nthresh': 64,

    'minarea': 4,

    'filter_type': 'conv',

    # 7x7 convolution mask of a gaussian PSF with FWHM = 0.8 arcsec
    'filter_kernel': KERNEL,
}


def sxprocess(exp):
    import sxdes
    import numpy as np

    var = np.median(exp.variance.array)
    cat, seg = sxdes.run_sep(
        image=exp.image.array,
        noise=np.sqrt(var),
        config=SX_CONFIG,
    )
    return cat, seg


def sxprocess_and_show(exp):
    import matplotlib.pyplot as mplt
    cat, seg = sxprocess(exp)

    fig, axs = mplt.subplots(ncols=2)
    axs[0].imshow(exp.image.array)
    axs[0].scatter(
        cat['x'],
        cat['y'],
        s=1,
        color='red',
    )

    axs[1].set(
        xlabel='flux',
        ylabel='FWHM [arcsec]',
    )
    axs[1].set_xscale('log')

    w, = np.where(cat['flux_auto'] > 1)
    axs[1].scatter(
        cat['flux_auto'][w],
        cat['flux_radius'][w] * 2 * 0.2,
        s=1,
    )
    mplt.show()
