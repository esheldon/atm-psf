from .constants import SCALE


def config_file_to_run(path):
    import os
    bname = os.path.basename(path)

    if '.yaml' not in bname:
        raise RuntimeError(f'Expected .yaml file, got {path}')

    run = bname.replace('.yaml', '')
    assert run != bname

    return run


def T_to_fwhm(T):
    import numpy as np
    sigma = np.sqrt(T/2)
    return sigma * 2.3548200450309493 * SCALE


def get_image_fwhm(img, cen=None, show=False, save=False):
    import numpy as np
    import matplotlib.pyplot as plt
    import espy.images

    r, improf = espy.images.get_profile(img, cen=cen)
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


def makedir(d):
    import os
    if not os.path.exists(d):
        print('making dir:', d)
        try:
            os.makedirs(d)
        except Exception:
            pass


def split_ccds_string(ccds_str):
    n = len(ccds_str)
    # cut of the [ and ]
    keep = ccds_str[1:n-1]
    return [int(ccd) for ccd in keep.split(',')]


def get_band(val):
    try:
        len(val)
        return val
    except TypeError:
        return 'ugrizy'[val]


def run_sim_by_type(config, **kw):
    from .process import run_sim
    from .runner_fast import run_fast_sim

    sim_type = config['sim_type']

    print(f'running sim_type: {sim_type}')
    print('importing imsim....')
    _do_import_imsim()

    if sim_type == 'imsim':
        run_sim(config=config, **kw)
    elif sim_type == 'fast':
        run_fast_sim(config=config, **kw)
    else:
        raise ValueError(f'bad sim_type {sim_type}')


def _do_import_imsim():
    import time

    tm0 = time.time()
    import imsim  # noqa
    tm = (time.time() - tm0) / 60

    print(f'took {tm:.1f} minutes')
