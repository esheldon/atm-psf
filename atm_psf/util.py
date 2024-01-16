def get_fwhm(img, cen=None, show=False, save=False, scale=0.2):
    import numpy as np
    import matplotlib.pyplot as plt
    import espy.images

    r, improf = espy.images.get_profile(img, cen=cen)
    r = r * scale
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
