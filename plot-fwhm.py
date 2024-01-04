def main(flist, outfile):
    import fitsio
    import matplotlib.pyplot as mplt
    from esutil.stat import get_stats, print_stats
    import numpy as np

    fwhms = []
    for f in flist:
        t = fitsio.read(f)
        fwhms.append(t['fwhm'].mean())

    fwhms = np.array(fwhms)
    print_stats(fwhms)
    stats = get_stats(fwhms)
    medval = np.median(fwhms)

    text = r'mean: %.2f $\pm$ %.2f' % (stats['mean'], stats['err'])
    textmed = 'median: %.2f' % medval

    fig, ax = mplt.subplots(figsize=(6, 4))
    ax.set(xlabel='PSF FWHM')
    hist, xvals, _ = ax.hist(fwhms, bins=20)
    ax.text(xvals.mean()*1.3, 0.9*hist.max(), text)
    ax.text(xvals.mean()*1.3, 0.85*hist.max(), textmed)

    print('writing:', outfile)
    mplt.savefig(outfile, dpi=150)


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--flist', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args.flist, args.output)
