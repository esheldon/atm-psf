import numpy as np
import atm_psf


seed = 85
rng = np.random.RandomState(seed)

# loads the image, subtractes the sky using sky_level in truth catalog, loads
# WCS from the header, and adds a fake PSF with fwhm=0.8 for detection
exp = atm_psf.exposures.fits_to_exposure(
    fname='eimage-00229252-0-i-R11_S11-det040.fits',
    truth='truth-00229252-0-i-R11_S11-det040.fits',
    fwhm=0.8,
)


sources, training, reserved = atm_psf.io.load_sources_and_candidates(
    fname='/tmp/test.pkl',
)

candidates = atm_psf.pifftools.make_psf_candidates(
    sources=sources[training],
    exposure=exp,
)
# import IPython; IPython.embed()

piff_psf = atm_psf.pifftools.run_piff(candidates, exp)

atm_psf.io.save_stack_piff(fname='/tmp/testpiff.pkl', piff_psf=piff_psf)
