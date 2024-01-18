import numpy as np
import atm_psf


seed = 10
rng = np.random.RandomState(seed)

# loads the image, subtractes the sky using sky_level in truth catalog, loads
# WCS from the header, and adds a fake PSF with fwhm=0.8 for detection
exp = atm_psf.exposures.fits_to_exposure(
    fname='eimage-00229252-0-i-R11_S11-det040.fits',
    truth='truth-00229252-0-i-R11_S11-det040.fits',
    fwhm=0.8,
)

detmeas = atm_psf.measure.DetectMeasurer(exposure=exp, rng=rng)
detmeas.detect()
detmeas.measure()
sources = detmeas.sources

# run detection
# sources = atm_psf.measure.detect_and_measure(exposure=exp, rng=rng)

# find stars in the size/flux diagram
# note this is a bool array
star_select = atm_psf.select.select_stars(sources)

# split into training and reserved/validation sets
# these are again bool arrays with size length(sources)
training, reserved = atm_psf.pifftools.split_candidates(
    rng=rng, star_select=star_select, reserve_frac=0.2,
)

# save sources and candidate list
atm_psf.io.save_sources_and_candidates(
    fname='/tmp/test.pkl',
    sources=sources,
    training=training,
    reserved=reserved,
)
