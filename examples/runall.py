import numpy as np
import atm_psf


seed = 10
rng = np.random.RandomState(seed)

# loads the image, subtractes the sky using sky_level in truth catalog, loads
# WCS from the header, and adds a fake PSF with fwhm=0.8 for detection
exp = atm_psf.exposures.fits_to_exposure(
    fname='eimage-00229252-0-i-R11_S11-det040.fits',
    truth='truth-00229252-0-i-R11_S11-det040.fits',
    rng=rng,
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
reserved = atm_psf.pifftools.split_candidates(
    rng=rng, star_select=star_select, reserve_frac=0.2,
)

candidates = atm_psf.pifftools.make_psf_candidates(
    sources=sources[star_select],
    exposure=exp,
)

piff_psf, meta = atm_psf.pifftools.run_piff(
    psf_candidates=candidates,
    reserved=reserved[star_select],
    exposure=exp,
)

# remeasure with new psf
exp.setPsf(piff_psf)
detmeas.measure()

atm_psf.io.save_stack_piff(fname='/tmp/test-piff.pkl', piff_psf=piff_psf)

# save sources and candidate list
alldata = {
    'sources': sources,
    'star_select': star_select,
    'reserved': reserved,
}
alldata.update(meta)

atm_psf.io.save_source_data(fname='/tmp/test-data.pkl', data=alldata)
