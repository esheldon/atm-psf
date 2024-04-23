def run_sim_and_piff(
    run_config,
    imsim_config,
    opsim_db,
    obsid,
    instcat,
    ccds,
    seed,
    nstars_min=50,
    cleanup=True,
    no_find_sky=False,
    show=False,
):
    """
    Run the simulation using galsim and run piff on the image

    Parameters
    ----------
    run_config: dict
        The run configuration
    imsim_config: str
        Path to galsim config file to run imsim
    opsim_db: str
        Path to opsim database
    obsid: int
        The observation id
    instcat: str
        Path for the the output instcat
    ccds: list of int
        List of CCD numbers
    seed: int
        Seed for random number generator
    nstars_min: int, optional
        Minimum number of stars required to run PIFF, default 50
    cleanup: bool, optional
        If set to True, remove the simulated data, the image, truth and instcat
        files.  Default True.
    no_find_sky: bool
        If set to True, find the sky rather than just use sky from catalog
    show: bool
        If set to True, show plots
    """

    import numpy as np

    rng = np.random.default_rng(seed)

    sseed = rng.integers(0, 2**31)
    run_simulation(
        run_config=run_config,
        imsim_config=imsim_config,
        opsim_db=opsim_db,
        obsid=obsid,
        instcat=instcat,
        ccds=ccds,
        seed=sseed,
    )

    tmp = ccds.replace('[', '').replace(']', '')
    ccdnums = [int(s) for s in tmp.split(',')]

    for ccd in ccdnums:

        image_file, truth_file, piff_file, source_file = _get_paths(
            obsid=obsid, ccd=ccd,
        )

        pseed = rng.integers(0, 2**31)

        process_image_with_piff(
            image_file=image_file,
            truth_file=truth_file,
            piff_file=piff_file,
            source_file=source_file,
            seed=pseed,
            nstars_min=nstars_min,
            no_find_sky=no_find_sky,
            show=show,
        )
        if cleanup:
            _remove_file(image_file)
            _remove_file(truth_file)

    # if cleanup:
    #     instcat_file = _get_instcat_path(obsid)
    #     _remove_file(instcat_file)


def _remove_file(fname):
    import os
    print('removing:', fname)
    os.remove(fname)


def _obsid_to_dirname(obsid):
    return f'{obsid:08d}'


def _get_instcat_path(obsid):
    import os
    dname = _obsid_to_dirname(obsid)
    return os.path.join(dname, 'instcat.txt')


def _get_paths(obsid, ccd):
    from glob import glob
    import os

    dname = _obsid_to_dirname(obsid)

    pattern = f'{dname}/eimage-{obsid:08d}-*-det{ccd:03d}.fits'
    flist = glob(pattern)
    if len(flist) != 1:
        raise RuntimeError(f'missing file: {pattern}')
    image_file = flist[0]
    truth_file = image_file.replace('eimage', 'truth')

    bname = os.path.basename(image_file)

    piff_name = bname.replace(
        '.fits', '.pkl'
    ).replace(
        'eimage', 'piff'
    )
    assert piff_name != bname
    source_name = piff_name.replace(
        'piff', 'source',
    ).replace(
        '.pkl', '.fits',
    )
    assert source_name != piff_name
    piff_file = os.path.join(dname, piff_name)
    source_file = os.path.join(dname, source_name)

    return image_file, truth_file, piff_file, source_file


def run_simulation(
    run_config, imsim_config, opsim_db, obsid, instcat, ccds, seed,
):
    """
    Run the simulation using galsim

    Parameters
    ----------
    run_config: dict
        The run configuration
    imsim_config: str
        Path to galsim config file to run imsim
    opsim_db: str
        Path to opsim database
    obsid: int
        The observation id
    instcat: str
        Path for the the output instcat
    ccds: list of int
        List of CCD numbers
    seed: int
        Seed for random number generator
    """
    import os
    import sqlite3
    import atm_psf
    import numpy as np

    rng = np.random.default_rng(seed)

    print('connecting to:', opsim_db)

    with sqlite3.connect(opsim_db) as conn:
        # galsim will also write to this dir
        outdir = '%08d' % obsid

        instcat_out = os.path.join(outdir, 'instcat.txt')

        if not os.path.exists(instcat_out):
            print('making instcat')
            atm_psf.instcat_tools.replace_instcat_from_db(
                rng=rng,
                fname=instcat,
                conn=conn,
                obsid=obsid,
                output_fname=instcat_out,
                allowed_include=run_config['allowed_include'],
                # sed='starSED/phoSimMLT/lte034-4.5-1.0a+0.4.BT-Settl.spec.gz',
                selector=lambda d: d['magnorm'] > 17
            )

    # galsim will write to subdir, so chdir to it
    print('running galsim')
    _run_galsim(
        imsim_config=imsim_config,
        instcat=instcat_out,
        ccds=ccds,
    )


GALSIM_COMMAND = r"""
galsim %(imsim_config)s \
    input.instance_catalog.file_name="%(instcat)s" \
    output.nfiles=%(nfiles)d \
    output.det_num="%(ccds)s"
"""


def _run_galsim(imsim_config, instcat, ccds):
    import os
    nfiles = len(ccds.split(','))

    command = GALSIM_COMMAND % {
        'imsim_config': imsim_config,
        'instcat': instcat,
        'nfiles': nfiles,
        'ccds': ccds,
    }
    print(command)
    res = os.system(command)
    if res != 0:
        raise RuntimeError('failed galsim call')


def process_image_with_piff(
    image_file, truth_file, piff_file, source_file, seed, nstars_min=50,
    no_find_sky=False,
    show=False,
):
    """
    process the image using piff

    Parameters
    ----------
    image_file: str
        Path to image file
    truth_file: str
        Path to truth catalog file
    piff_file: str
        Output file path
    source_file: str
        Output catlaog path
    seed: int
        Seed for random number generator
    nstars_min: int
        Minimum number of stars required to run PIFF, default 50
    no_find_sky: bool
        If set to True, find the sky rather than just use sky from catalog
    show: bool
        If set to True, show plots
    """
    import numpy as np
    import atm_psf

    rng = np.random.RandomState(seed)
    alldata = {
        'seed': seed,
        'image_file': image_file,
        'truth_file': truth_file,
    }

    # loads the image, subtractes the sky using sky_level in truth catalog,
    # loads WCS from the header, and adds a fake PSF with fwhm=0.8 for
    # detection
    exp, hdr = atm_psf.exposures.fits_to_exposure(
        fname=image_file,
        truth=truth_file,
        rng=rng,
        no_find_sky=no_find_sky,
    )
    instcat_meta = _load_instcat_meta_from_dir(image_file)

    detmeas = atm_psf.measure.DetectMeasurer(exposure=exp, rng=rng)
    detmeas.detect()
    detmeas.measure()
    sources = detmeas.sources

    # run detection
    # sources = atm_psf.measure.detect_and_measure(exposure=exp, rng=rng)

    # find stars in the size/flux diagram
    # note this is a bool array
    star_select = atm_psf.select.select_stars(sources, show=show)

    alldata['sources'] = sources
    alldata['star_select'] = star_select
    alldata['airmass'] = hdr['AMSTART']
    alldata['filter'] = hdr['FILTER']
    alldata['instcat_meta'] = instcat_meta

    nstars = star_select.sum()
    if nstars >= nstars_min:

        # split into training and reserved/validation sets
        # these are again bool arrays with size length(sources)
        reserved = atm_psf.pifftools.split_candidates(
            rng=rng, star_select=star_select, reserve_frac=0.2,
        )

        candidates = atm_psf.pifftools.make_psf_candidates(
            sources=sources[star_select],
            exposure=exp,
        )

        print('running piff')
        piff_psf, meta, not_kept = atm_psf.pifftools.run_piff(
            psf_candidates=candidates,
            reserved=reserved[star_select],
            exposure=exp,
            show=show,
        )
        # star_select is a full boolean, so we need to get the corresponding
        # indices
        nout = not_kept.sum()
        if nout > 0:
            print('skipped:', nout, 'candidates')
            ws, = np.where(star_select)
            star_select[ws[not_kept]] = False
            reserved[ws[not_kept]] = False

        # remeasure with new psf
        exp.setPsf(piff_psf)
        detmeas.measure()
        detmeas.measure_ngmix()

        print('saving piff to:', piff_file)
        atm_psf.io.save_stack_piff(fname=piff_file, piff_psf=piff_psf)

        # save sources and candidate list
        alldata.update({
            'reserved': reserved,
            'image_file': image_file,
            'truth_file': truth_file,
            'ngmix_result': detmeas.ngmix_result,
        })
        alldata.update(meta)
    else:
        print(f'got nstars {nstars} < {nstars_min}')
        print('saving None piff to:', piff_file)
        atm_psf.io.save_stack_piff(fname=piff_file, piff_psf=None)

    print('saving sources data to:', source_file)
    atm_psf.io.save_source_data(fname=source_file, data=alldata)


def _load_instcat_meta_from_dir(image_file):
    """
    we always call it instcat.txt, just get the directory
    """
    import os
    import atm_psf

    dname = os.path.dirname(image_file)
    if dname == '':
        dname = './'

    instcat_file = os.path.join(
        dname,
        'instcat.txt',
    )
    print('loading instcat:', instcat_file)
    return atm_psf.instcat_tools.read_instcat_meta(instcat_file)
