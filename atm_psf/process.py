def run_sim(rng, config, instcat, ccds, use_existing=False):
    import os
    import numpy as np
    import logging
    import galsim
    import imsim
    import montauk
    from . import io
    from pprint import pformat
    from .logging import setup_logging

    setup_logging('info')
    logger = logging.getLogger('process.run_sim')

    logger.info('sim config:')
    logger.info('\n' + pformat(config))

    gs_rng = galsim.BaseDeviate(rng.integers(0, 2**60))

    logger.info(f'loading opsim data from {instcat}')
    obsdata = montauk.opsim_data.load_obsdata_from_instcat(instcat)

    logger.info(f'making {config["psf"]["type"]} PSF')

    if config['psf']['type'] == 'psfws':
        psf = montauk.psfws.make_psfws_psf(
            obsdata=obsdata,
            gs_rng=gs_rng,
            psf_config=config['psf']['options'],
        )
    else:
        psf = montauk.fixed_psf.make_fixed_psf(
            type=config['psf']['type'],
            options=config['psf']['options'],
        )

    sky_model = imsim.SkyModel(
        exptime=obsdata['exptime'],
        mjd=obsdata['mjd'],
        bandpass=obsdata['bandpass'],
    )
    if config['sky_gradient']:
        sky_gradient = montauk.sky.FixedSkyGradient(sky_model)
    else:
        sky_gradient = None

    if config['dcr']:
        dcr = montauk.dcr.DCRMaker(
            bandpass=obsdata['bandpass'],
            hour_angle=obsdata['HA'],
        )
    else:
        dcr = None

    cosmics = montauk.cosmic_rays.CosmicRays(
        cosmic_ray_rate=config['cosmic_ray_rate'],
        exptime=obsdata['exptime'],
        gs_rng=gs_rng,
    )

    if config['tree_rings']:
        tree_rings = montauk.tree_rings.make_tree_rings(ccds)
    else:
        tree_rings = None

    for iccd, ccd in enumerate(ccds):
        logger.info('-' * 70)
        logger.info(f'ccd: {ccd} {iccd+1}/{len(ccds)}')

        # e.g. simdata-00355204-0-i-R14_S00-det063.fits
        fname = io.get_sim_output_fname(
            obsid=obsdata['obshistid'],
            ccd=ccd,
            band=obsdata['band'],
        )
        if use_existing and os.path.exists(fname):
            logger.info(f'using existing file {fname}')
            continue

        diffraction_fft = imsim.stamp.DiffractionFFT(
            exptime=obsdata['exptime'],
            altitude=obsdata['altitude'],
            azimuth=obsdata['azimuth'],
            rotTelPos=obsdata['rotTelPos'],
        )

        dm_detector = montauk.camera.make_dm_detector(ccd)

        wcs, icrf_to_field = montauk.wcs.make_batoid_wcs(
            obsdata=obsdata, dm_detector=dm_detector,
        )
        logger.info(f'loading objects from {instcat}')
        cat = imsim.instcat.InstCatalog(file_name=instcat, wcs=wcs)

        if config['vignetting']:
            vignetter = montauk.vignetting.Vignetter(dm_detector)
        else:
            vignetter = None

        if montauk.fringing.should_apply_fringing(
            band=obsdata['band'], dm_detector=dm_detector,
        ):
            fringer = montauk.fringing.Fringer(
                boresight=obsdata['boresight'], dm_detector=dm_detector,
            )
        else:
            fringer = None

        sensor = montauk.sensor.make_sensor(
            dm_detector=dm_detector,
            tree_rings=tree_rings,
            gs_rng=gs_rng,
        )

        optics = montauk.optics.OpticsMaker(
            altitude=obsdata['altitude'],
            azimuth=obsdata['azimuth'],
            boresight=obsdata['boresight'],
            rot_tel_pos=obsdata['rotTelPos'],
            band=obsdata['band'],
            dm_detector=dm_detector,
            wcs=wcs,
            icrf_to_field=icrf_to_field,
        )

        photon_ops_maker = montauk.photon_ops.PhotonOpsMaker(
            exptime=obsdata['exptime'],
            band=obsdata['band'],
            dcr=dcr,
            optics=optics,
        )

        artist = montauk.artist.Artist(
            bandpass=obsdata['bandpass'],
            sensor=sensor,
            photon_ops_maker=photon_ops_maker,
            diffraction_fft=diffraction_fft,
            gs_rng=gs_rng,
        )

        calc_xy_indices = rng.choice(
            cat.getNObjects(), size=200, replace=False,
        )

        image, sky_image, truth = montauk.runner.run_sim(
            rng=rng,
            cat=cat,
            obsdata=obsdata,
            artist=artist,
            psf=psf,
            wcs=wcs,
            sky_model=sky_model,
            sensor=sensor,
            dm_detector=dm_detector,
            cosmics=cosmics,
            sky_gradient=sky_gradient,
            vignetting=vignetter,
            fringing=fringer,
            calc_xy_indices=calc_xy_indices,
            apply_pixel_areas=config['apply_pixel_areas'],
        )
        mid = calc_xy_indices.size // 2
        final_wcs, wcs_stats = montauk.wcs.fit_wcs(
            x=truth['realized_x'][calc_xy_indices],
            y=truth['realized_y'][calc_xy_indices],
            ra=truth['ra'][calc_xy_indices],
            dec=truth['dec'][calc_xy_indices],
            units=galsim.degrees,
            itrain=np.arange(mid),
            ireserve=np.arange(mid, calc_xy_indices.size),
        )
        image.wcs = final_wcs
        sky_image.wcs = final_wcs

        logging.info(f'writing to: {fname}')

        extra = {'det_name': dm_detector.getName()}
        extra.update(wcs_stats)

        io.save_sim_data(
            fname=fname, image=image, sky_image=sky_image, truth=truth,
            obsdata=obsdata,
            extra=extra,
        )


def run_sim_and_piff(
    rng,
    run_config,
    sim_config,
    opsim_db,
    obsid,
    instcat,
    ccds,
    cleanup=True,
    use_existing=False,
    plot_dir=None,
):
    """
    Run the simulation using galsim and run piff on the image

    Parameters
    ----------
    rng: np.random.default_rng
        The random number generator
    run_config: dict
        The run configuration
    sim_config: dict
        Simulation configuration
    opsim_db: str
        Path to opsim database
    obsid: int
        The observation id
    instcat: str
        Path for the the output instcat
    ccds: list of int
        List of CCD numbers
    cleanup: bool, optional
        If set to True, remove the simulated data, the image, truth and instcat
        files.  Default True.
    show: bool
        If set to True, show plots
    """

    import os
    import numpy as np
    import montauk
    from . import io

    # generate these now so runs with and without existing instcat
    # are consistent
    instcat_rng = np.random.default_rng(rng.integers(0, 2**60))
    sim_rng = np.random.default_rng(rng.integers(0, 2**60))
    piff_rng = np.random.default_rng(rng.integers(0, 2**60))

    instcat_out = get_instcat_output_path(obsid)

    if not os.path.exists(instcat_out) or not use_existing:
        dup = run_config.get('dup', 1)
        run_make_instcat(
            rng=instcat_rng,
            run_config=run_config,
            opsim_db=opsim_db,
            obsid=obsid,
            instcat=instcat,
            instcat_out=instcat_out,
            ccds=ccds,
            dup=dup,
        )

    do_run_sim = True
    if use_existing:
        obsdata = montauk.opsim_data.load_obsdata_from_instcat(instcat_out)
        fnames = [
            io.get_sim_output_fname(
                obsid=obsdata['obshistid'],
                ccd=ccd,
                band=obsdata['band'],
            )
            for ccd in ccds
        ]
        if all([os.path.exists(fname) for fname in fnames]):
            do_run_sim = False

    if do_run_sim:
        run_sim(
            rng=sim_rng,
            config=sim_config,
            instcat=instcat_out,
            ccds=ccds,
        )

    obsdata = montauk.opsim_data.load_obsdata_from_instcat(instcat_out)

    for ccd in ccds:

        fname = io.get_sim_output_fname(
            obsid=obsdata['obshistid'],
            ccd=ccd,
            band=obsdata['band'],
        )
        piff_file = io.get_piff_output_fname(
            obsid=obsdata['obshistid'],
            ccd=ccd,
            band=obsdata['band'],
        )
        source_file = io.get_source_output_fname(
            obsid=obsdata['obshistid'],
            ccd=ccd,
            band=obsdata['band'],
        )

        process_image_with_piff(
            rng=piff_rng,
            fname=fname,
            piff_file=piff_file,
            source_file=source_file,
            piff_config=run_config.get('piff', None),
            plot_dir=plot_dir,
        )
        if cleanup:
            _remove_file(fname)


def run_sim_and_nnpsf(
    rng,
    run_config,
    sim_config,
    opsim_db,
    obsid,
    instcat,
    ccds,
    cleanup=True,
    use_existing=False,
    plot_dir=None,
):
    """
    Run the simulation using galsim and run piff on the image

    Parameters
    ----------
    rng: np.random.default_rng
        The random number generator
    run_config: dict
        The run configuration
    sim_config: dict
        Simulation configuration
    opsim_db: str
        Path to opsim database
    obsid: int
        The observation id
    instcat: str
        Path for the the output instcat
    ccds: list of int
        List of CCD numbers
    cleanup: bool, optional
        If set to True, remove the simulated data, the image, truth and instcat
        files.  Default True.
    show: bool
        If set to True, show plots
    """

    import os
    import numpy as np
    import montauk
    from . import io

    # generate these now so runs with and without existing instcat
    # are consistent
    instcat_rng = np.random.default_rng(rng.integers(0, 2**60))
    sim_rng = np.random.default_rng(rng.integers(0, 2**60))
    nnpsf_rng = np.random.default_rng(rng.integers(0, 2**60))

    instcat_out = get_instcat_output_path(obsid)

    if not os.path.exists(instcat_out) or not use_existing:
        dup = run_config.get('dup', 1)
        run_make_instcat(
            rng=instcat_rng,
            run_config=run_config,
            opsim_db=opsim_db,
            obsid=obsid,
            instcat=instcat,
            instcat_out=instcat_out,
            ccds=ccds,
            dup=dup,
        )

    do_run_sim = True
    if use_existing:
        obsdata = montauk.opsim_data.load_obsdata_from_instcat(instcat_out)
        fnames = [
            io.get_sim_output_fname(
                obsid=obsdata['obshistid'],
                ccd=ccd,
                band=obsdata['band'],
            )
            for ccd in ccds
        ]
        if all([os.path.exists(fname) for fname in fnames]):
            do_run_sim = False

    if do_run_sim:
        run_sim(
            rng=sim_rng,
            config=sim_config,
            instcat=instcat_out,
            ccds=ccds,
        )

    obsdata = montauk.opsim_data.load_obsdata_from_instcat(instcat_out)

    for ccd in ccds:

        fname = io.get_sim_output_fname(
            obsid=obsdata['obshistid'],
            ccd=ccd,
            band=obsdata['band'],
        )
        nnpsf_file = io.get_nnpsf_output_fname(
            obsid=obsdata['obshistid'],
            ccd=ccd,
            band=obsdata['band'],
        )
        process_image_with_nnpsf(
            rng=nnpsf_rng,
            fname=fname,
            output=nnpsf_file,
            config=run_config.get('nnpsf', None),
            plot_dir=plot_dir,
        )
        if cleanup:
            _remove_file(fname)


def get_instcat_output_path(obsid):
    import os
    outdir = '%08d' % obsid
    return os.path.join(outdir, 'instcat.txt')


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


def run_make_instcat(
    rng, run_config, opsim_db, obsid, instcat, instcat_out, ccds, dup=1,
):
    """
    Run the code to make a new instcat from an input one and a pointing from
    the obsim db

    Parameters
    ----------
    rng: np.random.default_rng
        The random number generator
    run_config: dict
        The run configuration
    opsim_db: str
        Path to opsim database
    obsid: int
        The observation id
    instcat: str
        Path for the input instcat
    instcat_out: str
        Path for the output instcat
    ccds: list of int
        List of CCD numbers
    dup: int, optional
        Number of times to duplicate, with random ra/dec
    """
    import sqlite3
    from . import instcat_tools

    print('connecting to:', opsim_db)

    magmin = run_config.get('magmin', -1000)
    with sqlite3.connect(opsim_db) as conn:
        instcat_tools.replace_instcat_from_db(
            rng=rng,
            fname=instcat,
            conn=conn,
            obsid=obsid,
            output_fname=instcat_out,
            allowed_include=run_config['allowed_include'],
            # sed='starSED/phoSimMLT/lte034-4.5-1.0a+0.4.BT-Settl.spec.gz',
            sed=run_config.get('sed', None),
            selector=lambda d: d['magnorm'] > magmin,
            galaxy_file=run_config.get('galaxy_file', None),
            ccds=ccds,
            dup=dup,
        )


GALSIM_COMMAND = r"""
galsim %(imsim_config)s \
    input.instance_catalog.file_name="%(instcat)s" \
    output.nfiles=%(nfiles)d \
    output.det_num="%(ccds)s"
"""


def run_galsim(imsim_config, instcat, ccds):
    import os
    nfiles = len(ccds)

    ccdstr = '[' + ','.join([str(ccd) for ccd in ccds]) + ']'

    command = GALSIM_COMMAND % {
        'imsim_config': imsim_config,
        'instcat': instcat,
        'nfiles': nfiles,
        'ccds': ccdstr,
    }
    print(command)
    res = os.system(command)
    if res != 0:
        raise RuntimeError('failed galsim call')


def process_image_with_piff(
    rng,
    fname,
    piff_file,
    source_file,
    piff_config=None,
    plot_dir=None,
):
    """
    process the image using piff

    Parameters
    ----------
    rng: np.random.default_rng
        Random number generator
    image_file: str
        Path to image file
    truth_file: str
        Path to truth catalog file
    piff_file: str
        Output file path
    source_file: str
        Output catlaog path
    piff_config: dict, optional
        Dict to configure the piff run. Can have entries
            nstars_min
            spatial_order
    show: bool
        If set to True, show plots
    """
    import numpy as np
    from . import exposures
    from . import measure
    from . import select
    from . import pifftools
    from . import io
    from pprint import pformat
    from .logging import setup_logging
    import logging

    setup_logging('info')

    logger = logging.getLogger('process.process_image_with_piff')

    piff_config = pifftools.get_piff_config(piff_config)
    logger.info('\n' + pformat(piff_config))

    alldata = {'file': fname}

    # loads the image, subtractes the sky using sky_level in truth catalog,
    # loads WCS from the header, and adds a fake PSF with fwhm=0.8 for
    # detection
    # rng is only used for noise in the fixed gaussian psf
    exp, hdr = exposures.fits_to_exposure(fname=fname, rng=rng)
    instcat_meta = {key: hdr[key] for key in hdr.keys()}

    detmeas = measure.DetectMeasurer(exposure=exp, rng=rng)
    detmeas.detect()
    detmeas.measure()
    sources = detmeas.sources

    # find stars in the size/flux diagram
    # note this is a bool array
    star_select = select.select_stars(sources, plot_dir=plot_dir)

    alldata['sources'] = sources
    alldata['star_select'] = star_select
    # alldata['airmass'] = hdr['AMSTART']
    alldata['airmass'] = hdr['airmass']
    alldata['filter'] = hdr['band']
    alldata['instcat_meta'] = instcat_meta

    nstars = star_select.sum()
    if nstars >= piff_config['nstars_min']:

        # split into training and reserved/validation sets
        # these are again bool arrays with size length(sources)
        reserved = pifftools.split_candidates(
            rng=rng, star_select=star_select, reserve_frac=0.2,
        )

        candidates = pifftools.make_psf_candidates(
            sources=sources[star_select],
            exposure=exp,
        )

        logger.info('running piff')
        piff_psf, meta, not_kept = pifftools.run_piff(
            psf_candidates=candidates,
            reserved=reserved[star_select],
            exposure=exp,
            spatial_order=piff_config['spatial_order'],
            plot_dir=plot_dir,
        )
        # star_select is a full boolean, so we need to get the corresponding
        # indices
        nout = not_kept.sum()
        if nout > 0:
            logger.info(f'skipped: {nout} candidates')
            ws, = np.where(star_select)
            star_select[ws[not_kept]] = False
            reserved[ws[not_kept]] = False

        # remeasure with new psf
        exp.setPsf(piff_psf)
        detmeas.measure()
        detmeas.measure_ngmix()

        logger.info(f'saving piff to: {piff_file}')
        io.save_stack_piff(fname=piff_file, piff_psf=piff_psf)

        # save sources and candidate list
        alldata.update({
            'reserved': reserved,
            'ngmix_result': detmeas.ngmix_result,
        })
        alldata.update(meta)
    else:
        logger.info(f'got nstars {nstars} < {piff_config["nstars_min"]}')
        logger.info(f'saving None piff to: {piff_file}')
        io.save_stack_piff(fname=piff_file, piff_psf=None)

    logger.info(f'saving sources data to: {source_file}')
    io.save_source_data(fname=source_file, data=alldata)


def process_image_with_nnpsf(
    rng,
    fname,
    source_file,
    config=None,
    plot_dir=None,
):
    """
    process the image using nnpsf

    Parameters
    ----------
    rng: np.random.default_rng
        Random number generator
    fname: str
        Path to image file
    source_file: str
        Output catlaog path
    config: dict, optional
        Config for nnpsf
    plot_dir: str
        directory to put plots
    """
    import fitsio
    import numpy as np
    import nnpsf
    import torch
    from nnpsf.measure import get_models_and_psf_fluxes
    from nnpsf.plotting import show_sim_cat_with_stars, show_catalog_statistics
    from nnpsf.validation import plot_validation
    from nnpsf.io import load_montauk

    print('cuda available:', torch.cuda.is_available())
    print('cuda device count:', torch.cuda.device_count())
    if torch.cuda.device_count() > 0:
        print('cuda current device:', torch.cuda.current_device())

    torch.manual_seed(rng.integers(0, 2**63))

    data = load_montauk(fname)

    fitting_data = nnpsf.process_image(
        data=data,
        fit_config=config,
        rng=rng,
    )

    cat = fitting_data['cat']

    if plot_dir is not None:
        image_plot = source_file.replace('.fits', '') + '-image.jpg'
        show_sim_cat_with_stars(data=data, cat=cat, fname=image_plot)

        stats_plot = source_file.replace('.fits', '') + '-stats.png'
        show_catalog_statistics(cat=cat, fname=stats_plot)

        valid_plot = source_file.replace('.fits', '') + '-valid.jpg'

        stamps = fitting_data['stamp_data']['stamps']
        var_stamps = fitting_data['var_stamp_data']['stamps']
        ivalid, = np.where(cat['reserved'])
        valid_models, _ = get_models_and_psf_fluxes(
            stamps=stamps[ivalid],
            variance=var_stamps[ivalid],
            x=cat['x'][ivalid],
            y=cat['y'][ivalid],
            model=fitting_data['model'],
        )
        print('validating:', ivalid.size, 'stars')
        plot_validation(
            stamps=stamps[ivalid],
            models=valid_models,
            fname=valid_plot,
        )

    print('writing to:', source_file)
    fitsio.write(source_file, cat, clobber=True)


def _load_instcat_meta_from_dir(image_file):
    """
    we always call it instcat.txt, just get the directory
    """
    import os
    from . import instcat_tools

    dname = os.path.dirname(image_file)
    if dname == '':
        dname = './'

    instcat_file = os.path.join(
        dname,
        'instcat.txt',
    )
    print('loading instcat:', instcat_file)
    return instcat_tools.read_instcat_meta(instcat_file)
