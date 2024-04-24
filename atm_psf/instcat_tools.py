# radius for disc of random points
INSTCAT_RADIUS = 2.1

# (4096 * 0.2 / 60)**2
CCD_AREA = 186.4  # arcmin
GAL_DISK_DENSITY = 600  # per sq arcmin
MEAN_DISK_CCD_NUM = 111341

FILTERMAP = {
    'u': 0,
    'g': 1,
    'r': 2,
    'i': 3,
    'z': 4,
    'y': 5,
}

RFILTERMAP = {
    0: 'u',
    1: 'g',
    2: 'r',
    3: 'i',
    4: 'z',
    5: 'y',
}

NAME_MAP = {
    'rightascension': 'fieldra',
    'declination': 'fieldDec',
    'mjd': 'observationStartMJD',
    'altitude': 'altitude',
    'azimuth': 'azimuth',
    'filter': 'filter',
    'rotskypos': 'rotSkyPos',
    'dist2moon': 'moonDistance',
    'moonalt': 'moonAlt',
    'moonphase': 'moonPhase',
    'moonra': 'moonRA',
    # 'nsnap': missing,
    'obshistid': 'observationId',
    'rottelpos': 'rotTelPos',
    # 'seed': to be set
    'seeing': 'seeingFwhm500',  # TODO which to use of this and seeingFwhmEff?
    'sunalt': 'sunAlt',
    # 'minsource': missing
}


def replace_instcat_from_db(
    rng,
    fname,
    conn,
    obsid,
    output_fname,
    allowed_include=None,
    sed=None,
    selector=None,
    galaxy_file=None,
    ccds=None,
):
    """
    Replace the instacat metadata and positions according to
    opsim data

    The ra/dec are generated uniformly in a disc centered at the
    boresight from the opsim data.

    One can limit the objects in the output file with the input selector
    function and/or by limiting included source files with a list
    of strings for matching with allowed_include

    Parameters
    ----------
    rng: np.random.default_rng
        The random number generator
    fname: str
        The input instcat filename, typically only used for stars
    gals_fname: str
        The input fits file for galaxies
    conn: connection to opsim db
        e.g. a sqlite3 connection
    obsid: int
        The observationId to use
    output_fname: str
        Name for new output instcat file
    allowed_include: list of strings
        Only includes with a filename that includes the string
        are kept.  For  ['star', 'gal'] we would keep filenames
        that had star or gal in them
    sed: str
        Force use if the input SED for all objects
        e.g starSED/phoSimMLT/lte034-4.5-1.0a+0.4.BT-Settl.spec.gz
    selector: function
        Evaluates True for objects to be kept, e.g.
            f = lambda d: d['magnorm'] > 17
    """

    import sqlite3

    conn.row_factory = sqlite3.Row

    query = f'select * from observations where observationId = {obsid}'
    data = conn.execute(query).fetchall()
    assert len(data) == 1

    opsim_data = data[0]
    # replace_instcat(
    replace_instcat_streamed(
        rng=rng,
        fname=fname,
        opsim_data=opsim_data,
        output_fname=output_fname,
        allowed_include=allowed_include,
        sed=sed,
        selector=selector,
        galaxy_file=galaxy_file,
        ccds=ccds,
    )


# def replace_instcat(
#     rng, fname, opsim_data, output_fname, allowed_include=None, sed=None,
#     selector=None
# ):
#     """
#     Replace the instacat metadata and positions according to the
#     input opsim data
#
#     The ra/dec are generated uniformly in a disc centered at the
#     boresight from the opsim data.
#
#     One can limit the objects in the output file with the input selector
#     function and/or by limiting included source files with a list
#     of strings for matching with allowed_include
#
#     Parameters
#     ----------
#     rng: np.random.default_rng
#         The random number generator
#     fname: str
#         The input instcat filename
#     opsim_data: mapping
#         E.g. a sqlite.Row read from an opsim database.
#     output_fname: str
#         Name for new output instcat file
#     allowed_include: list of strings
#         Only includes with a filename that includes the string
#         are kept.  For  ['star', 'gal'] we would keep filenames
#         that had star or gal in them
#     sed: str
#         Force use if the input SED for all objects
#         e.g starSED/phoSimMLT/lte034-4.5-1.0a+0.4.BT-Settl.spec.gz
#     selector: function
#         Evaluates True for objects to be kept, e.g.
#             f = lambda d: d['magnorm'] > 17
#     """
#
#     assert output_fname != fname
#
#     data, orig_meta = read_instcat(
#         fname, allowed_include=allowed_include,
#     )
#
#     if selector is not None:
#         data = [d for d in data if selector(d)]
#
#    meta = replace_instcat_meta(
#        rng=rng,
#        meta=orig_meta,
#        opsim_data=opsim_data,
#    )
#     replace_instcat_radec(
#         rng=rng,
#         ra=meta['rightascension'],
#         dec=meta['declination'],
#         data=data,
#     )
#
#     if sed is not None:
#         print('replacing SED with:', sed)
#         for tdata in data:
#             tdata['sed1'] = sed
#
#     write_instcat(output_fname, data, meta)


def replace_instcat_meta(rng, meta, opsim_data):
    """
    Replace the metadata from an instcat with entries from an opsim
    database row

    Note all are replaced, see NAME_MAP

    Parameters
    ----------
    rng: np.random.default_rng
        The random number generator
    meta: dict
        The metadata from in instcat
    opsim_data: mapping
        E.g. a sqlite.Row read from an opsim database.
    """
    new_meta = meta.copy()
    for key in NAME_MAP:
        assert key in meta

        opsim_val = opsim_data[NAME_MAP[key]]

        if key == 'filter':
            # opsim has filter name, but instcat wants number
            new_meta[key] = FILTERMAP[opsim_val]
        else:
            new_meta[key] = opsim_val

    new_meta['seed'] = rng.integers(0, 2**30)
    new_meta['seqnum'] = 0
    return new_meta


def replace_instcat_radec(rng, ra, dec, data):
    """
    generate new positions for objects centered at the ra, dec

    Parameters
    ----------
    rng: np.random.default_rng
        The random number generator
    ra: float
        The ra of the new pointing
    dec: float
        The dec of the new pointing
    data: list
        List of instcat entries, as read with read_instcat
    """
    from esutil.coords import randcap
    n = len(data)

    rra, rdec = randcap(
        nrand=n,
        ra=ra,
        dec=dec,
        rad=INSTCAT_RADIUS,
        rng=rng,
    )

    for i, d in enumerate(data):
        d['ra'] = rra[i]
        d['dec'] = rdec[i]


class RadecGenerator():
    """
    generate new positions for objects centered at the ra, dec

    Parameters
    ----------
    rng: np.random.default_rng
        The random number generator
    ra: float
        The ra of the new pointing
    dec: float
        The dec of the new pointing
    """
    def __init__(self, rng, ra, dec):
        self.rng = rng
        self.ra = ra
        self.dec = dec

    def __call__(self):
        from esutil.coords import randcap

        ra, dec = randcap(
            nrand=1,
            ra=self.ra,
            dec=self.dec,
            rad=INSTCAT_RADIUS,
            rng=self.rng,
        )
        return ra[0], dec[0]


class CCDRadecGenerator():
    """
    generate new positions for objects centered at the ra, dec

    Parameters
    ----------
    rng: np.random.default_rng
        The random number generator
    ra: float
        The ra of the new pointing
    dec: float
        The dec of the new pointing
    """
    def __init__(self, rng, wcs):
        self.rng = rng
        self.wcs = wcs

    def __call__(self, n=None):
        import galsim

        if n is None:
            is_scalar = True
            n = 1
        else:
            is_scalar = False

        x = self.rng.uniform(low=1, high=4096, size=n)
        y = self.rng.uniform(low=1, high=4096, size=n)

        ra, dec = self.wcs.xyToRadec(
            x=x,
            y=y,
            units=galsim.degrees,
        )

        if is_scalar:
            ra = ra[0]
            dec = dec[0]

        return ra, dec


def read_instcat(fname, allowed_include=None):
    """
    Read data and metadata from an instcat

    Parameters
    ----------
    fname: str
        The instcat file name
    allowed_include: list of strings, optional
        Only includes with a filename that includes the string
        are kept.  For  ['star', 'gal'] we would keep filenames
        that had star or gal in them
    """
    import os
    from esutil.ostools import DirStack

    ds = DirStack()
    dirname = os.path.dirname(fname)
    ds.push(dirname)

    meta = read_instcat_meta(fname)
    data = read_instcat_data_as_dicts(
        fname, allowed_include=allowed_include,
    )

    ds.pop()
    return data, meta


def read_instcat_data_as_dicts(fname, allowed_include=None):
    """
    Read object entries from an instcat

    Parameters
    ----------
    fname: str
        The instcat file name
    allowed_include: list of strings, optional
        Only includes with a filename that includes the string
        are kept.  For  ['star', 'gal'] we would keep filenames
        that had star or gal in them

    object id ra dec magnorm sed1 sed2 gamma1 gamma2 kappa rest
    """
    entries = []

    print('reading:', fname)
    opener, mode = _get_opener(fname)

    with opener(fname, mode) as fobj:
        for line in fobj:
            ls = line.split()

            if ls[0] == 'object':

                entry = {
                    'objid': int(ls[1]),
                    'ra': float(ls[2]),
                    'dec': float(ls[3]),
                    'magnorm': float(ls[4]),
                    'sed1': ls[5],
                    'sed2': float(ls[6]),
                    'gamma1': float(ls[7]),
                    'gamma2': float(ls[8]),
                    'kappa': float(ls[9]),
                    'rest': ' '.join(ls[10:]),
                }
                entries.append(entry)

            elif ls[0] == 'includeobj':
                fname = ls[1]
                if allowed_include is not None:
                    keep = False
                    for allowed in allowed_include:
                        if allowed in fname:
                            keep = True
                            break
                else:
                    keep = True

                if keep:
                    print('reading included:', fname)
                    entries += read_instcat_data_as_dicts(fname)

    return entries


def instcat_to_fits(fname, out_fname, allowed_include=None):
    """
    Read data and metadata from an instcat and write to fits file

    Parameters
    ----------
    fname: str
        The instcat file name
    out_fname: str
        The output fits file name
    allowed_include: list of strings
        Only includes with a filename that includes the string
        are kept.  For  ['star', 'gal'] we would keep filenames
        that had star or gal in them
    """
    import fitsio
    import os
    from esutil.ostools import DirStack

    if allowed_include is None:
        allowed_include = ['star', 'gal', 'knots']

    assert fname != out_fname

    ds = DirStack()
    dirname = os.path.dirname(fname)
    ds.push(dirname)

    meta = read_instcat_meta(fname)

    print('opening output:', out_fname)
    with fitsio.FITS(out_fname, 'rw', clobber=True) as fits:

        print('\nopening:', fname)
        with open(fname, 'r') as main_fobj:
            for mline in main_fobj:

                ls = mline.split()

                if ls[0] == 'includeobj':

                    include_fname = ls[1]
                    if _check_allowed_include(allowed_include, include_fname):

                        print('    reading from:', include_fname)
                        ns = include_fname.split('_')
                        ind = ns.index('cat')
                        extname = '_'.join(ns[:ind])
                        _copy_include_to_fits(
                            fits=fits, fname=include_fname, meta=meta,
                            extname=extname,
                        )

                        # dlist = _load_include_dlist(include_fname)
                        #
                        # print('    writing ext:', extname)
                        # _write_dlist_to_fits(
                        #      fits=fits, meta=meta,
                        #      dlist=dlist, extname=extname,
                        # )

    ds.pop()


# def _write_dlist_to_fits(fits, meta, dlist, extname):
#     from tqdm import trange
#     chunksize = 10000
#     nchunks = len(dlist) // chunksize
#     if len(dlist) % chunksize != 0:
#         nchunks += 1
#
#     dtype = _get_dtype(dlist)
#
#     for ichunk in trange(nchunks):
#         start = ichunk * chunksize
#         end = (ichunk + 1) * chunksize
#         sublist = dlist[start:end]
#
#         outdata = _dlist_to_np(dlist=sublist, dtype=dtype)
#         if ichunk == 0:
#             fits.write(outdata, header=meta, extname=extname)
#         else:
#             fits[-1].append(outdata)


def _check_allowed_include(allowed_include, string):
    if allowed_include is not None:
        keep = False
        for allowed in allowed_include:
            if allowed in string:
                keep = True
                break
    else:
        keep = True

    return keep


def _copy_include_to_fits(fits, fname, meta, extname):
    from tqdm import tqdm
    opener, mode = _get_opener(fname)
    dlist = []

    sed_len = 0
    rest_len = 0
    chunksize = 10_000

    first = True
    ntot = 0

    for ipass in [1, 2]:
        print(f'ipass {ipass}')
        if ipass == 2:
            dtype = _get_dtype(sed_len=sed_len, rest_len=rest_len)

        with opener(fname, mode) as fobj:

            num = 0
            dlist = []
            for line in tqdm(fobj):
                ntot += 1
                ls = line.split()
                entry = _read_instcat_object_line_as_dict(ls)

                if ipass == 1:
                    ntot += 1
                    sed_len = max(sed_len, len(entry['sed1']))
                    rest_len = max(rest_len, len(entry['rest']))
                else:
                    num += 1
                    dlist.append(entry)
                    if len(dlist) == chunksize or num == ntot:
                        data = _dlist_to_np(dlist=dlist, dtype=dtype)
                        if first:
                            fits.write(data, header=meta, extname=extname)
                            first = False
                        else:
                            fits[-1].append(data)
                        del dlist
                        dlist = []


# def _load_include_dlist(fname, nokeep):
#     from tqdm import tqdm
#     opener, mode = _get_opener(fname)
#     dlist = []
#     with opener(fname, mode) as fobj:
#         for line in tqdm(fobj):
#             ls = line.split()
#             entry = _read_instcat_object_line_as_dict(ls)
#             dlist.append(entry)
#
#     return dlist


def _dlist_to_np(dlist, dtype):
    import numpy as np

    out = np.zeros(len(dlist), dtype=dtype)
    for i, d in enumerate(dlist):
        for name in d.keys():
            out[name][i] = d[name]

    return out


def _get_dtype(sed_len, rest_len):
    sed_dt = f'U{sed_len}'
    rest_dt = f'U{rest_len}'

    dtype = [
        ('objid', 'i8'),
        ('ra', 'f8'),
        ('dec', 'f8'),
        ('magnorm', 'f4'),
        ('sed1', sed_dt),
        ('sed2', 'f4'),
        ('gamma1', 'f4'),
        ('gamma2', 'f4'),
        ('kappa', 'f4'),
        ('rest', rest_dt),
    ]
    return dtype


def replace_instcat_streamed(
    rng,
    fname,
    opsim_data,
    output_fname,
    allowed_include=None,
    sed=None,
    selector=None,
    galaxy_file=None,
    ccds=None,
):
    """
    Replace the instacat metadata and positions according to the input opsim
    data.  Normally only stars are read from the instcat using
    allowed_include=['star'], galaxies should be provided through the
    galaxy_file option and limited to the CCDs associated with the wcss input

    The ra/dec for stars are generated uniformly in a disc centered at the
    boresight from the opsim data, while galaxies get limited to CCDs

    One can limit the objects in the output file with the input selector
    function and/or by limiting included source files with a list
    of strings for matching with allowed_include

    Parameters
    ----------
    rng: np.random.default_rng
        The random number generator
    fname: str
        The input instcat filename
    opsim_data: mapping
        E.g. a sqlite.Row read from an opsim database.
    output_fname: str
        Name for new output instcat file
    allowed_include: list of strings
        Only includes with a filename that includes the string
        are kept.  For  ['star', 'gal'] we would keep filenames
        that had star or gal in them
    sed: str
        Force use if the input SED for all objects
        e.g starSED/phoSimMLT/lte034-4.5-1.0a+0.4.BT-Settl.spec.gz
    selector: function
        Evaluates True for objects to be kept, e.g.
            f = lambda d: d['magnorm'] > 17
    """
    import os
    from esutil.ostools import DirStack

    assert output_fname != fname

    orig_meta = read_instcat_meta(fname)
    meta = replace_instcat_meta(rng=rng, meta=orig_meta, opsim_data=opsim_data)

    # ra/dec gen in full circle, only used for stars
    radec_gen = RadecGenerator(
        rng=rng,
        ra=meta['rightascension'],
        dec=meta['declination'],
    )

    print('writing new instcat:', output_fname)

    with open(output_fname, 'w') as fout:
        _write_instcat_meta(fout=fout, meta=meta)

        ds = DirStack()
        dirname = os.path.dirname(fname)
        ds.push(dirname)

        # the include files are only the base name, easiest to just
        # chdir into the same directory as the main file
        _copy_objects(
            fout=fout, fname=fname, selector=selector,
            allowed_include=allowed_include, sed=sed,
            radec_gen=radec_gen,
        )

        if galaxy_file is not None:
            print('getting alaxies from:', galaxy_file)
            _copy_galaxies_ccds(
                meta=meta,
                fout=fout,
                rng=rng,
                fname=galaxy_file,
                selector=selector,
                sed=sed,
                ccds=ccds,
            )

        ds.pop()


def _copy_objects(
    fout, fname, selector, allowed_include, sed, radec_gen,
):
    from tqdm import tqdm

    opener, mode = _get_opener(fname)

    print('\nopening:', fname)
    with opener(fname, mode) as fobj:
        for line in tqdm(fobj):

            ls = line.split()

            if ls[0] == 'object':
                entry = _read_instcat_object_line_as_dict(ls)
                if selector(entry):
                    ra, dec = radec_gen()
                    entry['ra'] = ra
                    entry['dec'] = dec

                    if sed is not None:
                        entry['sed1'] = sed

                    _write_instcat_line(fout=fout, entry=entry)

            elif ls[0] == 'includeobj':
                include_fname = ls[1]
                if allowed_include is not None:
                    keep = False
                    for allowed in allowed_include:
                        if allowed in include_fname:
                            keep = True
                            break
                else:
                    keep = True

                if keep:
                    _copy_objects(
                        fout=fout, fname=include_fname, selector=selector,
                        allowed_include=allowed_include, sed=sed,
                        radec_gen=radec_gen,
                    )


def _copy_galaxies_ccds(
    meta,
    fout,
    fname,
    rng,
    selector,
    ccds,
    sed=None,
):
    import numpy as np
    import fitsio

    wcss = [
        instcat_meta_to_wcs(meta, ccd) for ccd in ccds
    ]
    radec_generators = [
        CCDRadecGenerator(rng=rng, wcs=wcs)
        for wcs in wcss
    ]
    nums = [
        rng.poisson(111341)
        for i in range(len(wcss))
    ]
    ntot = sum(nums)

    with fitsio.FITS(fname) as fits:
        # get all indices first
        nrows = fits['disk_gal'].get_nrows()
        all_indices = rng.randint(0, nrows, size=ntot)
        all_indices.sort()

        # now for each CCD
        start = 0
        for radec_gen, num in zip(radec_generators, nums):

            end = start + num
            indices = all_indices[start:end]

            data = fits['disk_gal'][indices]

            w, = np.where(selector(data))
            data = data[w]

            ra, dec = radec_gen(data.size)
            data['ra'] = ra
            data['dec'] = dec

            if sed is not None:
                data['sed1'] = sed

            _write_instcat_lines_from_array(fout=fout, data=data)

            start = end


def _write_instcat_meta(fout, meta):
    for key, value in meta.items():
        line = f'{key} {value}\n'
        fout.write(line)


def _write_instcat_line(fout, entry):
    line = ['object'] + [str(v) for k, v in entry.items()]
    line = ' '.join(line)
    fout.write(line)
    fout.write('\n')


def _write_instcat_lines_from_array(fout, data):
    names = data.dtype.names

    for d in data:
        line = ['object']
        for name in names:
            line += [str(d[name])]

        line = ' '.join(line)
        fout.write(line)
        fout.write('\n')


def _read_instcat_object_line_as_dict(ls):
    """
    Read object entries from an instcat

    Parameters
    ----------
    ls: sequence
        the split line from the instcat
    Returns
    -------
    entry: dict holding data
    """

    return {
        'objid': int(ls[1]),
        'ra': float(ls[2]),
        'dec': float(ls[3]),
        'magnorm': float(ls[4]),
        'sed1': ls[5],
        'sed2': float(ls[6]),
        'gamma1': float(ls[7]),
        'gamma2': float(ls[8]),
        'kappa': float(ls[9]),
        'rest': ' '.join(ls[10:]),
    }


def read_instcat_meta(fname):
    """
    get the metadata from the instance catalog header (metadata)

    convert to yaml and let it do data conversions

    Parameters
    ----------
    fname: str
        Path to the instcat
    """
    import yaml

    opener, mode = _get_opener(fname)

    with opener(fname, mode) as fobj:
        entries = []
        for line in fobj:
            ls = line.split()
            if ls[0] in ['object', 'includeobj']:
                break

            entry = f'{ls[0]}: {ls[1]}'
            entries.append(entry)

    s = '{' + ',\n'.join(entries) + '}'
    return yaml.safe_load(s)


def instcat_meta_to_wcs(meta, detnum):
    """
    Create an approximate WCS using the metadata

    Parameters
    ----------
    meta: dict
        Metadata read from instcat
    detnum: int
        Detector number, 1-189
    """
    import galsim
    from .instcat_tools import RFILTERMAP
    from astropy.time import Time
    from imsim.telescope_loader import load_telescope
    from imsim.camera import get_camera
    from imsim.batoid_wcs import BatoidWCSBuilder

    camera_name = 'LsstCam'
    filt = meta['filter']
    band = RFILTERMAP[filt]

    obstime = Time(meta['mjd'], format='mjd')
    boresight = galsim.CelestialCoord(
        meta['rightascension'] * galsim.degrees,
        meta['declination'] * galsim.degrees,
    )
    rot = meta['rottelpos'] * galsim.degrees
    telescope = load_telescope(f"LSST_{band}.yaml", rotTelPos=rot)
    factory = BatoidWCSBuilder().makeWCSFactory(
        boresight, obstime, telescope, bandpass=band, camera=camera_name,
    )

    camera = get_camera(camera_name)
    det = camera[detnum]
    return factory.getWCS(det)


def write_instcat(fname, data, meta):
    """
    Write an instcat

    Parameters
    ----------
    fname: str
        Path to output file
    data: list
        List of dict representing instcat objects
    meta: dict
        Metadata to be written in header
    """
    import esutil as eu

    eu.ostools.makedirs_fromfile(fname, allow_fail=True)

    print('writing:', fname)
    with open(fname, 'w') as fobj:
        for key, value in meta.items():
            line = f'{key} {value}\n'
            fobj.write(line)

        # this relies on dicts being ordered
        for d in data:
            line = ['object'] + [str(v) for k, v in d.items()]
            line = ' '.join(line)
            fobj.write(line)
            fobj.write('\n')


def _get_opener(fname):
    import gzip

    if '.gz' in fname:
        opener = gzip.open
        mode = 'rt'
    else:
        opener = open
        mode = 'r'

    return opener, mode
