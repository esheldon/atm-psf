from functools import lru_cache

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
    'moondec': 'moonDec',
    # 'nsnap': missing,
    'obshistid': 'observationId',
    'rottelpos': 'rotTelPos',
    # 'seed': to be set
    'seeing': 'seeingFwhm500',  # TODO which to use of this and seeingFwhmEff?
    'sunalt': 'sunAlt',
    'vistime': 'visitExposureTime',
    'nsnap': 'numExposures',
    # 'minsource': missing
}


def replace_instcat_meta(rng, opsim_data, meta=None):
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
    if meta is None:
        new_meta = {}
    else:
        new_meta = meta.copy()

    for key in NAME_MAP:
        opsim_val = opsim_data[NAME_MAP[key]]

        if key == 'filter':
            # opsim has filter name, but instcat wants number
            new_meta[key] = FILTERMAP[opsim_val]
        else:
            new_meta[key] = opsim_val

    new_meta['seed'] = rng.integers(0, 2**30)
    new_meta['nsnap'] = 1
    new_meta['seqnum'] = 0
    return new_meta


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
        self.cache_size = 1_000_000
        self._cache_radec()

    def _cache_radec(self):
        from esutil.coords import randcap
        self.rra, self.rdec = randcap(
            nrand=self.cache_size,
            ra=self.ra,
            dec=self.dec,
            rad=INSTCAT_RADIUS,
            rng=self.rng,
        )
        self.used = 0

    def __call__(self):

        if self.used >= self.cache_size:
            self._cache_radec()

        index = self.used
        ra, dec = self.rra[index], self.rdec[index]
        self.used += 1
        return ra, dec


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

        ra, dec = self.wcs.xyToradec(
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
    from esutil.ostools import DirStack

    ds = DirStack()
    dirname, bname = path_split(fname)
    ds.push(dirname)

    meta = read_instcat_meta(bname)
    data = read_instcat_data_as_dicts(
        bname,
        allowed_include=allowed_include,
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
    from esutil.ostools import DirStack

    if allowed_include is None:
        allowed_include = ['star', 'gal', 'knots']

    assert fname != out_fname

    ds = DirStack()
    dirname, bname = path_split(fname)
    ds.push(dirname)

    meta = read_instcat_meta(bname)

    print('opening output:', out_fname)
    with fitsio.FITS(out_fname, 'rw', clobber=True) as fits:

        print('\nopening:', fname)
        with open(bname, 'r') as main_fobj:
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
    from esutil.pbar import pbar
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
            for line in pbar(fobj):
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
    dup=1,
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
    galaxy_file: str
        Path to the file for galaxies
    ccds: list
        List off CCDS, only used when galaxy_file is sent, to limit
        random ra/dec to the specified ccds
    dup: int, optional
        Number of times to duplicate, with random ra/dec
    """
    import os
    from esutil.ostools import DirStack, makedirs_fromfile

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
    makedirs_fromfile(output_fname, allow_fail=True)

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
            dup=dup,
        )

        if galaxy_file is not None:
            print('getting galaxies from:', galaxy_file)
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


def make_instcat_by_obsid_and_objfile(
    rng,
    object_file,
    opsim_data,
    output_fname,
    progress=False,
    selector=None,
):
    """
    Make a new instcat using the input opsim data and the object file.
    Objects within INSTCAT_RADIUS of the ra/dec in the opsim data will
    be written to the output

    TODO allow limiting to CCDs

    Parameters
    ----------
    rng: np.random.default_rng
        The random number generator
    object_file: str
        The input file with objects
    opsim_data: mapping
        E.g. a sqlite.Row read from an opsim database.
    output_fname: str
        Name for new output instcat file
    selector: function
        Evaluates True for objects to be kept, e.g.
            f = lambda d: d['magnorm'] > 17
    """
    import esutil as eu
    import numpy as np

    assert output_fname != object_file

    print('writing new instcat:', output_fname)
    eu.ostools.makedirs_fromfile(output_fname, allow_fail=True)

    with open(output_fname, 'w') as fout:
        meta = replace_instcat_meta(rng=rng, opsim_data=opsim_data)
        _write_instcat_meta(fout=fout, meta=meta)

        obj_data = _read_data(object_file)
        nobj = obj_data.size

        # ra = opsim_data['rightascension']
        # dec = opsim_data['declination']
        ra = meta['rightascension']
        dec = meta['declination']

        print(f'matching within {INSTCAT_RADIUS:.3g} degrees')
        dist = eu.coords.sphdist(
            ra1=ra, dec1=dec,
            ra2=obj_data['ra'], dec2=obj_data['dec'],
        )

        w, = np.where(dist < INSTCAT_RADIUS)
        print(f'kept {w.size} / {nobj} stars')
        if w.size == 0:
            raise RuntimeError('no matches found')

        obj_data = obj_data[w]

        if selector is not None:
            w, = np.where(selector(obj_data))
            print(f'kept {w.size} / {nobj} from selector')
            if w.size == 0:
                raise RuntimeError('no matches found')

            obj_data = obj_data[w]

        _write_instcat_lines_from_array(
            fout=fout, data=obj_data, progress=progress,
        )


def _copy_objects(
    fout, fname, selector, allowed_include, sed, radec_gen,
    dup=1,
):
    from esutil.pbar import pbar

    opener, mode = _get_opener(fname)

    print('\nopening:', fname)

    if dup > 1:
        print(f'duplicating {dup} times')

    with opener(fname, mode) as fobj:
        for line in pbar(fobj):

            ls = line.split()

            if ls[0] == 'object':
                entry = _read_instcat_object_line_as_dict(ls)
                if selector(entry):
                    for i in range(dup):
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
                        dup=dup,
                    )


def _copy_galaxies_ccds(
    meta,
    fout,
    fname,
    rng,
    selector,
    ccds,
    progress=False,
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
    nccd = len(ccds)

    with fitsio.FITS(fname) as fits:
        # get all indices first
        nrows = fits['disk_gal'].get_nrows()
        all_indices = rng.integers(0, nrows, size=ntot)
        all_indices.sort()

        # now for each CCD
        start = 0
        for i, ccd, radec_gen, num in zip(
            range(nccd), ccds, radec_generators, nums,
        ):
            print(f'ccd: {ccd} {i+1}/{nccd}')

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

            _write_instcat_lines_from_array(
                fout=fout, data=data, progress=progress,
            )

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


def _write_instcat_lines_from_array(fout, data, progress=False):
    from esutil.pbar import pbar

    if progress:
        miter = pbar(data)
    else:
        miter = data

    names = data.dtype.names

    for d in miter:
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


def _get_opener(fname):
    import gzip

    if '.gz' in fname:
        opener = gzip.open
        mode = 'rt'
    else:
        opener = open
        mode = 'r'

    return opener, mode


def path_split(fname):
    import os

    dirname, basename = os.path.split(fname)
    if dirname == '':
        dirname = './'
    return dirname, fname


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


# def replace_instcat_radec(rng, ra, dec, data):
#     """
#     generate new positions for objects centered at the ra, dec
#
#     Parameters
#     ----------
#     rng: np.random.default_rng
#         The random number generator
#     ra: float
#         The ra of the new pointing
#     dec: float
#         The dec of the new pointing
#     data: list
#         List of instcat entries, as read with read_instcat
#     """
#     from esutil.coords import randcap
#     n = len(data)
#
#     rra, rdec = randcap(
#         nrand=n,
#         ra=ra,
#         dec=dec,
#         rad=INSTCAT_RADIUS,
#         rng=rng,
#     )
#
#     for i, d in enumerate(data):
#         d['ra'] = rra[i]
#         d['dec'] = rdec[i]
#

# def write_instcat(fname, data, meta):
#     """
#     Write an instcat
#
#     Parameters
#     ----------
#     fname: str
#         Path to output file
#     data: list
#         List of dict representing instcat objects
#     meta: dict
#         Metadata to be written in header
#     """
#     import esutil as eu
#
#     eu.ostools.makedirs_fromfile(fname, allow_fail=True)
#
#     print('writing:', fname)
#     with open(fname, 'w') as fobj:
#         for key, value in meta.items():
#             line = f'{key} {value}\n'
#             fobj.write(line)
#
#         # this relies on dicts being ordered
#         for d in data:
#             line = ['object'] + [str(v) for k, v in d.items()]
#             line = ' '.join(line)
#             fobj.write(line)
#             fobj.write('\n')


@lru_cache
def _read_data(fname):
    import fitsio
    print('reading:', fname)
    return fitsio.read(fname)
