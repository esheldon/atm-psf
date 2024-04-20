# radius for disc of random points
INSTCAT_RADIUS = 2.1
FILTERMAP = {
    'u': 0,
    'g': 1,
    'r': 2,
    'i': 3,
    'z': 4,
    'y': 5,
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
    selector=None
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
        The input instcat filename
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
    )


def replace_instcat(
    rng, fname, opsim_data, output_fname, allowed_include=None, sed=None,
    selector=None
):
    """
    Replace the instacat metadata and positions according to the
    input opsim data

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

    assert output_fname != fname

    data, orig_meta = read_instcat(
        fname, allowed_include=allowed_include,
    )

    if selector is not None:
        data = [d for d in data if selector(d)]

    meta = replace_instcat_meta(rng=rng, meta=orig_meta, opsim_data=opsim_data)
    replace_instcat_radec(
        rng=rng,
        ra=meta['rightascension'],
        dec=meta['declination'],
        data=data,
    )

    if sed is not None:
        print('replacing SED with:', sed)
        for tdata in data:
            tdata['sed1'] = sed

    write_instcat(output_fname, data, meta)


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


def replace_instcat_streamed(
    rng,
    fname,
    opsim_data,
    output_fname,
    allowed_include=None,
    sed=None,
    selector=None
):
    """
    Replace the instacat metadata and positions according to the
    input opsim data

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

        ds.pop()


def _copy_objects(
    fout, fname, selector, allowed_include, sed, radec_gen,
):

    opener, mode = _get_opener(fname)

    print('opening:', fname)
    with opener(fname, mode) as fobj:
        for line in fobj:

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


def _write_instcat_meta(fout, meta):
    for key, value in meta.items():
        line = f'{key} {value}\n'
        fout.write(line)


def _write_instcat_line(fout, entry):
    line = ['object'] + [str(v) for k, v in entry.items()]
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
