name_map = {
    'rightascension': 'fieldra',
    'declination': 'fieldDec',
    'mjd': 'observationStartMJD',
    'altitude': 'altitude',
    'azimuth': 'azimuth',
    'filter': 'filter',
    'rotskypos': 'rotSkyPos',
    # 'dist2moon': missing,
    'moonalt': 'moonAlt',
    'moonphase': 'moonPhase',
    'moonra': 'moonRA',
    # 'nsnap': missing,
    # 'obshistid': missing
    'rottelpos': 'rotTelPos',
    # 'seed': to be set
    'seeing': 'seeingFwhm500',  # TODO which to use of this and seeingFwhmEff?
    'sunalt': 'sunAlt',
    # 'minsource': missing
}


}
def replace_instcat(rng, fname, opsim_data, output_fname, allowed_include=None):
    orig_data, orig_meta = read_instcat(fname, allowed_include=allowed_include)

    meta = replace_instcat_meta(rng=rng, meta=orig_meta, opsim_data=opsim_data)


def replace_instcat_meta(rng, meta, opsim_data):
    new_meta = meta.copy()
    for key in name_map:
        assert key in meta
        new_meta[key] = name_map[key]

    new_meta['seed'] = rng.integers(0, 2**30)
    new_meta['seqnum'] = 0
    return new_meta


def replace_instcat_data(rng, ra, dec, data):
    """
    generate new positions for objects centered at the ra, dec

    TODO get real density of objects, which isn't easy due to
    elliptical distribution of objects in original catalog
    """
    from esutil.coords import randcap
    n = len(data)

    rra, rdec = randcap(
        nrand=n,
        ra=ra,
        dec=dec,
        rad=4.0,
        rng=rng,
    )

def read_instcat(fname, allowed_include=None):
    meta = read_instcat_header(fname)
    data = read_instcat_data_as_dicts(fname, allowed_include=allowed_include)
    return data, meta


def read_instcat_data_as_dicts(fname, allowed_include=None):
    """
    object id ra dec magnorm sed1 sed2 gamma1 gamma2 kappa rest

    """
    entries = []

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


def read_instcat_header(fname):
    """
    get the metadata from the instance catalog header

    convert to yaml and let it do data conversions
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


def _get_opener(fname):
    import gzip

    if '.gz' in fname:
        opener = gzip.open
        mode = 'rt'
    else:
        opener = open
        mode = 'r'

    return opener, mode
