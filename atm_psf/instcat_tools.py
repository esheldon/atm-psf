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
