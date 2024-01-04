def read_instcat(fname):
    pass



def read_instcat_data_as_dicts(fname):
    """
    object id ra dec magnorm sed1 sed2 gamma1 gamma2 kappa rest

    """
    entries = []
    with open(fname) as fobj:
        for line in fobj:
            ls = line.split()

            if ls[0] != 'object':
                continue

            entry = {
                'objid': int(tokens[1]),
                'ra': float(tokens[2]),
                'dec': float(tokens[3]),
                'magnorm': float(tokens[4]),
                'sed1': tokens[5],
                'sed2': float(tokens[6]),
                'gamma1': float(tokens[7]),
                'gamma2': float(tokens[8]),
                'kappa': float(tokens[9]),
                'rest': ' '.join(tokens[10:]),
            }
            entries.append(entry)


def read_instcat_header(fname):
    """
    get the metadata from the instance catalog header

    convert to yaml and let it do data conversions
    """
    import yaml

    with open(fname) as fobj:
        entries = []
        for line in fobj:
            ls = line.split()
            if ls[0] == 'object':
                break

            entry = f'{ls[0]}: {ls[1]}'
            entries.append(entry)

    s = '{' + '\n'.join(entries) + '}'
    return yaml.safe_load(s)
