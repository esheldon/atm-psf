def get_default_sim_config():
    import montauk

    return {
        'sky_gradient': True,
        'vignetting': True,
        'apply_pixel_areas': True,
        'dcr': True,
        'tree_rings': True,
        'cosmic_ray_rate': montauk.defaults.DEFAULT_COSMIC_RAY_RATE,
        'magmin': -1000,
        'psf': {
            'type': 'psfws',
            'options': {},
        },
        'skip_bright': False,
    }


def load_sim_config(fname=None):
    """
    Load config.  Elements in the input will override the defaults. For
    defaults see get_default_config()

    Parameters
    ----------
    fname: str, optional
        Optinal path to a yaml config file

    Returns
    -------
    config: dict
        Config dict
    """
    from .io import load_yaml

    if fname is not None:
        data = load_yaml(fname)
    else:
        data = {}

    config = get_default_sim_config()
    config.update(data)

    if 'options' not in config['psf']:
        config['psf']['options'] = {}

    if (config['psf']['type'] == 'imsim-atmpsf'
            and 'nproc' not in config['psf']['options']):
        config['psf']['options']['nproc'] = 1

    return config
