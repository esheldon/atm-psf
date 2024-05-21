def get_default_sim_config():
    import mimsim

    return {
        'sky_gradient': True,
        'vignetting': True,
        'apply_sky_pixel_area': True,
        'dcr': True,
        'tree_rings': True,
        'cosmic_ray_rate': mimsim.defaults.DEFAULT_COSMIC_RAY_RATE,
        'magmin': -1000,
        'psf': {
            'type': 'psfws',
            'options': {},
        },
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

    return config
