def setup_logging(loglevel):
    """
    Set up python logging at the requested level

    Parameters
    ----------
    loglevel: str
        e.g. 'info' or 'debug', case insensitive
    """
    import sys
    import logging

    logging.basicConfig(
        stream=sys.stdout,
        level=getattr(logging, loglevel.upper()),
    )
