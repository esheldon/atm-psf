import logging
import warnings

import lsst.afw.table as afw_table
from lsst.meas.algorithms import SourceDetectionTask, SourceDetectionConfig
from lsst.meas.deblender import SourceDeblendTask, SourceDeblendConfig
from lsst.meas.base import (
    SingleFrameMeasurementConfig,
    SingleFrameMeasurementTask,
)

from metadetect.lsst.util import ContextNoiseReplacer, get_stats_mask

warnings.filterwarnings('ignore', category=FutureWarning)

LOG = logging.getLogger('atm_psf_measure')


def detect_and_measure_old(
    exposure,
    rng,
    thresh=5,
):
    """
    run detection and deblending of peaks, as well as basic measurments such as
    centroid.  The SDSS deblender is run in order to split footprints.

    We must combine detection and deblending in the same function because the
    schema gets modified in place, which means we must construct the deblend
    task at the same time as the detect task

    Parameters
    ----------
    exposure: lsst.afw.image.ExposureF
        The exposure to process
    rng: np.random.RandomState
        Random number generator for noise replacer
    thresh: float, optional
        The detection threshold in units of the sky noise

    Returns
    -------
    sources
        The sources
    """
    import lsst.meas.extensions.shapeHSM  # noqa

    schema = afw_table.SourceTable.makeMinimalSchema()

    # Setup algorithms to run
    meas_config = SingleFrameMeasurementConfig()
    meas_config.plugins.names = [
        'base_PixelFlags',
        'base_SdssCentroid',
        'base_PsfFlux',
        'base_SkyCoord',
        'base_SdssShape',
        'ext_shapeHSM_HsmSourceMoments',
        'ext_shapeHSM_HsmPsfMoments',
    ]

    # set these slots to none because we aren't running these algorithms
    meas_config.slots.apFlux = None
    meas_config.slots.gaussianFlux = None
    meas_config.slots.calibFlux = None
    meas_config.slots.modelFlux = None

    # goes with SdssShape above
    # meas_config.slots.shape = None

    # fix odd issue where it things things are near the edge
    meas_config.plugins['base_SdssCentroid'].binmax = 1

    meas_task = SingleFrameMeasurementTask(
        config=meas_config,
        schema=schema,
    )

    detection_config = SourceDetectionConfig()
    detection_config.reEstimateBackground = False
    # detection_config.reEstimateBackground = True
    # variance here actually means relative to the sqrt(variance)
    # from the variance plane.
    # TODO this would include poisson
    # TODO detection doesn't work right when we tell it to trust
    # the variance
    # detection_config.thresholdType = 'variance'
    detection_config.thresholdValue = thresh

    # these will be ignored when finding the image standard deviation
    detection_config.statsMask = get_stats_mask(exposure)

    detection_task = SourceDetectionTask(config=detection_config)

    # these tasks must use the same schema and all be constructed before any
    # other tasks using the same schema are run because schema is modified in
    # place by tasks, and the constructor does a check that fails if we do this
    # afterward

    deblend_task = SourceDeblendTask(
        config=SourceDeblendConfig(),
        schema=schema,
    )

    table = afw_table.SourceTable.make(schema)

    result = detection_task.run(table, exposure)

    if result is not None:
        sources = result.sources
        deblend_task.run(exposure, sources)

        with ContextNoiseReplacer(exposure, sources, rng) as replacer:

            for source in sources:

                if source.get('deblend_nChild') != 0:
                    continue

                source_id = source.getId()

                with replacer.sourceInserted(source_id):
                    meas_task.callMeasure(source, exposure)

        # sources['base_SdssShape_xx'] = sources['ext_shapeHSM_HsmSourceMoments_xx']
        # sources['base_SdssShape_xy'] = sources['ext_shapeHSM_HsmSourceMoments_xy']
        # sources['base_SdssShape_yy'] = sources['ext_shapeHSM_HsmSourceMoments_yy']
    else:
        sources = []

    return sources


class DetectMeasurer(object):
    def __init__(self, exposure, rng, thresh=5):
        self._exposure = exposure
        self._thresh = thresh
        self._seed = rng.randint(0, 2**20)

    def detect(self):
        import lsst.meas.extensions.shapeHSM  # noqa

        self._schema = afw_table.SourceTable.makeMinimalSchema()

        # Setup algorithms to run
        meas_config = SingleFrameMeasurementConfig()
        meas_config.plugins.names = [
            'base_PixelFlags',
            'base_SdssCentroid',
            'base_PsfFlux',
            'base_SkyCoord',
            'base_SdssShape',
            'ext_shapeHSM_HsmSourceMoments',
            'ext_shapeHSM_HsmPsfMoments',
        ]

        # set these slots to none because we aren't running these algorithms
        meas_config.slots.apFlux = None
        meas_config.slots.gaussianFlux = None
        meas_config.slots.calibFlux = None
        meas_config.slots.modelFlux = None

        # goes with SdssShape above
        # meas_config.slots.shape = None

        # fix odd issue where it things things are near the edge
        meas_config.plugins['base_SdssCentroid'].binmax = 1

        # sub-pixel offsets in the psf rendering
        meas_config.plugins['ext_shapeHSM_HsmPsfMoments'].useSourceCentroidOffset = True  # noqa

        self._meas_task = SingleFrameMeasurementTask(
            config=meas_config,
            schema=self._schema,
        )

        detection_config = SourceDetectionConfig()
        detection_config.reEstimateBackground = False
        # variance here actually means relative to the sqrt(variance)
        # from the variance plane.
        # TODO this would include poisson
        # TODO detection doesn't work right when we tell it to trust
        # the variance
        # detection_config.thresholdType = 'variance'
        detection_config.thresholdValue = self._thresh

        # these will be ignored when finding the image standard deviation
        detection_config.statsMask = get_stats_mask(self._exposure)

        self._detection_task = SourceDetectionTask(config=detection_config)

        # these tasks must use the same schema and all be constructed before
        # any other tasks, and must use the same schema because the schema is
        # modified in place by tasks, and the constructor does a check that
        # fails if we do this afterward

        self._deblend_task = SourceDeblendTask(
            config=SourceDeblendConfig(),
            schema=self._schema,
        )

        self._table = afw_table.SourceTable.make(self._schema)

        result = self._detection_task.run(self._table, self._exposure)

        if result is not None:
            sources = result.sources
            self._deblend_task.run(self._exposure, sources)
        else:
            sources = []

        self.sources = sources

    def measure(self):
        """
        This can be rerun as needed
        """
        import numpy as np
        # we want this to be repeatable
        rng = np.random.RandomState(self._seed)
        with ContextNoiseReplacer(
            self._exposure, self.sources, rng,
        ) as replacer:

            for source in self.sources:

                if source.get('deblend_nChild') != 0:
                    continue

                source_id = source.getId()

                with replacer.sourceInserted(source_id):
                    self._meas_task.callMeasure(source, self._exposure)
