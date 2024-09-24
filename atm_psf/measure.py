import logging
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

LOG = logging.getLogger('atm_psf_measure')


class DetectMeasurer(object):
    """
    Class to run detection and measurements
    """
    def __init__(self, exposure, rng, thresh=5):
        self._exposure = exposure
        self._thresh = thresh
        self._rng = rng
        try:
            self._seed = rng.randint(0, 2**30)
        except AttributeError:
            self._seed = rng.integers(0, 2**30)

    def detect(self):
        import lsst.meas.extensions.shapeHSM  # noqa
        import lsst.afw.table as afw_table
        from lsst.meas.base import (
            SingleFrameMeasurementConfig,
            SingleFrameMeasurementTask,
        )
        from lsst.meas.algorithms import (
            SourceDetectionTask, SourceDetectionConfig,
        )
        from lsst.meas.deblender import SourceDeblendTask, SourceDeblendConfig
        from metadetect.lsst.util import get_stats_mask

        self._schema = afw_table.SourceTable.makeMinimalSchema()

        # Setup algorithms to run
        meas_config = SingleFrameMeasurementConfig()
        meas_config.plugins.names = [
            'base_PixelFlags',
            'base_SdssCentroid',
            'base_PsfFlux',
            'base_SkyCoord',
            # 'base_SdssShape',
            # these will get used by star selector
            'ext_shapeHSM_HsmSourceMoments',
            'ext_shapeHSM_HsmPsfMoments',
        ]

        # set these slots to none because we aren't running these algorithms
        meas_config.slots.apFlux = None
        meas_config.slots.gaussianFlux = None
        meas_config.slots.calibFlux = None
        meas_config.slots.modelFlux = None

        # goes with SdssShape above
        meas_config.slots.shape = "ext_shapeHSM_HsmSourceMoments"

        # fix odd issue where it things things are near the edge
        meas_config.plugins['base_SdssCentroid'].binmax = 1

        # sub-pixel offsets in the psf rendering
        # meas_config.plugins['ext_shapeHSM_HsmPsfMoments'].useSourceCentroidOffset = True  # noqa

        self._meas_task = SingleFrameMeasurementTask(
            config=meas_config,
            schema=self._schema,
        )
        afw_table.CoordKey.addErrorFields(self._schema)

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

    def measure_ngmix(self):
        from .measure_ngmix import ngmix_measure
        self.ngmix_result = ngmix_measure(
            exp=self._exposure,
            sources=self.sources,
            stamp_size=32,
            rng=self._rng,
        )

    def measure(self):
        """
        This can be rerun as needed
        """
        import numpy as np
        from metadetect.lsst.util import ContextNoiseReplacer

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
