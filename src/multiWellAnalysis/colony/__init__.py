# src/multiWellAnalysis/colony/__init__.py

from .segmentation import segmentColonies

from .colonyFeats import (
    extractColonyGeometry,
    addColonySpatialFeatures,
    addColonyNeighborFeatures,
    addColonyGraphFeatures,
    addColonyIntensityMassFeatures,
)

from .legacy.wellAggregation import aggregateWellFeatures

from .legacy.tracking import (
    trackColoniesToFrames,
    findSeedFrameFromBiomass,
    findDispersalFrameFromBiomass,
    findBorderTouchingLabels,
)

__all__ = [
    'segmentColonies',

    # colony-level features
    'extractColonyGeometry',
    'addColonySpatialFeatures',
    'addColonyNeighborFeatures',
    'addColonyGraphFeatures',
    'addColonyIntensityMassFeatures',

    # aggregation
    'aggregateWellFeatures',

    # tracking
    'trackColoniesToFrames',
    'findSeedFrameFromBiomass',
    'findDispersalFrameFromBiomass',
    'findBorderTouchingLabels',
]
