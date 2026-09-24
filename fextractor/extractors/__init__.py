"""Built-in feature extractor backends."""

from .tensorflow import TensorFlowFeatureExtractor
from .dinov2 import DINOv2FeatureExtractor
from .dinov3 import DINOv3FeatureExtractor
from .dinov2_legacy import DINOv2LegacyFeatureExtractor
from .siglip import SigLIPFeatureExtractor
from .siglip2 import SigLIP2FeatureExtractor

__all__ = [
	"TensorFlowFeatureExtractor",
	"DINOv2FeatureExtractor",
	"DINOv3FeatureExtractor",
	"DINOv2LegacyFeatureExtractor",
	"SigLIPFeatureExtractor",
	"SigLIP2FeatureExtractor",
]
