"""fextractor: pretrained feature and representation extraction."""

from .base import FeatureExtractor
from .registry import create_extractor, get_extractor_class, list_backends

__all__ = [
	"FeatureExtractor",
	"create_extractor",
	"get_extractor_class",
	"list_backends",
]

__version__ = "0.1.0"
