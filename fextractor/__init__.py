"""fextractor: pretrained feature and representation extraction."""

from .base import FeatureExtractor
from .config import ExtractorConfig
from .factory import create_extractor
from .registry import BackendSpec, get_backend_spec, get_extractor_class, list_backends

__all__ = [
	"BackendSpec",
	"ExtractorConfig",
	"FeatureExtractor",
	"create_extractor",
	"get_backend_spec",
	"get_extractor_class",
	"list_backends",
]

__version__ = "0.1.0"
