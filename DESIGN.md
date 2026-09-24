# Design notes

## Scope

`fextractor` is an inference-only package. It does not train SimCLR, DINO or
SigLIP models. Training remains in the packages/projects that own those models.

## Preprocessing split

The original extraction macros each implemented FITS/image loading, NaN/blank
handling, zscale, normalization and model adaptation independently. In
`fextractor` these are split into:

- **domain preprocessing** (`fextractor.preprocessing`): scientific image
  decoding, non-finite/blank handling, optional sigma clipping, optional zscale,
  optional min-max normalization;
- **model preprocessing** (extractor backend): exact resize, channel conversion,
  tensor layout and pretrained processor/model normalization.

This is deliberate: DINO's ImageNet normalization and SigLIP's Hugging Face
processor are properties of those pretrained models, while zscale is a domain
choice.

## Datalist compatibility

The runner preserves the existing convention used by the supplied macros:

- input path: `item["filepaths"][0]` by default;
- output vector: `item["feats"]` by default.

Both keys/indexes are configurable.

## TensorFlow / SimCLR

The TensorFlow backend assumes the supplied model is already the representation
model/encoder. For the radio SimCLR use case, export/load the encoder rather than
the complete contrastive-training graph. This keeps `fextractor` independent
from the SimCLR training implementation.

## Future modalities

A time-series extractor only needs to implement `BaseFeatureExtractor`. The
current image preprocessing package is not part of the abstract interface, so
adding `fextractor.timeseries` does not require changing the runner or REST API.
