# fextractor

`fextractor` is a lightweight Python package for extracting feature vectors / learned representations from pretrained machine-learning models.

The first release targets scientific images and supports:

- generic TensorFlow/Keras encoder models;
- DINOv2 models from the official `facebookresearch/dinov2` torch hub repository;
- SigLIP models through Hugging Face Transformers;
- FITS, PNG, and JPEG inputs;
- the CAESAR/sclassifier-style JSON datalist format (`filepaths` in, `feats` out).

The package is intentionally inference-oriented. Training code, including SimCLR training code, does not belong here. An exported SimCLR encoder is simply handled by the generic TensorFlow backend.

## Installation

Core package:

```bash
pip install -e .
```

Install only the backend(s) you need:

```bash
pip install -e '.[tensorflow]'
pip install -e '.[dino]'
pip install -e '.[siglip]'
pip install -e '.[all]'
```

## TensorFlow / radio SimCLR example

The historical macro invocation:

```bash
python extract_tfmodel_representation.py \
  --inputfile=metadata.json \
  --datalist_key=data \
  --model=encoder-resnet18_simclr_hulk256-smgps_ch1_100epochs.h5 \
  --model_weights=encoder_weights-resnet18_simclr_hulk256-smgps_ch1_100epochs.h5 \
  --imgsize=224 \
  --zscale \
  --zscale_contrast=0.25 \
  --save_to_json \
  --outfile=featdata.json
```

maps to:

```bash
fextractor \
  --backend tensorflow \
  --inputfile metadata.json \
  --datalist-key data \
  --model encoder-resnet18_simclr_hulk256-smgps_ch1_100epochs.h5 \
  --model-weights encoder_weights-resnet18_simclr_hulk256-smgps_ch1_100epochs.h5 \
  --profile radio_simclr \
  --outfile featdata.json
```

The `radio_simclr` profile records the tested preprocessing setup:

- image size: 224;
- one input channel;
- zscale enabled;
- zscale contrast: 0.25;
- min-max normalization to `[0, 1]`.

Python API:

```python
from fextractor.extractors import TensorFlowFeatureExtractor
from fextractor.preprocessing import get_profile

profile = get_profile("radio_simclr")
extractor = TensorFlowFeatureExtractor(
	model_path="encoder-resnet18_simclr_hulk256-smgps_ch1_100epochs.h5",
	weights_path="encoder_weights-resnet18_simclr_hulk256-smgps_ch1_100epochs.h5",
	imgsize=profile.imgsize,
	in_chans=profile.in_chans,
	preprocessing=profile.preprocessing,
)

features = extractor.extract("source.fits")
```

## DINOv2

```bash
fextractor \
  --backend dinov2 \
  --inputfile metadata.json \
  --model dinov2_vits14 \
  --zscale \
  --outfile featdata_dino.json
```

## SigLIP

```bash
fextractor \
  --backend siglip \
  --inputfile metadata.json \
  --model google/siglip-so400m-patch14-384 \
  --outfile featdata_siglip.json
```

## Single-image extraction

```bash
fextractor \
  --backend tensorflow \
  --image source.fits \
  --model encoder.h5 \
  --model-weights encoder_weights.h5 \
  --profile radio_simclr \
  --outfile source_features.json
```

## Preprocessing design

Preprocessing is split into two conceptual layers.

**Domain preprocessing**, shared by backends:

- FITS/PNG/JPEG decoding;
- non-finite pixel handling;
- optional blank/zero handling;
- optional sigma clipping;
- optional zscale;
- min-max normalization.

**Model preprocessing**, owned by each backend:

- input resize and channel formatting;
- DINOv2 ImageNet normalization;
- Hugging Face SigLIP image processor;
- TensorFlow input tensor shape.

This avoids the inconsistent preprocessing that was duplicated across the original standalone macros.

## Datalist format

Input:

```json
{
  "data": [
    {
      "sname": "source-1",
      "id": 0,
      "label": "UNKNOWN",
      "filepaths": ["/path/to/source.fits"]
    }
  ]
}
```

Output entries receive a `feats` array. The output document also contains an `fextractor` provenance block describing the backend and preprocessing configuration.

## Future modalities

The base `FeatureExtractor` interface is modality-neutral. Future time-series extractors can therefore be added without changing the datalist runner or REST-facing API.

## Development

Run static compilation:

```bash
python -m compileall -q fextractor tests
```

Run tests after installing test dependencies:

```bash
pip install -e '.[test]'
pytest
```

All Python source files in this repository use tab indentation.
