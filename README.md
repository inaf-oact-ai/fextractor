<p align="left">
  <img src="share/fextractor_logo.png" alt="fextractor logo" width="320"/>
</p>

# fextractor

`fextractor` is an inference-oriented Python framework for extracting fixed-dimensional feature vectors and learned representations from pretrained models applied to scientific data.

The package provides a common interface for multiple data modalities and model families while keeping domain preprocessing, model-specific preprocessing, extraction, provenance, and CAESAR-style datalist handling separate.

Current supported modalities are:

- **images** — FITS and common raster formats;
- **time series** — univariate or multivariate regularly or irregularly sampled series.

The architecture is intentionally extensible: new representation models can be added as extractor backends without changing the high-level runner or output format.

## Features

### Image representation extraction

Registered image backends currently include:

| Backend | Model family | Default model / purpose |
| --- | --- | --- |
| `tensorflow` | TensorFlow / Keras | User-supplied encoder model |
| `dinov2` | DINOv2 | `facebook/dinov2-small` |
| `dinov3` | DINOv3 | `facebook/dinov3-vits16-pretrain-lvd1689m` |
| `dinov2_legacy` | DINOv2 Torch Hub | `dinov2_vits14` |
| `siglip` | SigLIP | `google/siglip-so400m-patch14-384` |
| `siglip2` | SigLIP2 | `google/siglip2-so400m-patch14-384` |

Aliases:

- `tf` → `tensorflow`
- `dino` → `dinov2`

Supported scientific-image preprocessing includes:

- FITS / PNG / JPEG decoding;
- non-finite pixel handling;
- optional zero-to-minimum replacement;
- optional sigma clipping;
- optional astronomical zscale;
- min-max normalization;
- backend-owned resize, channel conversion and model-native normalization.

### Time-series representation extraction

Registered time-series representation backends currently include:

| Backend | Model | Default model |
| --- | --- | --- |
| `chronos2` | Amazon Chronos-2 | `amazon/chronos-2` |
| `moirai2` | Salesforce Moirai-2 | `Salesforce/moirai-2.0-R-small` |

Aliases:

- `chronos` → `chronos2`
- `moirai` → `moirai2`

Both backends expose contextual model representations and reduce token-level representations to one feature vector through a configurable aggregation strategy.

Supported aggregation modes are:

```text
mean
std
max
mean_std
mean_max
mean_std_max
last
reg
flatten
```

`mean_std` is the default for the current Chronos-2 and Moirai-2 extractors.

The time-series architecture is backend-neutral, so additional time-series embedding models can be registered without changing the input/output runner.

## Installation

`fextractor` requires Python 3.10 or newer.

Install the core package:

```bash
pip install -e .
```

Install only the optional dependencies required by a backend:

```bash
pip install -e '.[tensorflow]'
pip install -e '.[tensorflow-gpu]'

pip install -e '.[dino]'
pip install -e '.[dino-legacy]'
pip install -e '.[siglip]'
pip install -e '.[siglip2]'

pip install -e '.[chronos]'
pip install -e '.[moirai]'
```

For time-series diagnostic plots:

```bash
pip install -e '.[plot]'
```

For Gaussian-process time-series regularization:

```bash
pip install -e '.[gp]'
```

Combined dependency groups are also available:

```bash
pip install -e '.[torch-all]'
pip install -e '.[timeseries-all]'
pip install -e '.[all]'
```

## Command-line interface

The installed command is:

```bash
fextractor
```

At minimum, specify a backend and an input:

```bash
fextractor \
  --backend <backend> \
  --inputfile <input> \
  --outfile fextractor_results.json
```

The input type is inferred from the selected backend modality and file extension.

## Image examples

### TensorFlow / radio SimCLR encoder

The historical standalone SimCLR extraction setup is represented by the `simclr_radio` preprocessing profile:

```bash
fextractor \
  --backend tensorflow \
  --inputfile metadata.json \
  --datalist-key data \
  --model encoder-resnet18_simclr_hulk256-smgps_ch1_100epochs.h5 \
  --model-weights encoder_weights-resnet18_simclr_hulk256-smgps_ch1_100epochs.h5 \
  --profile simclr_radio \
  --outfile featdata.json
```

The profile defines:

- image size: `224`;
- input channels: `1`;
- zscale enabled;
- zscale contrast: `0.25`;
- min-max normalization to `[0, 1]`.

`fextractor` assumes that a TensorFlow model supplied for representation extraction is already the required encoder/representation model. Training graphs and SimCLR training code remain outside this package.

### DINOv2

```bash
fextractor \
  --backend dinov2 \
  --inputfile source.fits \
  --zscale \
  --outfile source_dinov2.json
```

A different Hugging Face model can be selected with `--model`.

### DINOv3

```bash
fextractor \
  --backend dinov3 \
  --inputfile source.fits \
  --outfile source_dinov3.json
```

### SigLIP / SigLIP2

```bash
fextractor \
  --backend siglip2 \
  --inputfile source.fits \
  --outfile source_siglip2.json
```

The SigLIP backends use their Hugging Face image processor for model-specific preprocessing.

## Time-series inputs

Time-series data are converted internally to a canonical `TimeSeries` representation with shape

```text
[n_time, n_channels]
```

and may contain:

- timestamps;
- one or more value channels;
- measurement uncertainties;
- observed/interpolated/predicted masks;
- channel names;
- metadata.

Supported file extensions include:

```text
.csv
.ecsv
.fits
.fit
.fts
.npy
.npz
```

JSON datalists containing either file references or inline arrays are also supported.

### Long layout

A typical long-format table contains one sample per row:

```text
time,flux,flux_err
0.0,1.02,0.04
1.0,1.14,0.05
2.0,0.97,0.04
...
```

Example:

```bash
fextractor \
  --backend chronos2 \
  --inputfile lightcurve.csv \
  --time-column time \
  --value-columns flux \
  --error-columns flux_err \
  --outfile chronos_features.json
```

For multivariate input:

```bash
fextractor \
  --backend chronos2 \
  --inputfile series.csv \
  --time-column time \
  --value-columns flux hardness \
  --error-columns flux_err hardness_err \
  --outfile features.json
```

### Wide layout

Windowed time series can also be encoded as one table row.

For irregular sampling:

```text
t0,t1,t2,...,flux_0,flux_1,flux_2,...
```

Use for example:

```bash
fextractor \
  --backend chronos2 \
  --inputfile sample.csv \
  --timeseries-layout wide \
  --time-prefix t \
  --value-prefixes flux_ \
  --channel-names flux \
  --outfile features.json
```

For regularly sampled wide data, timestamps may instead be reconstructed from a start time and cadence:

```bash
fextractor \
  --backend chronos2 \
  --inputfile sample.csv \
  --timeseries-layout wide \
  --time-start-column t_start \
  --cadence-column dt \
  --value-prefixes flux_ \
  --channel-names flux \
  --outfile features.json
```

Multiple channels and corresponding uncertainty prefixes are supported.

### Inline JSON time series

Datalist entries do not have to reference external files. Arrays may be stored directly in each record.

Example:

```json
{
  "data": [
    {
      "sname": "source-1",
      "t_start": 0.0,
      "dt": 1.0,
      "flux": [1.0, 1.2, 0.9, 1.1]
    }
  ]
}
```

Extraction:

```bash
fextractor \
  --backend chronos2 \
  --inputfile metadata.json \
  --value-columns flux \
  --time-start-key t_start \
  --cadence-key dt \
  --outfile features.json
```

## Time-series preprocessing

Time-series preprocessing is performed independently of the representation model.

### Time coordinate

Available transformations:

```text
none
origin
```

`origin` subtracts the first timestamp.

### Value transformations

Values can be transformed independently per channel using:

```text
none
maxabs
minmax
standard
asinh
```

Example:

```bash
--value-transform standard
```

For `asinh`, an optional scale can be supplied:

```bash
--value-transform asinh \
--value-transform-scale 0.1
```

### Temporal alignment

Series can be aligned around a detected feature:

```text
peak-max
peak-min
peak-abs
```

Example:

```bash
--alignment peak-max \
--alignment-window-before 50 \
--alignment-window-after 100
```

This shifts the selected anchor to time zero and optionally crops the physical time interval around it.

### Regularization

Chronos-2 and Moirai-2 operate on regularly sampled sequences when timestamps are present.

Irregular time series can therefore be regularized before representation extraction:

```bash
--regularize \
--cadence 1.0
```

Two regularization approaches are available:

```text
bin
gp
```

#### Bin regularization

```bash
--regularization-method bin \
--bin-aggregation mean
```

or, when uncertainties are available:

```bash
--bin-aggregation inverse-variance
```

Missing bins may use:

```text
nan
linear
pchip
akima
cubic
```

through `--missing-strategy`.

#### Gaussian-process regularization

```bash
fextractor \
  --backend chronos2 \
  --inputfile lightcurve.csv \
  --time-column time \
  --value-columns flux \
  --error-columns flux_err \
  --regularize \
  --regularization-method gp \
  --cadence 1.0 \
  --outfile features.json
```

The GP uses a Matérn-3/2 model. Optional parameters are available for the kernel amplitude, correlation scale and noise floor:

```bash
--gp-sigma ...
--gp-rho ...
--gp-jitter ...
```

## Chronos-2

Basic extraction:

```bash
fextractor \
  --backend chronos2 \
  --inputfile lightcurve.csv \
  --time-column time \
  --value-columns flux \
  --regularize \
  --cadence 1.0 \
  --outfile chronos_features.json
```

Useful backend options include:

```bash
--aggregation mean_std
--context-length 512
--batch-size 256
--device cuda
```

If CUDA is requested but unavailable, the backend falls back to CPU.

## Moirai-2

Basic extraction:

```bash
fextractor \
  --backend moirai2 \
  --inputfile lightcurve.csv \
  --time-column time \
  --value-columns flux \
  --regularize \
  --cadence 1.0 \
  --outfile moirai_features.json
```

Moirai-specific representation packing can be configured with:

```bash
--patching-mode time_variate
--token-order by_variate
```

Available patching modes:

```text
time_only
time_variate
```

Available token orders for `time_variate`:

```text
by_variate
interleave_time
```

## Diagnostic time-series plots

`fextractor` can save diagnostic PNGs showing the original and/or preprocessed time series.

Install plotting support:

```bash
pip install -e '.[plot]'
```

Then use:

```bash
--timeseries-plot input
```

```bash
--timeseries-plot processed
```

or:

```bash
--timeseries-plot both
```

Example:

```bash
fextractor \
  --backend chronos2 \
  --inputfile lightcurve.csv \
  --time-column time \
  --value-columns flux \
  --regularize \
  --cadence 1.0 \
  --timeseries-plot both \
  --timeseries-plot-dir plots \
  --outfile features.json
```

When no plot directory is supplied, plots are written alongside the output JSON.

For datalists, diagnostic files are named using the entry index:

```text
000000_timeseries.png
000001_timeseries.png
...
```

Plots distinguish, where applicable:

- original observations;
- observed regularized bins;
- interpolated samples;
- Gaussian-process predictions;
- GP ±1σ intervals.

The lower x-axis represents sample/bin position while the upper axis reconstructs physical time whenever possible.

## Datalist format

The runner preserves the CAESAR/sclassifier-style datalist convention.

Typical image input:

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

After extraction, each processed entry receives:

```json
"feats": [...]
```

A top-level `fextractor` block records extraction provenance such as:

- backend;
- modality;
- model;
- selected device;
- preprocessing configuration;
- representation aggregation and backend-specific options.

With `--skip-errors`, entries that cannot be processed are retained and receive:

```json
"fextractor_error": "..."
```

instead of aborting the entire datalist.

## Single-source output

For a single image or time-series file, output has the form:

```json
{
  "feats": [
    ...
  ],
  "fextractor": {
    "backend": "...",
    "modality": "...",
    "...": "..."
  }
}
```

## Architecture

The central interface is:

```python
FeatureExtractor
```

Each backend declares a modality and implements model loading plus feature extraction.

Backends are registered through the central registry, which stores:

- canonical backend name;
- modality;
- extractor implementation;
- factory;
- aliases;
- default model;
- optional dependency group.

Model loading is lazy: models are loaded only when extraction is first requested.

The high-level flow is:

```text
input
  |
  v
input-type detection
  |
  v
domain preprocessing
  |
  v
backend/model preprocessing
  |
  v
pretrained model
  |
  v
representation
  |
  v
optional token aggregation
  |
  v
feature vector + provenance
```

This separation allows additional image or time-series embedding models to reuse the existing CLI, datalist runner, preprocessing infrastructure and output contract.

## Preprocessing philosophy

`fextractor` separates two kinds of preprocessing.

### Domain preprocessing

This represents transformations that belong to the scientific data rather than to a particular neural network.

For images this includes, for example:

- invalid-pixel handling;
- clipping;
- zscale;
- normalization.

For time series this includes:

- timestamp ordering;
- coordinate transformations;
- value transformations;
- alignment;
- windowing;
- regularization;
- interpolation.

### Model preprocessing

This belongs to the selected pretrained model and is implemented inside each backend.

Examples include:

- image resizing;
- channel conversion;
- ImageNet-style normalization;
- Hugging Face processors;
- Chronos input representation;
- Moirai patch/token construction.

Keeping the two layers separate prevents model assumptions from leaking into the scientific preprocessing pipeline.

## Python API

Backends can also be created programmatically through the common configuration/factory interface.

```python
from fextractor.config import ExtractorConfig
from fextractor.factory import create_extractor

config = ExtractorConfig(
    backend="chronos2",
    model="amazon/chronos-2",
    device="cuda",
)

extractor = create_extractor(config)

features = extractor.extract("lightcurve.csv")
```

For advanced time-series workflows, `TimeSeries` objects can also be prepared explicitly and passed to a time-series extractor.

## Containers

Backend-specific container definitions are maintained under:

```text
containers/
```

Current runtime families include:

```text
fextractor-tf
fextractor-torch
fextractor-chronos
fextractor-moirai
fextractor-fats
```

The separate images avoid forcing mutually incompatible or heavyweight ML stacks into a single runtime.

### Legacy FATS runtime

`containers/fextractor-fats` provides a dedicated runtime for the legacy `FATS` package and handcrafted time-series feature extraction.

Because FATS requires its own Python 2.7 environment, it is kept separate from the Python 3 `fextractor` backend registry while exposing a compatible JSON/datalist-oriented execution model for containerized workflows.

## Extending fextractor

Adding a new learned representation backend normally consists of:

1. implementing a `FeatureExtractor` or modality-specific subclass;
2. defining its configuration factory;
3. registering a `BackendSpec` in `fextractor.registry`;
4. adding its optional dependencies;
5. adding unit/integration tests.

For token-based time-series models, subclassing `TokenTimeSeriesFeatureExtractor` allows the existing aggregation machinery to be reused.

The common runner and output contract do not need to change for each new model.

## Development

Install test dependencies:

```bash
pip install -e '.[test]'
```

Run the unit test suite:

```bash
pytest
```

Run static compilation:

```bash
python -m compileall -q fextractor tests
```

Integration tests requiring heavyweight external models are marked with:

```text
integration
```

All Python source files in this repository use tab indentation.

## Scope

`fextractor` is an **inference-only** project.

Model training remains in the projects that own the corresponding training pipelines. For example, an exported SimCLR encoder is treated as a normal TensorFlow representation model rather than making SimCLR training part of `fextractor`.

The goal of this project is to provide a consistent and extensible representation-extraction layer for scientific workflows, including standalone use and integration into CAESAR/CAESAR-REST processing pipelines.

## License

See [LICENSE](LICENSE) and [COPYRIGHT.md](COPYRIGHT.md).
