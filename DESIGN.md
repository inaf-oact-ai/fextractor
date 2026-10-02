# Design notes

## Scope

`fextractor` is an inference-only package. It does not train SimCLR, DINO,
SigLIP, Chronos, Moirai, or other representation models. Training remains in
the packages/projects that own those models.

The package provides a common representation-extraction layer for multiple
scientific-data modalities. Current core modalities are images and time series.

## Backend architecture

All learned representation extractors implement the common `FeatureExtractor`
interface or a modality-specific subclass.

Each backend declares its modality and is registered through the central
backend registry. The registry stores the backend name, implementation, factory,
aliases, default model, and optional dependency group.

This allows the CLI, runner, datalist handling, output format, and provenance
machinery to remain independent of individual model families.

Models are loaded lazily on first extraction.

## Preprocessing split

`fextractor` deliberately separates domain preprocessing from model
preprocessing.

### Domain preprocessing

Domain preprocessing represents transformations that belong to the scientific
data rather than to a particular pretrained model.

For images this includes:

- scientific-image decoding;
- non-finite/blank handling;
- optional sigma clipping;
- optional zscale;
- min-max normalization.

For time series this includes:

- timestamp sorting;
- time-coordinate transforms;
- channel-wise value transforms;
- temporal alignment and windowing;
- regularization;
- missing-value interpolation.

### Model preprocessing

Model preprocessing is implemented by each extractor backend and contains
operations required by the selected pretrained model.

Examples include:

- image resize and channel conversion;
- ImageNet-style normalization;
- Hugging Face image processors;
- Chronos input preparation;
- Moirai patch/token construction;
- Falcon-1 masking, RevIN and latent-token extraction.

This split prevents model-specific assumptions from leaking into scientific
domain preprocessing.

## Image extraction

The image modality supports TensorFlow/Keras encoders and several pretrained
vision representation families, including DINOv2, DINOv3, SigLIP, and
SigLIP2.

The TensorFlow backend assumes the supplied model is already the desired
representation model/encoder. For the radio SimCLR use case, export/load the
encoder rather than the complete contrastive-training graph. This keeps
`fextractor` independent from the SimCLR training implementation.

## Time-series representation

Time-series data are represented internally by a canonical `TimeSeries`
object containing values with shape `[n_time, n_variates]` and, where
available:

- timestamps;
- uncertainties;
- observed/interpolated/predicted masks;
- channel names;
- metadata.

The I/O layer supports long and wide tabular layouts as well as arrays embedded
directly in JSON datalist records.

Chronos-2, Moirai-2 and Falcon-1 are currently implemented as learned
time-series representation backends.

They expose contextual token representations through the shared time-series
extractor abstraction. Token representations can then be converted into one
feature vector using a configurable aggregation strategy.

Falcon-1 does not expose a dedicated embedding API. Its forecasting backbone
contains multiple routed MoE experts plus a shared expert at each layer.
FEXTRACTOR captures the final normalized hidden patch states of the shared
expert from every MoE layer and concatenates them along the token axis. This
produces a stable representation with shape `[channel, token, hidden]` while
avoiding forecast/backcast values and sample-dependent token layouts from the
routed experts, whose patch sizes may differ.

Falcon-1 follows the upstream model's independent-channel semantics for
multivariate inputs. Model-native RevIN normalization remains inside Falcon;
timestamp regularization and domain preprocessing remain the responsibility
of the shared FEXTRACTOR preprocessing pipeline.

This separation is intentional: future time-series models should generally
need to implement model-specific token extraction while reusing the canonical
input representation, preprocessing pipeline, aggregation machinery, runner,
and output contract.

## Time-series regularization

Time-series preprocessing supports regularly and irregularly sampled input.

When a backend requires regular sampling, irregular observations can be mapped
to a fixed grid using either:

- bin-based regularization; or
- Gaussian-process regularization.

Bin regularization supports standard and inverse-variance aggregation where
measurement uncertainties are available.

Missing bins can optionally be interpolated.

Gaussian-process regularization retains predicted-sample and uncertainty
information in the canonical `TimeSeries` representation so diagnostic tools
can distinguish measured, interpolated, and model-predicted samples.

## Time-series alignment

Time series can optionally be aligned around a data-derived anchor such as a
maximum, minimum, or largest absolute excursion.

The detected anchor is retained in metadata. After alignment, the time
coordinate is shifted so the anchor is at zero, and an optional physical-time
window can be applied around it.

Alignment is part of domain preprocessing and is therefore independent of the
representation backend.

## Diagnostic plotting

Diagnostic time-series plotting operates on the canonical input and processed
`TimeSeries` objects rather than on backend-specific tensors.

This makes it possible to inspect the effects of preprocessing independently of
Chronos, Moirai, or future time-series representation models.

Plots can show the input series, processed series, or both, and distinguish
observed, interpolated, and predicted samples where applicable.

## Datalist compatibility

The runner preserves the existing CAESAR/sclassifier-style convention:

- input path: `item["filepaths"][0]` by default;
- single-path entries may use `item["filepath"]`;
- output vector: `item["feats"]` by default.

For time-series extractors, datalist entries may alternatively contain arrays
directly in the JSON record. This allows regularly sampled sequences to be
defined using a start time and cadence without requiring one external file per
record.

The output JSON also contains an `fextractor` provenance block describing the
selected backend and its configuration.

## Containers and dependency isolation

The supported model families require substantially different machine-learning
stacks. Backend-specific containers are therefore preferred over a monolithic
runtime.

Dedicated runtime families currently exist for TensorFlow, Torch image models,
Chronos, Moirai, and Falcon.

A separate FATS container provides compatibility with the legacy Python 2.7
FATS package. FATS is intentionally isolated from the Python 3 core package and
is not registered as a normal `fextractor` Python backend.

This keeps legacy dependency requirements from constraining the modern
representation-extraction framework.

## Extensibility

Adding a new learned backend should normally require:

1. a `FeatureExtractor` or modality-specific implementation;
2. a factory that translates `ExtractorConfig` into that implementation;
3. a registry entry;
4. optional dependency declarations;
5. tests.

New time-series embedding models should reuse `TimeSeries`,
`TimeSeriesFeatureExtractor`, and, where appropriate,
`TokenTimeSeriesFeatureExtractor`.

The high-level runner and REST-facing output contract should not require
model-specific changes.
