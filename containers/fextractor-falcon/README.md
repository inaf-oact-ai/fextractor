# fextractor-falcon

Docker container for running Falcon-1 time-series representation extraction with fextractor.

The runtime expects the Hugging Face `ant-intl/Falcon-TST_Large` checkpoint under:

```text
models/falcon1/
```

before building the image. The container copies that directory to `/opt/models/falcon1` and runs with Hugging Face offline mode enabled.
