# sgl-fa3

The FlashAttention-3 forward kernels [SGLang](https://github.com/sgl-project/sglang)
serves with, packaged as a standalone wheel. Until this package existed they
were compiled into `sglang-kernel` as `sgl_kernel.flash_ops`.

The kernels come from `../hopper` in this repository; this directory holds only
the torch op registration, the Python wrapper, the tests and the build and
release scripts.

## What is built

- Forward only, varlen only (`FLASHATTENTION_DISABLE_BACKWARD`,
  `FLASHATTENTION_VARLEN_ONLY`); sparse mask kernels off by default.
- `sm_90a`, plus `sm_80`/`sm_86` (A100, A10, L40, 4090). No SM100+ targets.
- BF16, FP16 and FP8 (E4M3).
- x86_64 Linux, CUDA 13.0 (`+cu130`), one abi3 wheel for CPython 3.11+.
- Pinned to `torch==2.14.1`: the extension uses the regular torch C++ ABI, so
  a release only loads against the torch it was built with.

## Usage

```python
from sgl_fa3 import flash_attn_varlen_func, flash_attn_with_kvcache, get_scheduler_metadata
```

The ops are registered as `torch.ops.sgl_fa3.fwd` and
`torch.ops.sgl_fa3.get_scheduler_metadata`.

## Building

```bash
sgl_fa3/scripts/build.sh 3.11 13.0
```

builds in a `pytorch/manylinux2_28-builder:cuda13.0` container and writes
`sgl_fa3/dist/` (`+cu130` wheel for the sgl-whl index) and
`sgl_fa3/dist-pypi/` (same wheel without the local version, for PyPI). The
build reads `../hopper`, so it needs a full checkout; an sdist is not supported.

For a local build against an installed torch:

```bash
cd sgl_fa3 && pip install . --no-build-isolation -v
```

## Releasing

Releases are cut by the `Release sgl-fa3` workflow in sgl-project/sglang
(`.github/workflows/release-whl-fa3.yml`). It checks out this branch, writes the
version into `python/sgl_fa3/_version.py`, runs `scripts/build.sh`, tests the
wheel on H200 with `tests/`, then publishes to PyPI and the
sgl-project/whl index (`scripts/update_whl_index.py`).
