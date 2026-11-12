#!/bin/bash
# Build the sgl-fa3 wheel inside a CUDA-versioned manylinux container.
#
# Usage: sgl_fa3/scripts/build.sh <PYTHON_VERSION> <CUDA_VERSION>
#   PYTHON_VERSION: e.g. 3.11 -- only picks the interpreter that drives the build.
#                   The wheel is abi3 (cp311-abi3) and serves CPython 3.11+.
#   CUDA_VERSION:   13.0
#
# Only x86_64 is built: FA3 targets sm_80/sm_86/sm_90a, and no aarch64 host
# SGLang serves on (GH200/GB200/GB300) needs it.
#
# Writes:
#   sgl_fa3/dist/      -- wheel tagged +cu130 and manylinux2014_x86_64
#   sgl_fa3/dist-pypi/ -- the same wheel with +cu130 stripped
#                         (PyPI rejects local-version segments)
#
# Env knobs: BUILD_JOBS (parallel compile units, default min(2/3 * nproc, 64)),
#            GITHUB_ARTIFACTORY (GitHub mirror for the CUTLASS download).
set -ex

if [ $# -lt 2 ]; then
  echo "Usage: $0 <PYTHON_VERSION> <CUDA_VERSION>"
  exit 1
fi

PYTHON_VERSION="$1"
CUDA_VERSION="$2"
ARCH="x86_64"

case "${CUDA_VERSION}" in
  13.0) CU_TAG=cu130 ;;
  *)
    echo "Unsupported CUDA_VERSION: ${CUDA_VERSION} (only 13.0 is published)" >&2
    exit 1
    ;;
esac

if [ "$(uname -m)" != "${ARCH}" ]; then
  echo "sgl-fa3 is only built on ${ARCH}, this host is $(uname -m)" >&2
  exit 1
fi

BASE_IMG="pytorch/manylinux2_28-builder"
PY_TAG="cp${PYTHON_VERSION//.}-cp${PYTHON_VERSION//.}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PKG_DIR}/.." && pwd)"
PKG_NAME="$(basename "${PKG_DIR}")"

DEPS_TAG="sgl-fa3-deps:cuda${CUDA_VERSION}-${PY_TAG}-${ARCH}"
BUILD_JOBS="${BUILD_JOBS:-0}"
GITHUB_ARTIFACTORY="${GITHUB_ARTIFACTORY:-github.com}"

echo "----------------------------------------"
echo "PYTHON_VERSION: ${PYTHON_VERSION}"
echo "CUDA_VERSION:   ${CUDA_VERSION}"
echo "CU_TAG:         ${CU_TAG}"
echo "BASE_IMG:       ${BASE_IMG}"
echo "PKG_DIR:        ${PKG_DIR}"
echo "BUILD_JOBS:     ${BUILD_JOBS:-auto}"
echo "----------------------------------------"

docker build \
  -f "${PKG_DIR}/Dockerfile" "${PKG_DIR}" \
  --build-arg BASE_IMG="${BASE_IMG}" \
  --build-arg CUDA_VERSION="${CUDA_VERSION}" \
  --build-arg ARCH="${ARCH}" \
  --build-arg PYTHON_TAG="${PY_TAG}" \
  --build-arg CU_TAG="${CU_TAG}" \
  -t "${DEPS_TAG}" \
  --network=host

# The whole repository is mounted because the kernels are compiled from ../hopper.
# --no-build-isolation: CMake must find the torch the image pins.
docker run --rm \
  --network=host \
  -v "${REPO_ROOT}:/src" \
  -v "${SCRIPT_DIR}/rename_whl.sh:/rename_whl.sh:ro" \
  -w "/src/${PKG_NAME}" \
  -e BUILD_JOBS="${BUILD_JOBS}" \
  -e GITHUB_ARTIFACTORY="${GITHUB_ARTIFACTORY}" \
  "${DEPS_TAG}" \
  bash -c '
set -eux
if [ "${BUILD_JOBS}" -gt 0 ] 2>/dev/null; then
  export CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
else
  export CMAKE_BUILD_PARALLEL_LEVEL=$(echo "$(( $(nproc) * 2 / 3 )) 64" | awk "{print (\$1 < \$2) ? \$1 : \$2}")
fi
export CMAKE_ARGS="${CMAKE_ARGS:-} -DGITHUB_ARTIFACTORY=${GITHUB_ARTIFACTORY}"
echo "CMAKE_BUILD_PARALLEL_LEVEL=${CMAKE_BUILD_PARALLEL_LEVEL}"

# Cleared in here, not on the host: earlier runs left them root-owned.
rm -rf dist dist-pypi
python3 -m pip wheel . --no-build-isolation --no-deps -w dist -Cbuild-dir=build -v
bash /rename_whl.sh dist '"${CU_TAG}"' '"${ARCH}"'

# Sibling dist-pypi/ with the local version stripped.
mkdir -p dist-pypi
for w in dist/*.whl; do
  tmp=$(mktemp -d)
  python3 -m wheel unpack "$w" --dest "$tmp"
  unpacked=$(find "$tmp" -mindepth 1 -maxdepth 1 -type d | head -1)
  info=$(find "$unpacked" -maxdepth 1 -type d -name "*.dist-info" | head -1)
  meta="$info/METADATA"
  orig=$(grep "^Version:" "$meta" | head -1 | sed "s/^Version:[[:space:]]*//")
  new=$(echo "$orig" | sed "s/+cu[0-9]\+$//")
  if [ "$orig" != "$new" ]; then
    sed -i "s/^Version:.*/Version: ${new}/" "$meta"
    old_base=$(basename "$info")
    new_base="${old_base/${orig}/${new}}"
    mv "$info" "$(dirname "$info")/${new_base}"
  fi
  python3 -m wheel pack "$unpacked" --dest-dir dist-pypi
  rm -rf "$tmp"
done
'

echo "Wheels in ${PKG_DIR}/dist:"
ls -lh "${PKG_DIR}/dist"/*.whl
echo "PyPI-ready wheels in ${PKG_DIR}/dist-pypi:"
ls -lh "${PKG_DIR}/dist-pypi"/*.whl
