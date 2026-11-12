# Generates a PEP 503 simple index for sgl-fa3 wheels under
# <whl-repo>/cu<version>/sgl-fa3/index.html, so consumers can
# `pip install sgl-fa3 --extra-index-url https://...whl/cu130`.
# Mirrors sglang's scripts/update_flashmla_whl_index.py.

import argparse
import hashlib
import pathlib
import re

SUPPORTED_CUDA_VERSIONS = ["130"]


def update_wheel_index(cuda_version, wheel_dir, whl_repo):
    index_dir = pathlib.Path(whl_repo) / f"cu{cuda_version}" / "sgl-fa3"
    index_dir.mkdir(exist_ok=True, parents=True)
    base_url = "https://github.com/sgl-project/whl/releases/download"

    suffix = f"+cu{cuda_version}"
    for path in sorted(pathlib.Path(wheel_dir).glob("*.whl")):
        if suffix not in path.name:
            continue
        with open(path, "rb") as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()
        match = re.match(r"sgl_fa3-([0-9][^-+]*)(?:\+cu[0-9]+)?-", path.name)
        if not match:
            continue
        ver = match.group(1)
        full_url = f"{base_url}/sgl-fa3-v{ver}/{path.name}#sha256={sha256}"
        with (index_dir / "index.html").open("a") as f:
            f.write(f'<a href="{full_url}">{path.name}</a><br>\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda", type=str, required=True, choices=SUPPORTED_CUDA_VERSIONS
    )
    parser.add_argument("--wheel-dir", type=str, default="dist")
    parser.add_argument(
        "--whl-repo",
        type=str,
        default="sgl-whl",
        help="Checkout of sgl-project/whl to write the index into",
    )
    args = parser.parse_args()
    update_wheel_index(args.cuda, args.wheel_dir, args.whl_repo)


if __name__ == "__main__":
    main()
