"""Download only the selected, pinned model's data into this project, and verify Hub digests.

    .venv-laya/Scripts/python.exe tools/prepare_models.py --model vega
    .venv-laya/Scripts/python.exe tools/prepare_models.py --model multilingual
    .venv-laya/Scripts/python.exe tools/prepare_models.py --model all --check
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / "models" / ".huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
MANIFEST = json.loads((ROOT / "data/python-models.json").read_text(encoding="utf-8"))


def digest(path, kind="sha256"):
    h = hashlib.new(kind)
    if kind == "sha1":
        h.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare(spec, check=False):
    directory = ROOT / spec["directory"]
    marker = directory / "local-laya-verified.json"
    if check:
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved["revision"] != spec["revision"] or saved["repo"] != spec["repo"]:
            raise ValueError(f"wrong revision in {marker}")
        files = saved["files"]
    else:
        from huggingface_hub import HfApi, snapshot_download, hf_hub_download

        info = HfApi().model_info(spec["repo"], revision=spec["revision"], files_metadata=True)
        files = {}
        available = {f.rfilename: f for f in info.siblings}
        for name in spec["files"]:
            f = available[name]
            files[name] = {"size": f.size, "kind": "sha256" if f.lfs else "sha1",
                           "digest": f.lfs.sha256 if f.lfs else f.blob_id}
        print(f"Downloading {spec['repo']} @ {spec['revision'][:12]} to {directory}", flush=True)
        snapshot_download(spec["repo"], revision=spec["revision"], local_dir=str(directory),
                          allow_patterns=spec["files"], max_workers=3)
    if set(files) != set(spec["files"]):
        raise ValueError(f"incomplete manifest in {marker}")
    for name, expected in files.items():
        file = directory / name
        def valid():
            return file.is_file() and file.stat().st_size == expected["size"] and digest(file, expected["kind"]) == expected["digest"]
        verified = valid()
        if not verified and not check:
            print(f"  restoring pinned {name}", flush=True)
            hf_hub_download(spec["repo"], name, revision=spec["revision"], local_dir=str(directory), force_download=True)
            verified = valid()
        if not verified:
            raise ValueError(f"digest mismatch: {file}")
        print(f"  verified {name} ({expected['size']:,} bytes)", flush=True)
    if not check:
        marker.write_text(json.dumps({"repo": spec["repo"], "revision": spec["revision"], "files": files}, indent=2) + "\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*MANIFEST["models"], "all"], default="vega")
    parser.add_argument("--check", action="store_true", help="verify cached files without using the network")
    args = parser.parse_args()
    for name, spec in MANIFEST["models"].items():
        if args.model not in (name, "all"):
            continue
        prepare(spec, args.check)
        if spec.get("backbone"):
            prepare(spec["backbone"], args.check)
    print("Model files verified. Inference uses local paths and offline mode.")


if __name__ == "__main__":
    main()
