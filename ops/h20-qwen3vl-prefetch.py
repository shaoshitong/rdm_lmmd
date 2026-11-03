#!/usr/bin/env python3
"""Prefetch the pinned Qwen3-VL-4B vision-containing shard on the remote node.

Official metadata defines every expected byte count/digest. A mirror may only
transport these exact files; no first shard or complete language model is loaded.
HF's normal cache locks and incomplete files provide resumable transfers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import struct
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

REPO = "Qwen/Qwen3-VL-4B-Instruct"
REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
OFFICIAL = "https://huggingface.co"
SHARD = "model-00002-of-00002.safetensors"
SHARD_BYTES = 3908490048
SHARD_SHA256 = "046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6"
VISUAL_KEYS = 315
REQUIRED_JSON = {"config.json", "preprocessor_config.json", "model.safetensors.index.json"}
RECEIPT = "qwen3vl-4b-vision-prefetch-verified.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def emit(event, **fields):
    print(json.dumps({"event": event, "time": now(), **fields}, sort_keys=True), flush=True)


def field(obj, name, default=None):
    return obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)


def official_inventory(info):
    """Validate pinned official metadata before any payload request."""
    if field(info, "sha") != REVISION:
        raise ValueError("official metadata did not resolve the pinned revision")
    rows = []
    for entry in field(info, "siblings", []):
        name = field(entry, "rfilename")
        if not isinstance(name, str) or not (name.endswith(".json") or name == SHARD):
            continue
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError("unsafe filename in official metadata")
        size = field(entry, "size")
        lfs = field(entry, "lfs")
        algorithm = "sha256" if lfs else "git_blob_sha1"
        digest = field(lfs, "sha256") if lfs else field(entry, "blob_id")
        width = 64 if lfs else 40
        if not isinstance(size, int) or size < 1 or not isinstance(digest, str) \
                or re.fullmatch(r"[0-9a-f]{%d}" % width, digest) is None:
            raise ValueError(f"missing official size/hash for {name}")
        rows.append({"name": name, "bytes": size, "algorithm": algorithm, "digest": digest})
    by_name = {row["name"]: row for row in rows}
    if len(by_name) != len(rows) or not REQUIRED_JSON.issubset(by_name) or SHARD not in by_name:
        raise ValueError("official inventory has duplicate or missing required files")
    shard = by_name[SHARD]
    if (shard["bytes"], shard["algorithm"], shard["digest"]) != \
            (SHARD_BYTES, "sha256", SHARD_SHA256):
        raise ValueError("official shard metadata disagrees with the approved pin")
    return sorted(rows, key=lambda row: row["name"])


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    try:
        with temporary.open("w") as output:
            json.dump(data, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_file(path, expected):
    if path.stat().st_size != expected["bytes"]:
        raise ValueError(f"size mismatch: {expected['name']}")
    sha256 = hashlib.sha256()
    official = sha256 if expected["algorithm"] == "sha256" else hashlib.sha1()
    if expected["algorithm"] == "git_blob_sha1":
        official.update(f"blob {expected['bytes']}\0".encode())
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            sha256.update(block)
            if official is not sha256:
                official.update(block)
    if official.hexdigest() != expected["digest"]:
        raise ValueError(f"official hash mismatch: {expected['name']}")
    return {**expected, "path": str(path), "sha256": sha256.hexdigest(), "verified": True}


def verify_visual_inventory(directory):
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    mapping = index.get("weight_map", {})
    visual = {key for key in mapping if key.startswith("model.visual.")}
    if len(visual) != VISUAL_KEYS or any(mapping[key] != SHARD for key in visual):
        raise ValueError("official index does not place all 315 visual keys in the selected shard")
    # Header inspection is CPU-only and never materializes any model tensors.
    with (directory / SHARD).open("rb") as source:
        length = source.read(8)
        if len(length) != 8:
            raise ValueError("truncated safetensors header")
        header_bytes = struct.unpack("<Q", length)[0]
        if not 1 <= header_bytes <= 64 * 1024**2:
            raise ValueError("invalid safetensors header length")
        header = json.loads(source.read(header_bytes))
    actual = {key for key in header if key.startswith("model.visual.")}
    if actual != visual:
        raise ValueError("safetensors visual keys disagree with the verified official index")
    return sorted(visual)


class CacheProgress:
    """Report observed cache bytes, including resumable .incomplete files."""
    def __init__(self, cache, rows, interval):
        self.cache, self.rows, self.interval = cache, rows, interval
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.previous = 0

    def report(self):
        files = []
        base = self.cache / ("models--" + REPO.replace("/", "--")) / "blobs"
        for row in self.rows:
            blob = base / row["digest"]
            available = 0
            for candidate in (blob, blob.with_name(blob.name + ".incomplete")):
                try:
                    available = max(available, candidate.stat().st_size)
                except FileNotFoundError:
                    pass
            files.append({"name": row["name"], "observed_bytes": available,
                          "expected_bytes": row["bytes"]})
        total = sum(row["observed_bytes"] for row in files)
        emit("cache_bytes", observed_bytes=total, delta_bytes=total - self.previous,
             expected_bytes=sum(row["bytes"] for row in self.rows), files=files,
             hashes_verified=False)
        self.previous = total

    def run(self):
        while not self.stop.wait(self.interval):
            self.report()

    def __enter__(self):
        self.report()
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.report()


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data/shaoshitong/rdm-sets"))
    parser.add_argument("--endpoint", default="https://hf-mirror.com",
                        help="Payload transport only; metadata always uses official HF")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--progress-seconds", type=float, default=5)
    parser.add_argument("--trust-env-transport", action="store_true",
                        help="Use existing proxy variables for payload requests")
    return parser.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    if platform.system() == "Darwin":
        raise RuntimeError("payload downloads are remote-only; refusing to run on Mac")
    if not args.root.is_absolute() or not args.root.is_dir():
        raise ValueError("--root must be an existing absolute remote directory")
    if args.max_workers < 1 or args.progress_seconds <= 0:
        raise ValueError("worker count and progress interval must be positive")
    endpoint = urlsplit(args.endpoint)
    if endpoint.scheme != "https" or not endpoint.netloc or endpoint.username \
            or endpoint.password or endpoint.query or endpoint.fragment or endpoint.path not in ("", "/"):
        raise ValueError("transport endpoint must be an HTTPS origin without credentials")
    # Public repo: no credential is needed or sent to the transport mirror.
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    import httpx
    import huggingface_hub as hf
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
    emit("official_metadata", repo=REPO, revision=REVISION, endpoint=OFFICIAL)
    info = hf.HfApi(endpoint=OFFICIAL, token=False).repo_info(
        REPO, repo_type="model", revision=REVISION, files_metadata=True)
    rows = official_inventory(info)
    cache = args.root / "hf/hub"
    control = args.root / "control"
    inventory = {"repo": REPO, "revision": REVISION, "official_endpoint": OFFICIAL,
                 "transport_endpoint": args.endpoint.rstrip("/"), "files": rows,
                 "metadata_checked_at": now(), "first_shard_requested": False}
    atomic_json(control / "qwen3vl-4b-prefetch-inventory.json", inventory)
    # Only the process-local transport client changes after official metadata.
    hf.set_client_factory(lambda: httpx.Client(
        trust_env=args.trust_env_transport, follow_redirects=True, timeout=120))
    emit("fetch", repo=REPO, revision=REVISION, selected_files=[row["name"] for row in rows],
         expected_bytes=sum(row["bytes"] for row in rows), cache=str(cache))
    with CacheProgress(cache, rows, args.progress_seconds):
        directory = Path(hf.snapshot_download(
            REPO, repo_type="model", revision=REVISION,
            allow_patterns=[row["name"] for row in rows], cache_dir=cache,
            max_workers=args.max_workers, endpoint=args.endpoint.rstrip("/"), token=False))
    if directory.name != REVISION:
        raise ValueError("download snapshot directory does not match the pinned revision")
    verified = []
    for row in rows:
        emit("hash_start", name=row["name"], bytes=row["bytes"])
        verified.append(verify_file(directory / row["name"], row))
        emit("hash_verified", name=row["name"], bytes=row["bytes"], algorithm=row["algorithm"])
    visual = verify_visual_inventory(directory)
    receipt = {**inventory, "files": verified, "snapshot": str(directory),
               "verified_at": now(), "all_official_hashes_verified": True,
               "visual_key_count": len(visual), "visual_keys": visual,
               "model_tensors_loaded": False, "first_shard_requested": False,
               "receipt_version": 1}
    atomic_json(control / RECEIPT, receipt)
    emit("complete", receipt=str(control / RECEIPT), snapshot=str(directory),
         bytes=sum(row["bytes"] for row in rows), visual_key_count=len(visual),
         all_official_hashes_verified=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        # Avoid echoing request URLs/headers or possible tokens from dependency errors.
        message = re.sub(r"https?://\S+", "[request URL omitted]", str(error))
        message = re.sub(r"hf_[A-Za-z0-9]+", "[token omitted]", message)
        emit("failed", error_type=type(error).__name__, message=message[:1200])
        raise SystemExit(1)
