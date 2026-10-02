"""Pack supplied files into a checksummed plain-file bundle."""
from pathlib import Path, PurePosixPath
import copy
import gzip
import hashlib
import io
import json
import pickle


from beacon.data import sha256 as digest


def encode(path, kind):
    if kind not in ("array", "arrays", "sampled_matrix", "expression"):
        raw = path.read_bytes()
        return (gzip.decompress(raw) if path.suffix == ".gz" else raw), {}
    import numpy as np
    if kind == "expression":
        import pandas as pd
        frame = pd.read_csv(path, index_col=0)
        arrays = {"values": frame.to_numpy(), "genes": frame.index.to_numpy(dtype=str), "cells": frame.columns.to_numpy(dtype=str)}
    elif kind == "sampled_matrix":
        # Only pack explicitly supplied, trusted local pickle inputs.
        with path.open("rb") as handle:
            values = np.asarray(pickle.load(handle))
        arrays = {"values": values}
    elif kind == "array":
        arrays = {"values": np.load(path, allow_pickle=False)}
    else:
        with np.load(path, allow_pickle=False) as source:
            arrays = {key: source[key] for key in source.files}
    if any(value.dtype.hasobject for value in arrays.values()):
        raise ValueError(f"Object arrays are not permitted: {path}")
    out = io.BytesIO()
    np.savez_compressed(out, **arrays)
    raw = out.getvalue()
    with np.load(io.BytesIO(raw), allow_pickle=False) as check:
        for key, value in arrays.items():
            other = check[key]
            if value.dtype != other.dtype or value.shape != other.shape or value.tobytes() != other.tobytes():
                raise ValueError(f"Array conversion changed bits: {path}, {key}")
            if (value.flags.f_contiguous, value.flags.c_contiguous) != (other.flags.f_contiguous, other.flags.c_contiguous):
                raise ValueError(f"Array conversion changed memory order: {path}, {key}")
    return raw, {key: {"dtype": str(value.dtype), "shape": list(value.shape), "fortran": bool(value.flags.f_contiguous)} for key, value in arrays.items()}


def pack(files, root, destination, *, metadata=None, transform=None, plan_only=False):
    """Return a packing plan with source paths; publish only file metadata.

    ``transform(item, raw, arrays)`` may return replacement bytes and array
    metadata. Callers supply publication-ready conditions and metadata.
    """
    plan = dict(copy.deepcopy(metadata or {}), assets=copy.deepcopy(files))
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination == root or destination.is_relative_to(root):
        raise ValueError("Bundle destination must be outside the source tree")
    totals, names = {}, set()
    for item in plan["assets"]:
        name = item["path"]
        relative = PurePosixPath(name)
        if relative.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")) or len(relative.parts) < 2:
            raise ValueError(f"Invalid bundle path: {name}")
        if name in names or relative.name == "manifest.json":
            raise ValueError(f"Duplicate file or reserved manifest path: {name}")
        names.add(name)
        path = (root / item["source"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or escaping allowlisted input: {path}")
        if path.suffix in (".h5ad", ".tar", ".zip") or "RAW.tar" in path.name:
            raise ValueError(f"Raw archive forbidden: {path}")
        totals[item["category"]] = totals.get(item["category"], 0) + path.stat().st_size
    print(json.dumps({"source_bytes_per_category": totals, "files": len(plan["assets"])}, indent=2))
    if plan_only:
        return plan
    destination.mkdir(parents=True, exist_ok=False)
    sizes = {}
    for item in plan["assets"]:
        source = root / item["source"]
        original_digest = digest(source)
        if item.get("expected_sha256") and item["expected_sha256"] != original_digest:
            raise ValueError(f"Frozen input changed: {source}")
        raw, arrays = encode(source, item["kind"])
        if transform is not None:
            raw, arrays = transform(item, raw, arrays)
        if digest(source) != original_digest:
            raise ValueError(f"Input changed during packaging: {source}")
        target = destination / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(raw)
        file_digest = hashlib.sha256(raw).hexdigest()
        if digest(target) != file_digest:
            raise ValueError(f"Written file checksum differs: {target}")
        item.update(original_sha256=original_digest, sha256=file_digest, source_bytes=source.stat().st_size,
                    file_bytes=len(raw), arrays=arrays)
        sizes[item["category"]] = sizes.get(item["category"], 0) + len(raw)
    plan.update(status="complete", layout="plain_files", source_bytes_per_category=totals, file_bytes_per_category=sizes)
    with (destination / "manifest.json").open("x") as stream:
        public = dict(plan, assets=[{key: item[key] for key in
            ("path", "category", "role", "kind", "conditions", "original_sha256", "sha256", "source_bytes", "file_bytes", "arrays")
            if key in item} for item in plan["assets"]])
        json.dump(public, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"file_bytes_per_category": sizes, "status": "complete"}, indent=2))

    return plan
