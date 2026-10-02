"""Tiny plain-file build/download round trip; no model fits."""
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch
import copy
import gzip
import io
import json
import tarfile
import tempfile
import numpy as np
import pandas as pd
from beacon.data import Bundle, sha256
from data.pack import pack
from data.download import main as download


def rejected(function):
    try:
        function()
    except (ValueError, FileExistsError):
        return
    raise AssertionError("Invalid bundle operation was accepted")


def main():
    with tempfile.TemporaryDirectory(prefix="beacon_bundle_") as directory:
        root = Path(directory)
        source = root / "source"
        source.mkdir()
        values = np.asfortranarray(np.array([[0., -0., np.inf], [1., np.nan, -np.inf]], dtype=np.float32))
        np.savez_compressed(source / "predictions.npz", expression=values, row_major=np.ascontiguousarray(values), labels=np.array([1, 0], np.int8))
        pd.DataFrame(np.arange(12, dtype=np.float64).reshape(3, 4) / 7, index=["A", "B", "C"]).to_csv(source / "expression.csv")
        table = b"gene\tvalue\nA\t1\n"
        (source / "genes.tsv.gz").write_bytes(gzip.compress(table))
        (source / "metadata.json").write_text('{"context": "synthetic"}\n')
        assets = [dict(source=name, path=target, category=target.split("/")[0], kind=kind,
                       conditions={"context": "synthetic"}) for name, target, kind in (
            ("predictions.npz", "predictions/synthetic/split_42/seed_42.npz", "arrays"),
            ("expression.csv", "beeline/synthetic/expression.npz", "expression"),
            ("genes.tsv.gz", "prepared/synthetic/genes.tsv", "bytes"),
            ("metadata.json", "prepared/synthetic/metadata.json", "bytes"))]
        destination = root / "data"
        with redirect_stdout(io.StringIO()):
            pack(assets, source, destination, plan_only=True)
            assert not destination.exists()
            packed = pack(assets, source, destination)
            rejected(lambda: pack(assets, source, destination))
            duplicate = copy.deepcopy(assets)
            duplicate.append(duplicate[0].copy())
            rejected(lambda: pack(duplicate, source, root / "duplicate"))
            escaping = copy.deepcopy(assets)
            escaping[0]["path"] = "../outside.npz"
            rejected(lambda: pack(escaping, source, root / "escaping"))
            frozen = copy.deepcopy(assets)
            frozen[0]["expected_sha256"] = "0" * 64
            rejected(lambda: pack(frozen, source, root / "changed_source"))
            assert not (root / "changed_source/manifest.json").exists()
        config = root / "config.toml"
        config.write_text('data_root = "data"\n')
        bundle = Bundle(config)
        bundle.verify()
        assert len(list(destination.rglob("manifest.json"))) == 1
        for item, original in zip(bundle.assets, packed["assets"]):
            assert "source" not in item
            assert item["original_sha256"] == sha256(source / original["source"])
            assert item["sha256"] == sha256(destination / item["path"])
            assert item["conditions"] == {"context": "synthetic"}
        # A reviewer can load the same file directly, without the package loader.
        with np.load(destination / assets[0]["path"], allow_pickle=False) as opened:
            assert opened["expression"].tobytes() == values.tobytes()
            assert opened["labels"].dtype == np.dtype("int8")
        actual = bundle.arrays(0)
        assert actual["expression"].dtype == values.dtype and actual["expression"].tobytes() == values.tobytes()
        assert actual["expression"].flags.f_contiguous and bundle.assets[0]["arrays"]["expression"]["fortran"]
        assert actual["row_major"].flags.c_contiguous and actual["row_major"].tobytes() == values.tobytes()
        frame, restored = pd.read_csv(source / "expression.csv", index_col=0), bundle.frame(1, index_col=0)
        for dtype in (None, np.float32):
            left, right = frame.to_numpy(dtype=dtype), restored.to_numpy(dtype=dtype)
            assert left.dtype == right.dtype and left.tobytes() == right.tobytes()
            assert left.flags.f_contiguous == right.flags.f_contiguous
        assert bundle.path(2).read_bytes() == table
        assert bundle.frame(2, sep="\t").gene.tolist() == ["A"]
        assert bundle.json(3) == {"context": "synthetic"}
        archive = root / "bundle.tar.gz"
        with tarfile.open(archive, "w:gz") as handle:
            for path in sorted(destination.rglob("*")):
                if path.is_file():
                    handle.add(path, arcname=str(path.relative_to(destination)))
        def fetch(archive, expected=None):
            config.write_text(f'data_root = "downloaded"\nbundle_url = "{archive.as_uri()}"\nbundle_sha256 = "{expected or sha256(archive)}"\n')
            with patch("sys.argv", ["download", "--config", str(config)]), redirect_stdout(io.StringIO()):
                download()
        rejected(lambda: fetch(archive, "0" * 64))
        changed = root / "changed.tar.gz"
        with tarfile.open(changed, "w:gz") as handle:
            for path in sorted(destination.rglob("*")):
                if path.is_file():
                    name = str(path.relative_to(destination))
                    if name == assets[0]["path"]:
                        item = tarfile.TarInfo(name)
                        item.size = 1
                        handle.addfile(item, io.BytesIO(b"x"))
                    else:
                        handle.add(path, arcname=name)
        rejected(lambda: fetch(changed))
        unsafe = root / "unsafe.tar.gz"
        with tarfile.open(unsafe, "w:gz") as handle:
            item = tarfile.TarInfo("../outside.txt")
            item.size = 1
            handle.addfile(item, io.BytesIO(b"x"))
        rejected(lambda: fetch(unsafe))
        assert not (root / "downloaded").exists()
        fetch(archive)
        downloaded = Bundle(config)
        downloaded.verify()
        assert downloaded.arrays(0)["expression"].tobytes() == values.tobytes()
        downloaded.path(0).write_bytes(b"corrupt")
        rejected(downloaded.verify)
        rejected(lambda: downloaded.arrays(0))
        downloaded.assets[0]["path"] = "../outside.npz"
        rejected(lambda: downloaded.path(0))
    print("Plain-file build/download, byte/layout preservation, provenance and rejection checks passed")


if __name__ == "__main__":
    main()
