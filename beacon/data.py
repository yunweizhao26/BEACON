"""One configured data root containing ordinary, checksummed files."""
from pathlib import Path, PurePosixPath
import hashlib
import io
import json
try:
    import tomllib
except ImportError:
    from pip._vendor import tomli as tomllib

RELEASE = Path(__file__).resolve().parents[1]

def configuration(path=None):
    path = Path(path or RELEASE / "config.toml").resolve()
    with path.open("rb") as handle:
        value = tomllib.load(handle)
    value["data_root"] = (path.parent / value["data_root"]).resolve()
    return value

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

class Bundle:
    def __init__(self, config=None):
        self.root = configuration(config)["data_root"]
        raw = (self.root / "manifest.json").read_bytes()
        self.manifest_sha256 = hashlib.sha256(raw).hexdigest()
        self.manifest = json.loads(raw)
        if self.manifest.get("status") != "complete" or self.manifest.get("layout") != "plain_files":
            raise ValueError("Expected a complete plain-file data bundle")
        self.assets = self.manifest["assets"]
        if len({item["path"] for item in self.assets}) != len(self.assets):
            raise ValueError("Duplicate bundle file paths")

    def path(self, record):
        """Return the ordinary file path for a manifest record."""
        item = self.assets[record]
        name = item["path"]
        if PurePosixPath(name).is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
            raise ValueError("Invalid bundle file path")
        path = (self.root / name).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Bundle member escapes data_root")
        return path

    def read(self, record):
        item = self.assets[record]
        raw = self.path(record).read_bytes()
        if hashlib.sha256(raw).hexdigest() != item["sha256"]:
            raise ValueError(f"Bundle checksum mismatch for record {record}")
        return raw

    def arrays(self, record):
        import numpy as np
        with np.load(io.BytesIO(self.read(record)), allow_pickle=False) as data:
            return {name: data[name] for name in data.files}

    def array(self, record):
        return self.arrays(record)["values"]

    def json(self, record):
        return json.loads(self.read(record))

    def frame(self, record, **kwargs):
        import pandas as pd
        if self.assets[record].get("kind") == "expression":
            if kwargs != {"index_col": 0}:
                raise ValueError("Expression tables require index_col=0")
            arrays = self.arrays(record)
            return pd.DataFrame(arrays["values"], index=arrays["genes"], columns=arrays["cells"])
        return pd.read_csv(io.BytesIO(self.read(record)), **kwargs)

    def dataset(self, dataset):
        import numpy as np
        entry = next(row for row in self.manifest["datasets"] if row["dataset"] == dataset)
        frame = self.frame(entry["expression"], index_col=0)
        if frame.index.has_duplicates:
            raise ValueError("Duplicate expression gene names")
        genes = frame.index.astype(str).tolist()
        expression = frame.to_numpy(dtype=np.float32)
        lookup = {gene: i for i, gene in enumerate(genes)}
        network = self.frame(entry["network"])
        if network.shape[1] < 2:
            raise ValueError("Network needs two columns")
        truth = np.zeros((len(genes), len(genes)), dtype=np.int8)
        for source, target in network.iloc[:, :2].itertuples(index=False, name=None):
            source, target = lookup.get(str(source)), lookup.get(str(target))
            if source is not None and target is not None and source != target:
                truth[source, target] = 1
        if not truth.sum():
            raise ValueError("No network edges overlap expression genes")
        tfs = set(self.frame(entry["transcription_factors"])["TF"].astype(str))
        tf_indices = np.array([i for i, gene in enumerate(genes) if gene in tfs], dtype=int)
        return expression, np.array(genes), truth, tf_indices

    def verify(self):
        for record, item in enumerate(self.assets):
            if sha256(self.path(record)) != item["sha256"]:
                raise ValueError(f"Bundle file checksum mismatch: {item['path']}")

def write_result(directory, arrays, log, condition):
    import numpy as np
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for name, values in arrays.items():
        if not name.replace("_", "").isalpha():
            raise ValueError("Output roles must be plain descriptive names")
        np.savez_compressed(directory / (name + ".npz"), **values)
    (directory / "manifest.json").write_text(json.dumps({"condition": condition, "training_log": log}, indent=2, default=float) + "\n")
