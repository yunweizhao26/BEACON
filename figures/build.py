"""Checksummed manuscript inputs and descriptive figure/table outputs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from beacon.data import Bundle, RELEASE
POP = ("Prior target in-degree", "Prior regulator out-degree")
CONTEXT = {1501: "hESC", 1605: "mDC", 1709: "mHSC-E", 1801: "mESC"}

class Build:
    def __init__(self, config=None, out=None, selected=None):
        self.bundle = Bundle(config)
        self.directory = Path(out or RELEASE / "results/figures")
        self.directory.mkdir(parents=True, exist_ok=False)
        self.selected = selected
        self.base = self.assays = Path(".")
        self.outputs = []
        self.inputs = set()

    def record(self, path):
        key = str(path)
        if key in self.bundle.manifest["plotting"]:
            result = self.bundle.manifest["plotting"][key]
        else:
            result = self.bundle.manifest["tables"][key.removesuffix(".csv")]
        self.inputs.add(result)
        return result

    def csv(self, path, **kwargs):
        return self.bundle.frame(self.record(path), **kwargs)

    def js(self, path):
        return self.bundle.json(self.record(path))

    def npz(self, path):
        if isinstance(path, tuple):
            _, condition = path
            if path[0] == "sampled":
                from evaluation.metrics import popularity
                e, = [e for e in self.bundle.manifest["experiments"] if e["suite"] == "sampled_pairs" and e["condition"]["dataset"] == condition]
                data = self.bundle.arrays(e["reference"]["predictions"])
                dataset, = [d for d in self.bundle.manifest["datasets"] if d["dataset"] == condition]
                train = self.bundle.array(dataset["sampled"]["train"])
                return dict(edges=data["edges"], labels=data["labels"], **popularity(train, data["edges"]))
            e, = [e for e in self.bundle.manifest["experiments"] if e["suite"] == "external" and e["condition"]["context"] == "tcell_" + condition and e["condition"]["control"] == "beacon"]
            record = e["reference"]["training"]
        elif path == "response_bootstrap":
            record = self.bundle.manifest["prepared"]["bootstrap"]
        else:
            record = self.record(path)
        self.inputs.add(record)
        return self.bundle.arrays(record)

    def case(self, suite, dataset, seed, coverage=.8, role="metadata"):
        e, = [e for e in self.bundle.manifest["experiments"] if e["suite"] == suite and e["condition"]["dataset"] == dataset and e["condition"]["split_seed"] == seed and e["condition"]["coverage"] == coverage and e["condition"]["control"] == "beacon" and not e["condition"]["snn_weight"] and e["condition"]["ratio"] == 5 and not e["condition"]["corruption"]]
        record = e["reference"][role]
        self.inputs.add(record)
        return self.bundle.json(record) if role == "metadata" else self.bundle.arrays(record)

    def comparator(self, method, dataset, seed):
        c, = [c for c in self.bundle.manifest["comparators"] if c["suite"] == "fixed_pools" and c.get("dataset") == dataset and c["split_seed"] == seed and c["method"] == method and c["coverage"] == .8]
        return self.bundle.arrays(c["artifacts"]["predictions"])

    def validate_labels(self):
        from evaluation.responses import label_bundle
        for context, spec in self.bundle.manifest["labels"].items():
            data = self.bundle.arrays(spec["artifact"])
            label_bundle(self.bundle, context, data["target_genes"], data["tf_genes"])
        # Calibration plus source hashes link the saved rendering tables to their labels.
        for key, record in self.bundle.manifest["plotting"].items():
            if key.endswith(("manifest.json", "summary.json")) and ("evaluation/" in key or key.startswith(("tcells/", "rpe1/"))):
                info = self.bundle.json(record)
                expected = info.get("label_sha256")
                if expected and expected not in {self.bundle.assets[s["artifact"]]["original_sha256"] for s in self.bundle.manifest["labels"].values()}:
                    raise ValueError("Plot evaluation refers to a different response label bundle")

    def tcells(self):
        tables = {name: [] for name in ("per_tf_metrics", "aggregate", "paired_comparisons", "pooled_metrics")}
        for condition in ("resting", "stimulated"):
            for name in tables:
                frame = self.csv(f"tcells/{condition}/{name}.csv").query('definition == "primary"').copy()
                if frame.empty or set(frame.condition) != {condition}:
                    raise ValueError("Missing calibrated primary T-cell cohort")
                tables[name].append(frame)
        result = {name: pd.concat(parts, ignore_index=True) for name, parts in tables.items()}
        for condition, frame in result["per_tf_metrics"].groupby("condition"):
            wide = frame.pivot(index="tf", columns="method", values="average_precision")
            aggregate = result["aggregate"].query('condition == @condition').set_index("method")
            if wide.isna().any().any() or not np.allclose(wide.mean().loc[aggregate.index], aggregate.macro_ap_all_tf):
                raise ValueError("T-cell aggregate differs from regulator scores")
        return result

    def pools(self):
        frame = self.csv("metrics_ap_and_trapezoid.csv")
        metadata = self.csv("summary/completion_runs.csv")
        keys = ["run", "dataset_id", "split_seed", "coverage", "ratio", "corruption", "variant"]
        meta = metadata[keys].drop_duplicates()
        completion = frame[frame.suite.eq("completion")].drop(
            columns=["dataset_id", "split_seed", "coverage", "control", "opt_seed"]
        ).merge(meta, on="run", validate="many_to_one")
        if len(completion) != frame.suite.eq("completion").sum():
            raise ValueError("Completion metadata join lost rows")
        return frame, completion

    def save(self, fig, name, metric, notes):
        import matplotlib.pyplot as plt
        if self.selected is None or name == self.selected:
            for extension in ("png", "pdf"):
                path = self.directory / f"{name}.{extension}"
                if path.exists():
                    raise FileExistsError(path)
                fig.savefig(path, dpi=300, bbox_inches="tight")
                self.outputs.append(dict(output=path.name, metric=metric, notes=notes, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        plt.close(fig)

    def table(self, name, text, metric, notes):
        if self.selected is not None and name != self.selected:
            return
        path = self.directory / (name + ".tex")
        with path.open("x") as stream:
            stream.write(text)
        self.outputs.append(dict(output=path.name, metric=metric, notes=notes, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

    def finish(self):
        if not self.outputs:
            raise ValueError("No selected output was generated")
        (self.directory / "manifest.json").write_text(json.dumps(dict(status="complete", inputs=sorted(self.inputs), bundle_manifest_sha256=self.bundle.manifest_sha256, outputs=self.outputs), indent=2) + "\n")
