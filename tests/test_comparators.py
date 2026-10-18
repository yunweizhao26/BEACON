import json
import sys
import textwrap
import numpy as np
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from beacon.model import Training
from comparators.adapters import internal_split, Monitor, RULES, train_scregnet
from comparators.train import foundation

SCREGNET_STUB = {
    "models.py": '''
        import torch
        TRAINING_DROPOUT = []
        class scTransNet_GCN(torch.nn.Module):
            def __init__(self, input_dim, args, foundation_dim, device):
                super().__init__()
                self.args = args
                self.linear = torch.nn.Linear(2 * foundation_dim, 2)
            def forward(self, expression, adjacency, edges, foundation):
                if self.training:
                    TRAINING_DROPOUT.append(self.args.dropout)
                pair = torch.cat([foundation[edges[:, 0]], foundation[edges[:, 1]]], dim=1)
                return self.linear(torch.nn.functional.dropout(pair, p=self.args.dropout, training=self.training))
        ''',
    "utils.py": '''
        import numpy as np
        import scipy.sparse as sp
        import torch
        class scRNADataset(torch.utils.data.Dataset):
            def __init__(self, rows, n, flag=True):
                self.rows, self.n = rows, n
            def __len__(self):
                return len(self.rows)
            def __getitem__(self, i):
                label = np.zeros(2, dtype=np.float32)
                label[self.rows[i, 2]] = 1
                return torch.as_tensor(self.rows[i, :2]), torch.as_tensor(label)
            def Adj_Generate(self, tfs, loop=False):
                return sp.identity(self.n, dtype=np.float32, format="coo")
        def adj2saprse_tensor(adj):
            return torch.sparse_coo_tensor(np.vstack([adj.row, adj.col]), adj.data, adj.shape)
        class load_data:
            def __init__(self, frame):
                self.frame = frame
            def exp_data(self):
                return self.frame.to_numpy(dtype=np.float32)
        def set_seed(seed):
            torch.manual_seed(seed)
        ''',
}


def check_scregnet_dropout():
    import torch
    with TemporaryDirectory() as temporary:
        package = Path(temporary) / "src"
        package.mkdir()
        (package / "__init__.py").write_text("")
        for name, text in SCREGNET_STUB.items():
            (package / name).write_text(textwrap.dedent(text))
        for name in [m for m in sys.modules if m == "src" or m.startswith("src.")]:
            del sys.modules[name]
        rng = np.random.default_rng(0)
        train = np.full((6, 6), -1, dtype=np.int8)
        train[0, 1:4], train[1, 2:6] = 1, 0
        monitor = Monitor(dict(every=1, patience=10), np.array([[0, 4], [1, 0]]), np.array([1, 0]))
        train_scregnet(Path(temporary), train, np.array(list("ABCDEF")), np.array([0, 1]), pd.DataFrame(rng.normal(size=(6, 4))),
                       rng.normal(size=(6, 3)).astype(np.float32), torch.device("cpu"), 3, monitor)
        seen = sys.modules["src.models"].TRAINING_DROPOUT
        sys.path.remove(temporary)
    manifest = Path(__file__).resolve().parents[1] / "comparators/manifest.json"
    configured = json.loads(manifest.read_text())["scRegNet_configuration"]["dropout"]
    assert len(monitor.history) == 3 and len(seen) == 3, (monitor.history, seen)
    assert seen == [configured] * 3, f"training dropout after held-out checks: {seen}"


def main():
    train = np.full((40, 40), -1, dtype=np.int8)
    train.flat[:100] = 1
    train.flat[100:900] = 0
    fit, edges, labels = internal_split(train)
    reference = Training().internal_split(train)
    assert np.array_equal(fit, reference["train"])
    assert np.array_equal(edges, reference["edges"])
    assert np.array_equal(labels, reference["labels"])
    assert (labels == 1).sum() == 10 and (labels == 0).sum() == 80
    assert (fit[tuple(edges.T)] == -1).all()
    np.testing.assert_array_equal(train[tuple(edges.T)], labels)
    monitor = Monitor(dict(every=2, patience=4), np.array([[0, 0], [0, 1]]), np.array([0, 1]))
    assert not monitor.check(1, lambda e: np.array([1., 0.]))
    assert not monitor.check(2, lambda e: np.array([0., 1.]))
    assert not monitor.check(4, lambda e: np.array([0., 1.]))
    assert monitor.check(6, lambda e: np.array([0., 1.]))
    assert monitor.best() == 2
    assert RULES["reggain"]["grid"] == (50, 100, 200, 500, 1000, 2000)
    genes = np.array(["A", "B"])
    embeddings = np.arange(6, dtype=np.float32).reshape(2, 3)
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "embeddings.npz"
        np.savez(path, genes=genes, embeddings=embeddings, ensembl_ids=np.array([None, "ID"], dtype=object))
        actual = foundation(None, SimpleNamespace(embeddings=path), genes)
        np.testing.assert_array_equal(actual, embeddings)
        assert actual.dtype == embeddings.dtype
    check_scregnet_dropout()
    print("Comparator split, selection, embedding-reader and scRegNet dropout checks passed")


if __name__ == "__main__":
    main()
