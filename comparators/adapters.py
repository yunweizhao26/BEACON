import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import average_precision_score

RULES = {
    "gnnlink": dict(released=100, max=10000, every=10, patience=300),
    "genelink": dict(released=20, max=100, every=1, patience=20),
    "gclink": dict(released=20, max=100, every=1, patience=20),
    "reggain": dict(released=500, grid=(50, 100, 200, 500, 1000, 2000)),
    "scregnet": dict(released=300, max=300, every=5, patience=50),
}
MIN_VALIDATION_POSITIVES = 10
COLLAPSE_STD = 1e-4


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reggain_config(seed, epochs, device):
    return dict(epochs=epochs, lr=.001, device=str(device), repeat=1,
                seed=seed, k=50, adjacency_powers=[0, 1, 2],
                first_layer_dims=[80, 80, 10], hidden_layer_dims_list="40 40 5,16 16 2",
                pos=10, edge_alpha1=.6, edge_alpha2=.3, edge_beta1=.3, edge_beta2=.3,
                node_alpha1=.5, node_alpha2=.2, node_beta1=.2, node_beta2=.2)


def internal_split(train):
    edges = np.argwhere(train >= 0)
    labels = train[tuple(edges.T)].astype(np.int8)
    rng = np.random.default_rng(42)
    held = []
    for value in (1, 0):
        index = np.flatnonzero(labels == value)
        count = int(round(.1 * len(index)))
        held.append(np.sort(rng.choice(index, count, replace=False)) if count else np.empty(0, dtype=int))
    held = np.concatenate(held).astype(int)
    fit = train.copy()
    fit[tuple(edges[held].T)] = -1
    return fit, edges[held], labels[held]

def inputs(train, genes, expression):
    return dict(genes=genes, expression_frame=expression, expression=expression.to_numpy(dtype=np.float32),
                train_positive=np.argwhere(train == 1), train_unlabeled=np.argwhere(train == 0))


class Monitor:
    def __init__(self, rule, edges, labels):
        self.rule, self.edges, self.labels, self.history = rule, edges, labels, []

    def check(self, epoch, score):
        if self.edges is None or epoch % self.rule["every"]:
            return False
        self.history.append(dict(epoch=epoch, validation_ap=float(average_precision_score(self.labels, score(self.edges)))))
        return epoch - self.best() >= self.rule["patience"]

    def best(self):
        return max(self.history, key=lambda h: (h["validation_ap"], -h["epoch"]))["epoch"]

def gnnlink(source, data, seed, epochs, monitor=None):
    sys.path.insert(0, str(source))
    import tensorflow as tf
    from Transorfomer import TransorfomerModel
    from inits import preprocess_graph
    from utils import load_data
    seed_all(seed)
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(seed)
    n = len(data["genes"])
    positive = sp.csr_matrix((np.ones(len(data["train_positive"]), dtype=np.float32), tuple(data["train_positive"].T)), shape=(n, n)).toarray()
    negative = sp.csr_matrix((np.ones(len(data["train_unlabeled"]), dtype=np.float32), tuple(data["train_unlabeled"].T)), shape=(n, n)).toarray()
    interaction = sp.csr_matrix(positive + np.eye(n, dtype=np.float32))
    labels = positive.reshape((-1, 1))
    biases = preprocess_graph(interaction)
    feature = load_data(data["expression_frame"]).exp_data()
    model = TransorfomerModel(feature, do_train=False)
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=int(os.environ.get("SLURM_CPUS_PER_TASK", "12")), inter_op_parallelism_threads=2)
    with tf.compat.v1.Session(config=config) as session:
        session.run(tf.group(tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()))
        feed = {model.encoded_gene: feature, model.bias_in: biases, model.lbl_in: labels,
                model.msk_in: (labels > 0).astype(np.int32), model.neg_msk: negative.reshape((-1, 1)).astype(np.int32)}
        logits = lambda: session.run(model.logits, feed_dict=feed).reshape((n, n))
        score = lambda edges: logits()[edges[:, 0], edges[:, 1]]
        for epoch in range(1, epochs + 1):
            session.run(model.train_op, feed_dict=feed)
            if monitor is not None and monitor.check(epoch, score):
                break
        return logits()


def genelink(source, data, seed, epochs, device, monitor=None):
    sys.path.insert(0, str(source / "Code" if (source / "Code").is_dir() else source))
    import torch
    import torch.nn.functional as functional
    from torch.optim import Adam
    from torch.optim.lr_scheduler import StepLR
    from torch.utils.data import DataLoader
    from scGNN import GENELink
    from utils import adj2saprse_tensor, load_data, scRNADataset
    seed_all(seed)
    feature = torch.tensor(load_data(data["expression_frame"]).exp_data(), dtype=torch.float32, device=device)
    train = np.vstack([np.column_stack([data["train_positive"], np.ones(len(data["train_positive"]), dtype=np.int64)]),
                       np.column_stack([data["train_unlabeled"], np.zeros(len(data["train_unlabeled"]), dtype=np.int64)])])
    train_set = scRNADataset(train, len(data["genes"]), flag=False)
    adjacency = adj2saprse_tensor(train_set.Adj_Generate(torch.arange(len(data["genes"]), dtype=torch.int64, device=device), loop=False)).coalesce().to(device)
    model = GENELink(input_dim=feature.size(1), hidden1_dim=128, hidden2_dim=64, hidden3_dim=32, output_dim=16, num_head1=3, num_head2=3,
                     alpha=0.2, device=device, type="dot", reduction="concate").to(device)
    optimizer = Adam(model.parameters(), lr=3e-3)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
    loader = DataLoader(train_set, batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(seed))

    def score(edges):
        model.eval()
        with torch.no_grad():
            values = torch.sigmoid(model(feature, adjacency, torch.tensor(edges, dtype=torch.int64, device=device))).cpu().numpy().reshape(-1)
        model.train()
        return values.astype(float)

    for epoch in range(1, epochs + 1):
        model.train()
        for train_x, train_y in loader:
            train_x, train_y = train_x.to(device), train_y.to(device).view(-1, 1)
            optimizer.zero_grad()
            loss = functional.binary_cross_entropy(torch.sigmoid(model(feature, adjacency, train_x)), train_y)
            loss.backward()
            optimizer.step()
            scheduler.step()
        if monitor is not None and monitor.check(epoch, score):
            break
    return score


def gclink(source, data, seed, epochs, device, monitor=None):
    import importlib.util
    import torch
    import torch.nn.functional as functional
    from torch.optim import Adam
    from torch.optim.lr_scheduler import StepLR
    from torch.utils.data import DataLoader

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    model_module, utils = load("beacon_gclink_model", source / "scGNN.py"), load("beacon_gclink_utils", source / "utils.py")
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    feature = torch.tensor(utils.load_data(data["expression_frame"]).exp_data(), dtype=torch.float32, device=device)
    pairs = np.vstack([np.column_stack([data["train_positive"], np.ones(len(data["train_positive"]), dtype=np.int64)]),
                       np.column_stack([data["train_unlabeled"], np.zeros(len(data["train_unlabeled"]), dtype=np.int64)])])
    dataset = utils.scRNADataset(pairs, len(data["genes"]), flag=False)
    adjacency = utils.adj2saprse_tensor(dataset.Adj_Generate(torch.arange(len(data["genes"]), device=device), loop=False)).coalesce().to(device)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(seed))
    encoder = model_module.GENELink(input_dim=feature.size(1), hidden1_dim=128, hidden2_dim=64, hidden3_dim=32, output_dim=16, num_head1=3,
                                    num_head2=3, alpha=0.2, device=device, type="dot", reduction="concate").to(device)

    def edge_view(adj, remove=False):
        index = adj.coalesce().indices()
        if remove:
            index = index[:, torch.rand(index.size(1), device=device) >= 0.2]
        return torch.sparse_coo_tensor(index, torch.ones(index.size(1), device=device), adj.size()).coalesce()

    optimizer = Adam(encoder.parameters(), lr=3e-3)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
    for _ in range(20):
        adj1, adj2 = edge_view(adjacency), edge_view(adjacency, True)
        for edge, label in loader:
            edge, label = edge.to(device), label.to(device).view(-1, 1)
            optimizer.zero_grad()
            loss = (functional.binary_cross_entropy(torch.sigmoid(encoder(feature, adj1, edge)[-1]), label)
                    + functional.binary_cross_entropy(torch.sigmoid(encoder(feature, adj2, edge)[-1]), label))
            loss.backward()
            optimizer.step()
            scheduler.step()

    def score(edges):
        encoder.eval()
        with torch.no_grad():
            values = torch.sigmoid(encoder(feature, edge_view(adjacency, True), torch.tensor(edges, dtype=torch.int64, device=device))[-1]).cpu().numpy().reshape(-1)
        encoder.train()
        return values.astype(float)

    optimizer = Adam(encoder.parameters(), lr=3e-3)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)
    for epoch in range(1, epochs + 1):
        encoder.train()
        for edge, label in loader:
            edge, label = edge.to(device), label.to(device).view(-1, 1)
            optimizer.zero_grad()
            adj1, adj2 = edge_view(adjacency), edge_view(adjacency, True)
            z1, *_, raw1 = encoder(feature, adj1, edge)
            z2, *_, raw2 = encoder(feature, adj2, edge)
            logits = functional.normalize(z1) @ functional.normalize(z2).T / 0.2
            identity = torch.arange(logits.size(0), device=device)
            contrast = (functional.cross_entropy(logits, identity) + functional.cross_entropy(logits.T, identity)) / 2
            loss = (functional.binary_cross_entropy(torch.sigmoid(raw1), label) + functional.binary_cross_entropy(torch.sigmoid(raw2), label) + 0.5 * contrast)
            loss.backward()
            optimizer.step()
            scheduler.step()
        if monitor is not None and monitor.check(epoch, score):
            break
    return score


def reggain(source, train, genes, expression, seed, epochs, device):
    sys.path.insert(0, str(source))
    import anndata as ad
    from RegGAIN_script.run_RegGAIN import run_reggain
    upper = np.char.upper(genes.astype(str))
    adata = ad.AnnData(X=expression.to_numpy(dtype=np.float32).T, var=pd.DataFrame(index=genes.astype(str)))
    pos = np.argwhere(train == 1)
    prior = pd.DataFrame({"from": genes[pos[:, 0]], "to": genes[pos[:, 1]], "edge_type": "directed"})
    result = run_reggain(adata, prior, reggain_config(seed, epochs, device))
    values = result["GRN"].set_index(["TF", "Target"])["value"]
    assert values.index.is_unique

    def score(edges):
        out = values.reindex(pd.MultiIndex.from_arrays([upper[edges[:, 0]], upper[edges[:, 1]]])).to_numpy(dtype=float)
        assert np.isfinite(out).all()
        return out
    return score


def reggain_snapshots(source, train, genes, tfs, expression, epochs_list, device, seed=42):

    import anndata as ad
    import torch
    sys.path.insert(0, str(source))
    from RegGAIN_script import run_RegGAIN as rg
    config = reggain_config(seed, max(epochs_list), device)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dev = torch.device(device)
    positive = np.argwhere(train == 1)
    prior = pd.DataFrame({"from": genes[positive[:, 0]], "to": genes[positive[:, 1]], "edge_type": "directed"})
    adata = rg.data_preparation(ad.AnnData(X=np.asarray(expression).T, var=pd.DataFrame(index=genes)), prior)
    assert list(adata.var_names) == list(np.char.upper(genes.astype(str)))
    pyg = rg.get_PYG_data(adata, dev)
    hidden = rg.parse_hidden_layers(config["hidden_layer_dims_list"])
    make = lambda: rg.ConvModel(edge_index=pyg.edge_index, input_dim=pyg.x.size(1), adjacency_powers=config["adjacency_powers"],
                                first_layer_dim_per_power=config["first_layer_dims"], hidden_layer_dims_per_power_list=hidden).to(dev)
    encoder_out = make()
    encoder_in = make()
    model = rg.Model(encoder_out=encoder_out, encoder_in=encoder_in, num_proj_hidden=64, tau=3).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    snapshots = {}
    for epoch in range(1, max(epochs_list) + 1):
        rg._train_single_epoch(model, pyg, optimizer, config, dev)
        if epoch in epochs_list:
            snapshots[epoch] = rg._test(model, pyg, dev)
    mask = rg.generate_pos_mask(pyg, config["pos"]).cpu().numpy()
    result = {}
    for epoch, (z_in, z_out) in snapshots.items():
        norm = lambda z: (z - z.mean(axis=1, keepdims=True)) / (z.std(axis=1, keepdims=True) + 1e-8)
        grn = np.zeros((len(genes), len(genes)))
        grn += np.dot(norm(z_out), norm(z_in).T) * mask
        np.fill_diagonal(grn, 0)
        primary = np.abs(grn[tfs]).astype(np.float32)
        alternative = np.abs(norm(z_in)[tfs] @ norm(z_out).T).astype(np.float32)
        result[epoch] = (primary, alternative)
    return result


def train_scregnet(source, train, genes, tfs, frame, foundation, device, epochs, monitor=None, seed=42):
    import torch
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader
    sys.path.insert(0, str(source))
    from src.models import scTransNet_GCN
    from src.utils import scRNADataset, adj2saprse_tensor, load_data, set_seed
    config = json.loads(Path(__file__).with_name("manifest.json").read_text())["scRegNet_configuration"]
    config.update(single_gpu=0, random_seed=seed, start_seed=seed, gnn_epochs=epochs)
    namespace = argparse.Namespace(**config)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); set_seed(seed)
    edges = np.argwhere(train != -1)
    labels = train[tuple(edges.T)].astype(np.int64)
    expression = torch.from_numpy(load_data(frame).exp_data()).to(device)
    found = torch.from_numpy(foundation.astype(np.float32)).to(device)
    dataset = scRNADataset(np.column_stack([edges, labels]), len(genes), flag=True)
    adjacency = adj2saprse_tensor(dataset.Adj_Generate(tfs, loop=namespace.loop)).to(device)
    model = scTransNet_GCN(expression.shape[1], namespace, found.shape[1], device).to(device)
    optimizer = getattr(optim, namespace.optimizer_name)(model.parameters(), lr=namespace.gnn_lr, weight_decay=namespace.gnn_weight_decay)
    loader = DataLoader(dataset, batch_size=namespace.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))

    def predict(pairs):
        p = model.args.dropout
        model.eval(); model.args.dropout = 0.0; out = []
        with torch.inference_mode():
            for s in range(0, len(pairs), 500_000):
                logits = model(expression, adjacency, torch.as_tensor(pairs[s:s + 500_000], device=device), found)
                out.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        model.args.dropout = p
        model.train()
        return np.concatenate(out).astype(float)

    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        for edge_batch, label_batch in loader:
            optimizer.zero_grad()
            probability = torch.softmax(model(expression, adjacency, edge_batch.to(device), found), dim=1)
            F.binary_cross_entropy(probability, label_batch.to(device)).backward()
            optimizer.step()
        if monitor is not None:
            stop = monitor.check(epoch, predict)
            if monitor.history and monitor.history[-1]["epoch"] == epoch and monitor.best() == epoch:
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            if stop:
                break
    final_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    def scorer(state):
        def score(pairs):
            model.load_state_dict(state)
            return predict(pairs)
        return score
    return scorer(final_state), (scorer(best_state) if best_state is not None else None)


def score_all(score, pairs):
    return np.concatenate([score(pairs[s:s + 1_000_000]) for s in range(0, len(pairs), 1_000_000)])
