
import os
import re
import sys
import csv
import json
import torch
import queue
import math
import time
import shutil
import umap.umap_ as umap
import numpy as np
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import multiprocessing
import torch.optim as optim
from itertools import product
import matplotlib.pyplot as plt
import torch.nn.functional as F
import datetime
from dataclasses import dataclass, asdict, replace
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
from typing import Optional
import numpy as np
from tqdm import tqdm

from typing import Dict, List, Tuple
from sklearn.decomposition import FactorAnalysis, PCA, NMF
import matplotlib.pyplot as plt
try:
    from lightning_lite.utilities.seed import seed_everything
except ImportError:
    try:
        from pytorch_lightning.utilities.seed import seed_everything
    except ImportError:
        def seed_everything(seed):
            import random
            import numpy as np
            import torch
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.mlls import VariationalELBO
from scipy.special import expit
from pathlib import Path
from multiprocessing import Queue
import pickle

@dataclass
class NegSamplingConfig:
    weights: Dict[str, float] = None

    neg_per_pos: int = 5
    distance_topk: int = 50
    khop_k: int = 2
    node_match: str = "either"
    degree_bins: int = 4
    seed: int = 42

NEG_SAMPLING_PRESETS: Dict[str, NegSamplingConfig] = {
    "random": NegSamplingConfig(weights={"random": 1.0}, neg_per_pos=5, seed=42),
    "node_hard": NegSamplingConfig(weights={"node_either": 1.0}, neg_per_pos=5, node_match="either", seed=42),
    "node_src": NegSamplingConfig(weights={"node_src": 1.0}, neg_per_pos=5, node_match="source", seed=42),
    "node_tgt": NegSamplingConfig(weights={"node_tgt": 1.0}, neg_per_pos=5, node_match="target", seed=42),
    "degree_balanced": NegSamplingConfig(weights={"degree_balanced": 1.0}, neg_per_pos=5, degree_bins=4, seed=42),
    "distance_hard": NegSamplingConfig(weights={"distance_hard": 1.0}, neg_per_pos=5, distance_topk=50, seed=42),
    "context_only": NegSamplingConfig(weights={"context": 1.0}, neg_per_pos=5, seed=42),
    "khop1": NegSamplingConfig(weights={"khop1": 1.0}, neg_per_pos=5, khop_k=1, seed=42),
    "khop2": NegSamplingConfig(weights={"khop2": 1.0}, neg_per_pos=5, khop_k=2, seed=42),
    "gnnlink": NegSamplingConfig(weights={"node_either": 1.0}, neg_per_pos=1, node_match="either", seed=42),
    "balanced_mix": NegSamplingConfig(
        weights={"node_either": 0.4, "degree_balanced": 0.3, "random": 0.3},
        neg_per_pos=5,
        node_match="either",
        degree_bins=4,
        seed=42,
    ),
}

def _resolve_sampling_option(option: object) -> Tuple[str, Optional[str], Optional[NegSamplingConfig]]:

    tag: Optional[str]
    preset: Optional[str]
    config_obj: Optional[NegSamplingConfig]

    if isinstance(option, dict):
        tag = option.get('tag')
        preset = option.get('preset')
        config_obj = option.get('config')
    elif isinstance(option, tuple):
        if len(option) == 3:
            tag, preset, config_obj = option
        elif len(option) == 2:
            tag, preset = option
            config_obj = None
        else:
            raise ValueError(f"Unsupported sampling option tuple length: {len(option)}")
    else:
        tag = str(option)
        preset = tag if tag in NEG_SAMPLING_PRESETS else None
        config_obj = None

    if not tag:
        tag = "default"

    if preset is not None and preset not in NEG_SAMPLING_PRESETS:
        raise ValueError(f"Unknown negative sampling preset '{preset}'.")

    if config_obj is not None and not isinstance(config_obj, NegSamplingConfig):
        if isinstance(config_obj, dict):
            config_obj = NegSamplingConfig(**config_obj)
        else:
            raise TypeError("config must be a NegSamplingConfig or dict")

    return tag, preset, config_obj

def _pos_neg_arrays(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pos = np.argwhere(H == 1)
    neg = np.argwhere(H == 0)
    return pos, neg

def _normalize_weights(d: Dict[str, float]) -> Dict[str, float]:
    tot = sum(max(0.0, v) for v in d.values())
    if tot <= 0:
        return {"random": 1.0}
    return {k: max(0.0, v) / tot for k, v in d.items()}

def _row_candidates(H: np.ndarray, i: int) -> np.ndarray:
    return np.flatnonzero(H[i] == 0)

def _col_candidates(H: np.ndarray, j: int) -> np.ndarray:
    return np.flatnonzero(H[:, j] == 0)

def _node_matched_pool(H: np.ndarray, pos: np.ndarray, mode: str = "either", max_per_pos: int = 100) -> List[Tuple[int, int]]:
    rng = np.random.default_rng(0)
    pool: List[Tuple[int, int]] = []
    for (i, j) in pos:
        picks: List[Tuple[int, int]] = []
        if mode in ("source", "either"):
            cand = _row_candidates(H, i)
            if cand.size:
                jj = rng.choice(cand, size=min(max_per_pos, cand.size), replace=False)
                picks.extend([(i, int(jj_k)) for jj_k in jj if jj_k != j])
        if mode in ("target", "either"):
            cand_i = _col_candidates(H, j)
            if cand_i.size:
                ii = rng.choice(cand_i, size=min(max_per_pos, cand_i.size), replace=False)
                picks.extend([(int(ii_k), j) for ii_k in ii if ii_k != i])
        pool.extend(picks)
    return pool

def _degree_vectors(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    out_deg = (H == 1).sum(axis=1)
    in_deg = (H == 1).sum(axis=0)
    return out_deg, in_deg

def _degree_balanced_pool(H: np.ndarray, pos: np.ndarray, bins: int = 4, max_per_bin: int = 5000) -> List[Tuple[int, int]]:
    out_deg, in_deg = _degree_vectors(H)
    out_bins = np.quantile(out_deg, np.linspace(0, 1, bins + 1))
    in_bins = np.quantile(in_deg, np.linspace(0, 1, bins + 1))

    def _bin_id(x: float, edges: np.ndarray) -> int:
        idx = int(np.searchsorted(edges, x, side="right") - 1)
        return max(0, min(idx, len(edges) - 2))

    pos_bins: Dict[Tuple[int, int], int] = {}
    for (i, j) in pos:
        bi = _bin_id(out_deg[i], out_bins)
        bj = _bin_id(in_deg[j], in_bins)
        pos_bins[(bi, bj)] = pos_bins.get((bi, bj), 0) + 1

    neg_idxs = np.argwhere(H == 0)
    pool_per_bin: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for (i, j) in neg_idxs:
        bi = _bin_id(out_deg[i], out_bins)
        bj = _bin_id(in_deg[j], in_bins)
        pool_per_bin.setdefault((bi, bj), []).append((int(i), int(j)))

    rng = np.random.default_rng(0)
    pool: List[Tuple[int, int]] = []
    for key, need in pos_bins.items():
        cand = pool_per_bin.get(key, [])
        if not cand:
            continue
        take = min(max_per_bin, len(cand))
        idx = rng.choice(len(cand), size=take, replace=False)
        for k in idx:
            pool.append(cand[k])
    return pool

def _distance_hard_pool(H: np.ndarray, embeddings: np.ndarray, topk: int = 50) -> List[Tuple[int, int]]:
    X = embeddings.astype(np.float32)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)

    pool: List[Tuple[int, int]] = []
    n = H.shape[0]
    for i in range(n):
        if n <= 1:
            continue
        kth = min(topk, n - 1)
        cand = np.argpartition(-S[i], kth=kth - 1)[:kth]
        cand = cand[H[i, cand] == 0]
        pool.extend([(i, int(j)) for j in cand])
    return pool

def _khop_pool(H_prior: np.ndarray, H_target: np.ndarray, k: int = 2) -> List[Tuple[int, int]]:
    if H_prior is None:
        return []
    A = (H_prior == 1).astype(np.int8)
    Ak = A.copy()
    if k == 2:
        Ak = (A @ A > 0).astype(np.int8)
    C = (Ak > 0) & (H_target == 0)
    ii, jj = np.where(C)
    return [(int(i), int(j)) for i, j in zip(ii, jj)]

def _context_negatives_pool(H_target: np.ndarray, context_adjs: List[np.ndarray]) -> List[Tuple[int, int]]:
    if not context_adjs:
        return []
    U = np.zeros_like(H_target, dtype=np.int8)
    for A in context_adjs:
        U |= (A == 1)
    C = (U == 1) & (H_target == 0)
    ii, jj = np.where(C)
    return [(int(i), int(j)) for i, j in zip(ii, jj)]

def _random_pool(H: np.ndarray) -> List[Tuple[int, int]]:
    ii, jj = np.where(H == 0)
    return [(int(i), int(j)) for i, j in zip(ii, jj)]

def _dedup_keep_limit(items: List[Tuple[int, int]], limit: int, seed: int) -> List[Tuple[int, int]]:
    rng = np.random.default_rng(seed)
    if not items:
        return []
    arr = np.array(items, dtype=np.int32)
    arr = arr[rng.permutation(len(arr))]
    seen = set()
    out: List[Tuple[int, int]] = []
    for i, j in arr:
        if (i, j) in seen:
            continue
        seen.add((i, j))
        out.append((int(i), int(j)))
        if len(out) >= limit:
            break
    return out

def _take_unique_excluding(
    items: List[Tuple[int, int]],
    limit: int,
    seed: int,
    excluded: set,
    local_taken: set,
) -> List[Tuple[int, int]]:
    if limit <= 0 or not items:
        return []
    rng = np.random.default_rng(seed)
    arr = np.array(items, dtype=np.int32)
    arr = arr[rng.permutation(len(arr))]
    out: List[Tuple[int, int]] = []
    for i, j in arr:
        edge = (int(i), int(j))
        if edge in excluded or edge in local_taken:
            continue
        local_taken.add(edge)
        out.append(edge)
        if len(out) >= limit:
            break
    return out

def build_train_valid_with_sampling(
    H: np.ndarray,
    embeddings: Optional[np.ndarray],
    train_ratio: float,
    valid_ratio: float,
    neg_cfg: NegSamplingConfig,
    prior_adj_for_khop: Optional[np.ndarray] = None,
    context_adjs: Optional[List[np.ndarray]] = None,
    test_ratio: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    rng = np.random.default_rng(neg_cfg.seed)
    pos, _ = _pos_neg_arrays(H)

    P = pos.copy()
    rng.shuffle(P)

    ratio_vec = np.array([train_ratio, valid_ratio, max(test_ratio, 0.0)], dtype=float)
    if np.any(ratio_vec < 0):
        raise ValueError("Split ratios must be non-negative")
    total = ratio_vec.sum()
    if total <= 0:
        raise ValueError("At least one split ratio must be > 0")
    fractions = ratio_vec / total
    raw_counts = fractions * len(P)
    counts = np.floor(raw_counts).astype(int)
    remainder = len(P) - counts.sum()
    if remainder > 0:
        order = np.argsort(raw_counts - counts)[::-1]
        for idx in order[:remainder]:
            counts[idx] += 1
    for idx, (fraction, count) in enumerate(zip(fractions, counts)):
        if fraction > 0 and count == 0 and len(P) > 0:
            donor = int(np.argmax(counts))
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[idx] += 1
    n_train, n_valid, n_test = counts
    n_train = min(n_train, len(P))
    n_valid = min(n_valid, len(P) - n_train)
    n_test = len(P) - n_train - n_valid

    P_train = P[:n_train]
    P_valid = P[n_train:n_train + n_valid]
    P_test = P[n_train + n_valid:]

    weights = _normalize_weights(neg_cfg.weights or {"random": 1.0})

    static_pools: Dict[str, List[Tuple[int, int]]] = {}
    if "random" in weights:
        static_pools["random"] = _random_pool(H)
    if "distance_hard" in weights:
        if embeddings is None:
            raise ValueError("distance_hard requires embeddings")
        static_pools["distance_hard"] = _distance_hard_pool(H, embeddings, topk=neg_cfg.distance_topk)
    if "khop1" in weights:
        static_pools["khop1"] = _khop_pool(prior_adj_for_khop, H, k=1)
    if "khop2" in weights:
        static_pools["khop2"] = _khop_pool(prior_adj_for_khop, H, k=2)
    if "context" in weights:
        static_pools["context"] = _context_negatives_pool(H, context_adjs or [])

    def _make_pools(positives: np.ndarray) -> Dict[str, List[Tuple[int, int]]]:
        pools = dict(static_pools)
        if any(k.startswith("node") for k in weights):
            pools["node"] = _node_matched_pool(H, positives, mode=neg_cfg.node_match, max_per_pos=200)
        if "degree_balanced" in weights:
            pools["degree_balanced"] = _degree_balanced_pool(H, positives, bins=neg_cfg.degree_bins)
        return pools

    def _sample_from_mixture(
        pools: Dict[str, List[Tuple[int, int]]],
        target_n: int,
        seed_offset: int,
        excluded_negatives: set,
    ) -> List[Tuple[int, int]]:
        if target_n <= 0:
            return []
        chosen: List[Tuple[int, int]] = []
        chosen_set: set = set()
        parts = {k: int(round(target_n * w)) for k, w in weights.items()}
        for k, m in parts.items():
            if m <= 0:
                continue
            key = k if k in pools else ("node" if k.startswith("node") else k)
            pool = pools.get(key, [])
            if not pool:
                continue
            selected = _take_unique_excluding(
                items=pool,
                limit=m,
                seed=neg_cfg.seed + seed_offset + (hash(k) % 10_000),
                excluded=excluded_negatives,
                local_taken=chosen_set,
            )
            chosen.extend(selected)
        if len(chosen) < target_n and "random" in pools:
            short = target_n - len(chosen)
            selected = _take_unique_excluding(
                items=pools["random"],
                limit=short,
                seed=neg_cfg.seed + seed_offset + 777,
                excluded=excluded_negatives,
                local_taken=chosen_set,
            )
            chosen.extend(selected)
        return chosen[:target_n]

    def _build_split(
        positives: np.ndarray,
        num_negatives: int,
        seed_offset: int,
        excluded_negatives: set,
    ) -> Tuple[np.ndarray, set]:
        G_split = np.full_like(H, -1, dtype=np.int8)
        if len(positives):
            G_split[tuple(positives.T)] = 1
        if num_negatives > 0 and len(positives):
            pools = _make_pools(positives)
            neg_edges = _sample_from_mixture(pools, num_negatives, seed_offset, excluded_negatives)
            if neg_edges:
                coords = np.array(neg_edges).T
                G_split[tuple(coords)] = 0
                excluded_negatives.update(neg_edges)
        return G_split, excluded_negatives

    n_neg_train = int(len(P_train) * neg_cfg.neg_per_pos)
    n_neg_valid = int(len(P_valid) * neg_cfg.neg_per_pos)
    n_neg_test = int(len(P_test) * neg_cfg.neg_per_pos)

    used_negatives: set = set()
    G_train, used_negatives = _build_split(P_train, n_neg_train, 0, used_negatives)
    G_valid, used_negatives = _build_split(P_valid, n_neg_valid, 10_000, used_negatives)
    G_test, used_negatives = _build_split(P_test, n_neg_test, 20_000, used_negatives)
    return G_train, G_valid, G_test

_GLOBAL_CUSTOM: List[Dict] = []

DATASET_INFO_MAPPING = {

    1501: ("STRING", "hESC 500"),
    1502: ("STRING", "hESC 1000"),
    1503: ("STRING", "hHEP 500"),
    1504: ("STRING", "hHEP 1000"),
    1505: ("STRING", "mDC 500"),
    1506: ("STRING", "mDC 1000"),
    1507: ("STRING", "mESC 500"),
    1508: ("STRING", "mESC 1000"),
    1509: ("STRING", "mHSC-E 500"),
    1510: ("STRING", "mHSC-E 1000"),
    1511: ("STRING", "mHSC-GM 500"),
    1512: ("STRING", "mHSC-GM 1000"),
    1513: ("STRING", "mHSC-L 500"),
    1514: ("STRING", "mHSC-L 1000"),

    1601: ("Non-Specific", "hESC 500"),
    1602: ("Non-Specific", "hESC 1000"),
    1603: ("Non-Specific", "hHEP 500"),
    1604: ("Non-Specific", "hHEP 1000"),
    1605: ("Non-Specific", "mDC 500"),
    1606: ("Non-Specific", "mDC 1000"),
    1607: ("Non-Specific", "mESC 500"),
    1608: ("Non-Specific", "mESC 1000"),
    1609: ("Non-Specific", "mHSC-E 500"),
    1610: ("Non-Specific", "mHSC-E 1000"),
    1611: ("Non-Specific", "mHSC-GM 500"),
    1612: ("Non-Specific", "mHSC-GM 1000"),
    1613: ("Non-Specific", "mHSC-L 500"),
    1614: ("Non-Specific", "mHSC-L 1000"),

    1701: ("Specific", "hESC 500"),
    1702: ("Specific", "hESC 1000"),
    1703: ("Specific", "hHEP 500"),
    1704: ("Specific", "hHEP 1000"),
    1705: ("Specific", "mDC 500"),
    1706: ("Specific", "mDC 1000"),
    1707: ("Specific", "mESC 500"),
    1708: ("Specific", "mESC 1000"),
    1709: ("Specific", "mHSC-E 500"),
    1710: ("Specific", "mHSC-E 1000"),
    1711: ("Specific", "mHSC-GM 500"),
    1712: ("Specific", "mHSC-GM 1000"),
    1713: ("Specific", "mHSC-L 500"),
    1714: ("Specific", "mHSC-L 1000"),

    1801: ("Lofgof", "mESC 500"),
    1802: ("Lofgof", "mESC 1000"),
}

def resolve_processed_root() -> str:
    direct = os.path.join("data", "processed")
    if os.path.isdir(direct):
        return direct
    data_root = os.path.join("data")
    if os.path.isdir(data_root):
        for entry in sorted(os.listdir(data_root)):
            candidate = os.path.join(data_root, entry, "processed")
            if os.path.isdir(candidate):
                return candidate
    return direct

def get_datasets():
    datasets = [
        {
            'dataset_id': 1000,
            'dataset_name': 'mDC',
            'expression_file': 'data/raws/mDC-ExpressionData.csv',
            'network_file': 'data/raws/mDC-network.csv',
        },
        {
            'dataset_id': 1001,
            'dataset_name': 'mESC',
            'expression_file': 'data/raws/mESC-ExpressionData.csv',
            'network_file': 'data/raws/mESC-network.csv',
        },
        {
            'dataset_id': 1002,
            'dataset_name': 'mHSC-E',
            'expression_file': 'data/raws/mHSC-E-ExpressionData.csv',
            'network_file': 'data/raws/mHSC-E-network.csv',
        },
        {
            'dataset_id': 1003,
            'dataset_name': 'mHSC-GM',
            'expression_file': 'data/raws/mHSC-GM-ExpressionData.csv',
            'network_file': 'data/raws/mHSC-GM-network.csv',
        },
        {
            'dataset_id': 1004,
            'dataset_name': 'mHSC-L',
            'expression_file': 'data/raws/mHSC-L-ExpressionData.csv',
            'network_file': 'data/raws/mHSC-L-network.csv',
        },
        {
            'dataset_id': 1005,
            'dataset_name': 'hESC',
            'expression_file': 'data/raws/hESC-ExpressionData.csv',
            'network_file': 'data/raws/hESC-network.csv',
        },
        {
            'dataset_id': 1006,
            'dataset_name': 'hHep',
            'expression_file': 'data/raws/hHep-ExpressionData.csv',
            'network_file': 'data/raws/hHep-network.csv',
        },
        {
            'dataset_id': 1007,
            'dataset_name': 'mouse',
            'expression_file': 'data/raws/mouse-ExpressionData.csv',
            'network_file': 'data/raws/mouse-network.csv',
        },
        {
            'dataset_id': 1008,
            'dataset_name': 'human',
            'expression_file': 'data/raws/human-ExpressionData.csv',
            'network_file': 'data/raws/human-network.csv',
        },
    ]
    benchmark_datasets = []
    processed_root = resolve_processed_root()
    for ds_id, (net_type, cell_type_str) in DATASET_INFO_MAPPING.items():
        base_dir = os.path.join(processed_root, net_type, cell_type_str)
        expression_file = os.path.join(base_dir, "BL--ExpressionData.csv")
        network_file = os.path.join(base_dir, "BL--network.csv")

        dataset_name = f"{net_type}_{cell_type_str.replace(' ', '_')}"

        benchmark_datasets.append({
            'dataset_id': ds_id,
            'dataset_name': dataset_name,
            'expression_file': expression_file,
            'network_file': network_file,
        })
    datasets.extend(benchmark_datasets)
    datasets.extend(_GLOBAL_CUSTOM)
    return datasets

def load_network_data(file_path, gene_list):

    gene_to_index = {gene: idx for idx, gene in enumerate(gene_list)}
    num_genes = len(gene_list)
    H = np.zeros((num_genes, num_genes))
    import csv
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            source_gene, target_gene = line
            if source_gene in gene_to_index and target_gene in gene_to_index:
                source_idx = gene_to_index[source_gene]
                target_idx = gene_to_index[target_gene]
                H[source_idx, target_idx] = 1
    return H

def load_data(dataset_info):
    dataset_id = dataset_info['dataset_id']
    if 'h5ad_file' in dataset_info:
        import anndata, scipy.sparse as sp
        adata = anndata.read_h5ad(dataset_info['h5ad_file'])

        mask = (
            (adata.obs["disease"] == dataset_info['disease']) &
            (adata.obs["tissue"]  == dataset_info['tissue'])
        )
        adata_sample = adata[mask]

        X = adata_sample.X
        if sp.issparse(X):
            X = X.toarray()
        ds_noisy = X.T.astype(np.float32)
        ds_clean = None
        gene_names = adata_sample.var.feature_name.tolist()
        cell_names = adata_sample.obs_names.tolist()

        print(f"Number of genes: {len(gene_names)}, Number of cells: {len(cell_names)}")

        with open(dataset_info['network_file'], 'r') as f:
            reader = csv.reader(f)
            next(reader)
            network_genes = set()
            for line in reader:
                source_gene, target_gene = line
                network_genes.add(source_gene)
                network_genes.add(target_gene)
        tt_missing = 0
        for gene in network_genes:
            print(f"total genes in network: {len(network_genes)}")
            if gene not in gene_names:
                print(f"Gene {gene} not found in the dataset.")
                tt_missing += 1

        with open("results/cl/gene_names.txt", "w") as f:
            for gene in gene_names:
                f.write(f"{gene}\n")
        print(f"Total missing genes: {tt_missing}")
    else:
        expression_file = dataset_info['expression_file']
        if not os.path.exists(expression_file):
            print(f"Expression file not found for Dataset {dataset_id}. Skipping.")
            return None, None, None, None
        import pandas as pd
        df = pd.read_csv(expression_file, index_col=0)
        ds_noisy = df.values.astype(np.float32)

        ds_clean = None
        gene_names = df.index.tolist()
        cell_names = df.columns.tolist()

    return ds_clean, ds_noisy, gene_names, cell_names

def load_transformer_data_for_contrastive(exp_file, train_file, test_file, val_file,
                                          embedding_type='fa', n_components=64):

    data_input = pd.read_csv(exp_file, index_col=0)
    gene_names = data_input.index
    features = data_input.values
    num_genes = features.shape[0]

    train_data = pd.read_csv(train_file, index_col=0).values
    valid_data = pd.read_csv(val_file,   index_col=0).values
    test_data  = pd.read_csv(test_file,  index_col=0).values

    train_data = train_data[np.lexsort(-train_data.T)]
    valid_data = valid_data[np.lexsort(-valid_data.T)]
    test_data  = test_data[np.lexsort(-test_data.T)]

    train_pos_idx = np.sum(train_data[:, 2])
    val_pos_idx   = np.sum(valid_data[:, 2])
    test_pos_idx  = np.sum(test_data[:, 2])

    G_train = create_adjacency_matrix(train_data, train_pos_idx, num_genes)
    G_valid = create_adjacency_matrix(valid_data, val_pos_idx, num_genes)
    G_test  = create_adjacency_matrix(test_data,  test_pos_idx, num_genes)

    ds_scaled = StandardScaler().fit_transform(features)
    embeddings = generate_embeddings(data_input=ds_scaled,
                                    embedding_type=embedding_type,
                                    n_components=n_components)
    return {
        'features': embeddings,
        'gene_names': gene_names,
        'G_train': G_train,
        'G_valid': G_valid,
        'G_test': G_test,
        'metrics': {
            'train_pos': train_pos_idx,
            'train_total': len(train_data),
            'valid_pos': val_pos_idx,
            'valid_total': len(valid_data),
            'test_pos': test_pos_idx,
            'test_total': len(test_data)
        }
    }

def create_adjacency_matrix(data, pos_idx, num_genes):
    adj = -np.ones((num_genes, num_genes), dtype=np.int8)

    for i in range(pos_idx):
        source, target = int(data[i,0]), int(data[i,1])
        adj[source, target] = 1

    for i in range(pos_idx, len(data)):
        source, target = int(data[i,0]), int(data[i,1])
        adj[source, target] = 0
    return adj

def generate_embeddings(data, embedding_type="PCA", n_components=64, scale=False, visualize=False):
    X = data
    try:
        import scanpy as sc
        if isinstance(X, sc.AnnData):
            if embedding_type != "PAGA":
                X = X.X
    except ImportError:
        pass

    X = np.array(X) if not isinstance(X, np.ndarray) else X

    if scale:
        X = StandardScaler().fit_transform(X)

    embedding = None
    embedding_type = embedding_type.lower()
    if embedding_type == "pca":
        model = PCA(n_components=n_components, random_state=0)
        embedding = model.fit_transform(X)
    elif embedding_type == "fa" or embedding_type == "factoranalysis":

        model = FactorAnalysis(n_components=n_components, random_state=0)
        embedding = model.fit_transform(X)
    elif embedding_type == "phate":
        try:
            import phate
        except ImportError:
            raise ImportError("phate library is not installed. Install via `pip install phate` to use this method.")

        phate_op = phate.PHATE(n_components=n_components, random_state=0)
        embedding = phate_op.fit_transform(X)
    elif embedding_type == "raw":
        embedding = X
    else:
        raise ValueError(f"Unknown method '{embedding_type}'. Supported methods are: PHATE, DiffusionMaps, PAGA, NMF, PCA, FA, raw.")

    embedding = np.array(embedding)
    return embedding

def load_ground_truth_grn(file_path, num_genes=None, gene_names=None):
    if gene_names is None:

        if num_genes is None:
            with open(file_path, 'r') as f:
                indices = []
                for line in f:
                    source, target = map(int, line.strip().split(','))
                    indices.extend([source, target])
            num_genes = max(indices) + 1
        H = np.zeros((num_genes, num_genes))
        with open(file_path, 'r') as f:
            for line in f:
                source, target = map(int, line.strip().split(','))
                H[source, target] = 1
    else:

        num_genes = len(gene_names)
        gene_to_idx = {gene.lower(): idx for idx, gene in enumerate(gene_names)}
        H = np.zeros((num_genes, num_genes))
        with open(file_path, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            for line in reader:

                source_gene, target_gene = line[0].lower(), line[1].lower()
                if source_gene in gene_to_idx and target_gene in gene_to_idx:
                    source_idx = gene_to_idx[source_gene]
                    target_idx = gene_to_idx[target_gene]
                    H[source_idx, target_idx] = 1

    total_negatives = np.sum(H == 0)
    total_positives = np.sum(H == 1)

    total_negatives = np.sum(H == 0)
    print("Changed: Total negatives: ", total_negatives, "Total positives: ", total_positives)
    return H

def load_context_adjs_for_same_celltype(dataset_id: int, gene_names: Optional[List[str]]) -> List[np.ndarray]:

    if gene_names is None or dataset_id not in DATASET_INFO_MAPPING:
        return []

    target_net_type, cell_type_str = DATASET_INFO_MAPPING[dataset_id]
    context_mats: List[np.ndarray] = []
    processed_root = resolve_processed_root()
    for ds_id, (net_type, cell_str) in DATASET_INFO_MAPPING.items():
        if cell_str != cell_type_str or net_type == target_net_type:
            continue
        base_dir = os.path.join(processed_root, net_type, cell_str)
        net_file = os.path.join(base_dir, "BL--network.csv")
        if os.path.exists(net_file):
            try:
                context_mats.append(load_ground_truth_grn(net_file, gene_names=gene_names))
            except Exception as exc:
                print(f"Failed to load context adjacency for {net_type} {cell_str}: {exc}")
    return context_mats

def sample_partial_grn(H, sample_ratio=8/10):
    print("sample_partial_grn")
    positive_edges = np.argwhere(H == 1)
    negative_edges = np.argwhere(H == 0)
    total_positives = len(positive_edges)
    total_negatives = len(negative_edges)
    print("positive_edges: ", total_positives, "negative_edges: ", total_negatives)
    num_pos_sample = int(total_positives * sample_ratio)
    num_neg_sample = int(total_negatives * sample_ratio)
    print("num_pos_sample: ", num_pos_sample, "num_neg_sample: ", num_neg_sample)
    np.random.shuffle(positive_edges)
    np.random.shuffle(negative_edges)
    sampled_pos = positive_edges[:num_pos_sample]
    sampled_neg = negative_edges[:num_neg_sample]
    print(4)

    G = np.full(H.shape, -1, dtype=np.int8)
    print(5)
    G[tuple(zip(*sampled_pos))] = 1
    G[tuple(zip(*sampled_neg))] = 0
    print(6)
    return G

def split_train_valid(G, train_ratio=7/8):
    print("a")
    positive_edges = np.argwhere(G == 1)
    negative_edges = np.argwhere(G == 0)
    num_pos_train = int(len(positive_edges) * train_ratio)
    num_neg_train = int(len(negative_edges) * train_ratio)
    print("num_pos_train: ", num_pos_train, "num_neg_train: ", num_neg_train)
    np.random.shuffle(positive_edges)
    np.random.shuffle(negative_edges)

    print("b")
    train_pos = positive_edges[:num_pos_train]
    valid_pos = positive_edges[num_pos_train:]
    train_neg = negative_edges[:num_neg_train]
    valid_neg = negative_edges[num_neg_train:]
    print("num_pos_train: ", len(train_pos), "num_neg_train: ", len(train_neg), "num_pos_valid: ", len(valid_pos), "num_neg_valid: ", len(valid_neg))
    print("c")
    G_train = -np.ones_like(G)
    G_valid = -np.ones_like(G)
    print("d")
    G_train[tuple(zip(*train_pos))] = 1
    G_train[tuple(zip(*train_neg))] = 0
    G_valid[tuple(zip(*valid_pos))] = 1
    G_valid[tuple(zip(*valid_neg))] = 0
    return G_train, G_valid

def generate_balanced_batches(embeddings, adjacency_matrix, batch_size, num_batches):
    num_nodes = embeddings.shape[0]
    positive_edges = np.argwhere(adjacency_matrix == 1)
    negative_edges = np.argwhere(adjacency_matrix == 0)

    np.random.shuffle(positive_edges)
    np.random.shuffle(negative_edges)

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(positive_edges))

        batch_positive = positive_edges[start_idx:end_idx]
        batch_negative = negative_edges[np.random.choice(len(negative_edges), size=len(batch_positive), replace=False)]

        batch_edges = np.concatenate([batch_positive, batch_negative])
        batch_labels = np.concatenate([np.ones(len(batch_positive)), np.zeros(len(batch_negative))])

        shuffle_idx = np.random.permutation(len(batch_edges))
        batch_edges = batch_edges[shuffle_idx]
        batch_labels = batch_labels[shuffle_idx]

        x1 = torch.tensor(embeddings[batch_edges[:, 0]], dtype=torch.float32).to(device)
        x2 = torch.tensor(embeddings[batch_edges[:, 1]], dtype=torch.float32).to(device)
        labels = torch.tensor(batch_labels, dtype=torch.float32).to(device)
        yield x1, x2, labels

def generate_batches(embeddings, adjacency_matrix, batch_size, num_batches, negative_ratio=10):
    positive_edges = np.argwhere(adjacency_matrix == 1)
    negative_edges = np.argwhere(adjacency_matrix == 0)
    np.random.shuffle(positive_edges)
    np.random.shuffle(negative_edges)
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(positive_edges))
        batch_positive = positive_edges[start_idx:end_idx]
        num_negatives = len(batch_positive) * negative_ratio
        if num_negatives > len(negative_edges):
            batch_negative = negative_edges[np.random.choice(len(negative_edges), size=num_negatives, replace=True)]
        else:
            batch_negative = negative_edges[np.random.choice(len(negative_edges), size=num_negatives, replace=False)]
        batch_edges = np.concatenate([batch_positive, batch_negative])
        batch_labels = np.concatenate([np.ones(len(batch_positive)), np.zeros(len(batch_negative))])

        shuffle_idx = np.random.permutation(len(batch_edges))
        batch_edges = batch_edges[shuffle_idx]
        batch_labels = batch_labels[shuffle_idx]
        x1 = torch.tensor(embeddings[batch_edges[:, 0]], dtype=torch.float32)
        x2 = torch.tensor(embeddings[batch_edges[:, 1]], dtype=torch.float32)
        labels = torch.tensor(batch_labels, dtype=torch.float32)
        yield x1, x2, labels

class DirectionalContrastiveModel(nn.Module):
    def __init__(self, input_dim, proj_dim):
        super().__init__()
        self.projection_source = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim)
        )
        self.projection_target = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, proj_dim)
        )

    def forward(self, x1, x2):
        return self.projection_source(x1), self.projection_target(x2)

    def get_embeddings(self, x, combine_mode="avg"):

        with torch.no_grad():
            src_emb = self.projection_source(x)
            tgt_emb = self.projection_target(x)
            if combine_mode == "avg":
                return 0.5*(src_emb + tgt_emb)
            elif combine_mode == "cat":
                return torch.cat([src_emb, tgt_emb], dim=-1)
            else:
                return src_emb

class SoftNearestNeighborLoss(nn.Module):
    def __init__(self, temperature=10., cos_distance=True):
        super(SoftNearestNeighborLoss, self).__init__()
        self.temperature = temperature
        self.cos_distance = cos_distance

    def pairwise_cos_distance(self, A, B):
        query_embeddings = F.normalize(A, dim=1)
        key_embeddings = F.normalize(B, dim=1)
        distances = 1 - torch.matmul(query_embeddings, key_embeddings.T)
        return distances

    def forward(self, embeddings, labels):
        batch_size = embeddings.shape[0]
        eps = 1e-9

        if self.cos_distance:
            pairwise_dist = self.pairwise_cos_distance(embeddings, embeddings)
        else:
            pairwise_dist = torch.cdist(embeddings, embeddings, p=2)

        pairwise_dist = pairwise_dist / self.temperature
        negexpd = torch.exp(-pairwise_dist)

        pairs_y = torch.broadcast_to(labels, (batch_size, batch_size))
        mask = pairs_y == torch.transpose(pairs_y, 0, 1)
        mask = mask.float()

        device = embeddings.device
        ones = torch.ones([batch_size, batch_size], dtype=torch.float32, device=device)
        dmask = ones - torch.eye(batch_size, dtype=torch.float32, device=device)

        alcn = torch.sum(torch.multiply(negexpd, dmask), dim=1)

        sacn = torch.sum(torch.multiply(negexpd, mask), dim=1)

        loss = -torch.log((sacn+eps)/alcn).mean()
        return loss

def train_snn_directional(embeddings, adjacency_matrix, input_dim, projection_dim, num_epochs, batch_size, learning_rate, negative_ratio, temperature, device):
    model = DirectionalContrastiveModel(input_dim, projection_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = SoftNearestNeighborLoss(temperature=temperature)
    num_positive_edges = (adjacency_matrix == 1).sum().item()
    num_batches = num_positive_edges // batch_size
    print(f"num_positive_edges: {num_positive_edges}, num_batches: {num_batches}")

    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for x1, x2, labels in generate_batches(embeddings, adjacency_matrix, batch_size, num_batches, negative_ratio=negative_ratio):
            x1, x2, labels = x1.to(device), x2.to(device), labels.to(device)
            optimizer.zero_grad()
            proj1, proj2 = model(x1, x2)
            embeddings_batch = torch.cat([proj1, proj2], dim=0)
            labels_batch = torch.cat([labels, labels], dim=0)
            loss = criterion(embeddings_batch, labels_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    print("Training complete.")
    return model

def get_k_metrics(true_labels, predicted_scores):
    k = int(np.sum(true_labels))
    sorted_indices = np.argsort(predicted_scores)[::-1]
    sorted_labels = true_labels[sorted_indices]
    precision_k = np.sum(sorted_labels[:k]) / k
    recall_k = np.sum(sorted_labels[:k]) / np.sum(true_labels)
    return precision_k, recall_k

class GPClassificationModel(ApproximateGP):
    def __init__(self, inducing_points, model_type='standard', direction_weight=1.0):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, inducing_points,
            variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        self.model_type = model_type
        self.mean_module = gpytorch.means.ConstantMean()
        self.stab_noise = 1e-2

        if model_type == 'directional':
            self.covar_module = gpytorch.kernels.ScaleKernel(
                DirectionalRBFKernel(
                    src_weight=2.0,
                    tgt_weight=1.0,
                    dir_weight=5.0)
            )
            self.direction_weight = direction_weight
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        covar_x = covar_x.add_jitter(self.stab_noise)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class DirectionalRBFKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True

    def __init__(self, src_weight=1.0, tgt_weight=1.0, dir_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.src_weight = nn.Parameter(torch.tensor(src_weight))
        self.tgt_weight = nn.Parameter(torch.tensor(tgt_weight))
        self.dir_weight = nn.Parameter(torch.tensor(dir_weight))

    def forward(self, x1, x2, diag=False, **params):
        if diag:

            return torch.ones(x1.size(0), dtype=x1.dtype, device=x1.device)

        D = (x1.shape[-1] - 1) // 2
        dir1, dir2 = x1[..., -1], x2[..., -1]

        src1, tgt1 = x1[..., :D], x1[..., D:-1]
        src2, tgt2 = x2[..., :D], x2[..., D:-1]

        dist_src = self.src_weight * (src1.unsqueeze(-2) - src2.unsqueeze(-3)).pow(2).sum(-1)
        dist_tgt = self.tgt_weight * (tgt1.unsqueeze(-2) - tgt2.unsqueeze(-3)).pow(2).sum(-1)
        dist_dir = self.dir_weight * (dir1.unsqueeze(-1) - dir2.unsqueeze(-2)).pow(2)

        return torch.exp(-0.5 * (dist_src + dist_tgt + dist_dir) / self.lengthscale.pow(2))

def train_gp_model(projected_embeddings, adjacency_matrix, device,
                  model_type='standard', direction_weight=1.0,
                  inducing_points_num=500, num_epochs=100,
                  batch_size=1024, run_seed=42):
    torch.manual_seed(run_seed)
    np.random.seed(run_seed)

    print(1)

    train_edges = np.argwhere((adjacency_matrix == 1) | (adjacency_matrix == 0))
    print(2)
    train_labels = adjacency_matrix[train_edges[:, 0], train_edges[:, 1]].astype(int)
    print(3)
    emb_i = torch.tensor(projected_embeddings[train_edges[:, 0]], dtype=torch.float32)
    emb_j = torch.tensor(projected_embeddings[train_edges[:, 1]], dtype=torch.float32)
    print(4)
    if model_type == 'directional':
        direction = torch.ones(len(train_edges), 1, dtype=torch.float32)
        print("Direction tensor:", direction, direction.type())
        print("emb_i tensor:", emb_i, emb_i.type())
        print("emb_j tensor:", emb_j, emb_j.type())
        print(6)
        X_train = torch.cat([emb_i, emb_j, direction], dim=1)
    else:
        X_train = torch.cat([emb_i, emb_j], dim=1)
    print(7)
    X_train = X_train.to(device)
    print(8)
    y_train = torch.from_numpy(train_labels).float().to(device)

    inducing_points = X_train[:min(inducing_points_num, X_train.shape[0])]
    inducing_points = inducing_points + 1e-5 * torch.randn_like(inducing_points)

    model = GPClassificationModel(
        inducing_points=inducing_points.to(device),
        model_type=model_type,
        direction_weight=direction_weight
    ).to(device)

    likelihood = BernoulliLikelihood().to(device)

    num_data = y_train.size(0)
    mll = VariationalELBO(likelihood, model, num_data=num_data)
    optimizer = torch.optim.AdamW([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.001)

    model.train()
    likelihood.train()
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    jitter_ctx = gpytorch.settings.cholesky_jitter(1e-1)
    for epoch in range(num_epochs):
        epoch_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            with jitter_ctx:
                output = model(x_batch)
                loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

    return model, likelihood, X_train, y_train

def evaluate_bayesian_model_gp(model, likelihood, projected_embeddings,
                              adjacency_matrix, H, device,
                              create_visualizations: bool = True, output_dir: str = "./results/p_value"):
    print("="*60)
    print("STARTING GP MODEL EVALUATION")
    print("="*60)

    num_genes = projected_embeddings.shape[0]

    print("\n1. Preparing test data...")
    test_edges = np.argwhere(adjacency_matrix != -1)
    true_labels = H[test_edges[:, 0], test_edges[:, 1]].astype(int)

    valid_mask = (true_labels >= 0) & (true_labels <= 1)
    print(f"Ground truth labels - 1s: {(true_labels == 1).sum()}, 0s: {(true_labels == 0).sum()}, -2s (ignored): {(true_labels == -2).sum()}")
    test_edges = test_edges[valid_mask]
    true_labels = true_labels[valid_mask]
    print(f"After filtering ignored edges: {len(true_labels)} edges remaining")

    emb_i_test = torch.tensor(projected_embeddings[test_edges[:, 0]], dtype=torch.float32)
    emb_j_test = torch.tensor(projected_embeddings[test_edges[:, 1]], dtype=torch.float32)
    if model.model_type == 'directional':
        direction = torch.ones(len(test_edges), 1, dtype=torch.float32)
        X_test = torch.cat([emb_i_test, emb_j_test, direction], dim=1)
    else:
        X_test = torch.cat([emb_i_test, emb_j_test], dim=1)

    X_test = X_test.to(device)
    y_test = torch.from_numpy(true_labels).float().to(device)

    print("\n2. Running GP model predictions...")
    model.eval()
    likelihood.eval()
    jitter_ctx = gpytorch.settings.cholesky_jitter(1e-1)
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), jitter_ctx:
        test_dist = model(X_test)
        test_preds = likelihood(test_dist)
        predicted_probs = test_preds.mean.cpu().numpy()
        y_test_np = y_test.cpu().numpy()

    print(f"Evaluation set: {len(y_test_np):,} edges, {y_test_np.sum()} positives")
    print("unique values in y_test: ", np.unique(y_test_np))

    print("\n3. Computing ranking metrics...")

    order = np.argsort(predicted_probs)[::-1].astype(np.int32)
    ranks = np.empty(len(order), dtype=np.int32)
    ranks[order] = np.arange(1, len(order) + 1)

    pos_idx = np.where(y_test_np == 1)[0]
    pos_ranks = [(int(idx), int(ranks[idx]),
                  1.0 - (ranks[idx] - 1) / len(order))
                 for idx in pos_idx]

    mean_pos_percentile = (sum(p for _, _, p in pos_ranks) /
                          max(1, len(pos_ranks))) if pos_ranks else 0.0

    detailed_rankings_path = os.path.join(output_dir, 'gp_detailed_rankings.json')
    os.makedirs(output_dir, exist_ok=True)
    detailed_data = {
        'positive_edge_rankings': pos_ranks,
        'all_predictions': predicted_probs.tolist(),
        'all_true_labels': y_test_np.tolist(),
        'edge_indices': test_edges.tolist()
    }
    with open(detailed_rankings_path, 'w') as f:
        json.dump(detailed_data, f, indent=2)

    print(f"Found {len(pos_ranks)} positive edges (detailed rankings saved to {detailed_rankings_path})")
    print(f"Mean percentile of positive edges: {mean_pos_percentile:.4f}")

    TOP_LIST = [10, 50, 100, 500, 1000]
    topN_summary = {}
    cum_pos = np.cumsum(y_test_np[order])
    total_pos = y_test_np.sum()
    prevalence = total_pos / len(y_test_np) if len(y_test_np) > 0 else 0

    print("\nTop-N Analysis:")
    print("-" * 60)

    for N in TOP_LIST:
        if N <= len(y_test_np):
            tp = int(cum_pos[N-1])
            prec = tp / N
            rec = tp / total_pos if total_pos else 0.0
            lift = prec / prevalence if prevalence > 0 else 0.0
            topN_summary[N] = {
                'precision': prec,
                'recall': rec,
                'lift': lift,
                'tp': tp
            }
            print(f"Top-{N}: TP={tp}, Precision={prec:.4f}, Recall={rec:.4f}, Lift={lift:.2f}")

    print("\n4. Computing standard metrics...")

    if len(np.unique(y_test_np)) < 2:
        auc_roc = 0.5
    else:
        auc_roc = roc_auc_score(y_test_np, predicted_probs)

    precision, recall, thresholds = precision_recall_curve(y_test_np, predicted_probs)
    auc_pr = auc(recall, precision)

    metrics = {

        'true_labels': y_test_np,
        'probabilities': predicted_probs,

        'num_positive': int(y_test_np.sum()),
        'num_negative': int(len(y_test_np) - y_test_np.sum()),

        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'pos_ranks_summary': {
            'count': len(pos_ranks),
            'mean_rank': sum(r for _, r, _ in pos_ranks) / max(1, len(pos_ranks)) if pos_ranks else 0,
            'best_rank': min((r for _, r, _ in pos_ranks), default=0),
            'worst_rank': max((r for _, r, _ in pos_ranks), default=0),
            'detailed_file': detailed_rankings_path
        },
        'mean_pos_percentile': mean_pos_percentile,
        'topN_summary': topN_summary,

        'precision_curve': precision,
        'recall_curve': recall,
        'thresholds': thresholds,

        'model_type': 'GP',
        'downsampled': False,
        'downsample_rate': 1.0
    }

    print(f"\nAUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR: {auc_pr:.4f}")

    if create_visualizations:
        print("\n5. Creating visualizations and reports...")

        os.makedirs(output_dir, exist_ok=True)

        rank_plot_path = os.path.join(output_dir, "gp_pos_rank_plot.png")
        _plot_rank_positions(predicted_probs[order],
                           y_test_np[order],
                           rank_plot_path)
        metrics['pos_rank_plot'] = rank_plot_path
        print(f"✓ Created rank position plot: {rank_plot_path}")

        dist_plot_path = os.path.join(output_dir, "gp_score_distributions.png")
        plot_score_distributions(metrics, dist_plot_path)
        metrics['score_distributions_plot'] = dist_plot_path
        print(f"✓ Created score distribution plots: {dist_plot_path}")

        emb_dist_path = os.path.join(output_dir, "gp_prediction_analysis.png")
        plot_gp_prediction_analysis(test_edges, y_test_np, predicted_probs, projected_embeddings, emb_dist_path)
        metrics['prediction_analysis_plot'] = emb_dist_path
        print(f"✓ Created GP prediction analysis plot: {emb_dist_path}")

        report_path = os.path.join(output_dir, "gp_evaluation_summary.txt")
        create_gp_summary_report(metrics, report_path)
        metrics['summary_report'] = report_path
        print(f"✓ Created GP summary report: {report_path}")

    print("\n" + "="*60)
    print("GP EVALUATION COMPLETE - SUMMARY")
    print("="*60)
    print(f"Total test edges evaluated: {len(y_test_np):,}")
    print(f"Positive edges: {metrics['num_positive']} ({100*metrics['num_positive']/len(y_test_np):.4f}%)")
    print(f"Negative edges: {metrics['num_negative']} ({100*metrics['num_negative']/len(y_test_np):.4f}%)")

    print("\nKEY METRICS:")
    print(f"• Mean positive edge percentile: {metrics['mean_pos_percentile']:.4f}")
    print(f"• AUC-ROC: {metrics['auc_roc']:.4f}")
    print(f"• AUC-PR: {metrics['auc_pr']:.4f}")

    if metrics['mean_pos_percentile'] >= 0.95:
        assessment = "EXCELLENT - Positive edges are in top 5%"
    elif metrics['mean_pos_percentile'] >= 0.90:
        assessment = "VERY GOOD - Positive edges are in top 10%"
    elif metrics['mean_pos_percentile'] >= 0.75:
        assessment = "GOOD - Positive edges are in top 25%"
    elif metrics['mean_pos_percentile'] >= 0.50:
        assessment = "MODERATE - Positive edges are in top 50%"
    else:
        assessment = "POOR - Positive edges are below median"

    print(f"\nOVERALL ASSESSMENT: {assessment}")

    if create_visualizations:
        print(f"\nAll GP evaluation results saved to: {output_dir}/")

    print("="*60 + "\n")

    return metrics

def calculate_auc_extremely_large(true_labels, scores, batch_size=10000, max_points=100000):

    total_size = len(true_labels)
    print(f"Total dataset size: {total_size} points")

    pos_count = 0
    neg_count = 0

    for start_idx in range(0, total_size, batch_size):
        end_idx = min(start_idx + batch_size, total_size)
        batch_labels = true_labels[start_idx:end_idx]

        binary_batch = np.array(batch_labels, dtype=np.int8)
        binary_batch = (binary_batch > 0).astype(np.int8)

        pos_count += np.sum(binary_batch == 1)
        neg_count += np.sum(binary_batch == 0)

    print(f"Found {pos_count} positive examples and {neg_count} negative examples")

    if pos_count == 0 or neg_count == 0:
        print("Warning: Only one class present, returning AUC of 0.5")
        return 0.5

    if total_size > max_points:
        print(f"Sampling {max_points} points for AUC calculation")

        pos_target = max_points // 2
        neg_target = max_points - pos_target

        pos_rate = min(1.0, pos_target / pos_count)
        neg_rate = min(1.0, neg_target / neg_count)

        print(f"Sampling rates: positive={pos_rate:.4f}, negative={neg_rate:.4f}")

        sampled_indices = []

        for start_idx in range(0, total_size, batch_size):
            end_idx = min(start_idx + batch_size, total_size)
            batch_indices = np.arange(start_idx, end_idx)
            batch_labels = true_labels[batch_indices]

            binary_batch = np.array(batch_labels, dtype=np.int8)
            binary_batch = (binary_batch > 0).astype(np.int8)

            pos_indices = batch_indices[binary_batch == 1]
            if len(pos_indices) > 0 and pos_rate < 1.0:
                sampled_pos = np.random.choice(
                    pos_indices,
                    max(1, int(len(pos_indices) * pos_rate)),
                    replace=False
                )
                sampled_indices.extend(sampled_pos)
            else:
                sampled_indices.extend(pos_indices)

            neg_indices = batch_indices[binary_batch == 0]
            if len(neg_indices) > 0 and neg_rate < 1.0:
                sampled_neg = np.random.choice(
                    neg_indices,
                    max(1, int(len(neg_indices) * neg_rate)),
                    replace=False
                )
                sampled_indices.extend(sampled_neg)
            else:
                sampled_indices.extend(neg_indices)

        sampled_indices = np.array(sampled_indices)
        if len(sampled_indices) > max_points:
            sampled_indices = np.random.choice(sampled_indices, max_points, replace=False)

        sampled_labels = true_labels[sampled_indices]
        sampled_scores = scores[sampled_indices]

        binary_labels = (sampled_labels > 0).astype(np.int8)

        print(f"Final sample: {len(binary_labels)} points, {np.sum(binary_labels)} positives")

        try:
            auc = roc_auc_score(binary_labels, sampled_scores)
            print(f"AUC calculation successful: {auc:.4f}")
            return auc
        except Exception as e:
            print(f"AUC calculation failed: {e}")
            return 0.5
    else:

        binary_labels = (true_labels > 0).astype(np.int8)
        try:
            return roc_auc_score(binary_labels, scores)
        except Exception as e:
            print(f"AUC calculation failed: {e}")
            return 0.5

def _plot_rank_positions(sorted_scores, sorted_labels, save_path, max_points=2_000_000):
    n = len(sorted_scores)
    if n > max_points:

        pos_mask = sorted_labels == 1
        neg_mask = ~pos_mask

        keep_neg = np.random.choice(np.where(neg_mask)[0],
                                    size=max_points - pos_mask.sum(),
                                    replace=False)
        keep_idx = np.sort(np.concatenate([np.where(pos_mask)[0], keep_neg]))
        scores = sorted_scores[keep_idx]
        labels = sorted_labels[keep_idx]
        ranks = keep_idx + 1
    else:
        scores = sorted_scores
        labels = sorted_labels
        ranks = np.arange(1, n+1)

    plt.figure(figsize=(10, 6))
    plt.scatter(ranks[labels == 0], scores[labels == 0],
                s=2, alpha=.3, label="negatives", color='gray')
    plt.scatter(ranks[labels == 1], scores[labels == 1],
                s=50, color="red", label="positives", zorder=10)
    plt.xscale('log')
    plt.xlabel("Rank (1 = highest score)")
    plt.ylabel("Posterior probability")
    plt.title("Rank position of positive edges")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_score_distributions(metrics, save_path="score_distributions.png"):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Bayesian Link Prediction: Score Analysis', fontsize=16)

    scores = metrics['probabilities']
    labels = metrics['true_labels']

    if 'pos_ranks_summary' in metrics and 'detailed_file' in metrics['pos_ranks_summary']:
        try:
            with open(metrics['pos_ranks_summary']['detailed_file'], 'r') as f:
                detailed_data = json.load(f)
            pos_ranks = detailed_data['positive_edge_rankings']
        except:
            pos_ranks = []
    else:
        pos_ranks = metrics.get('pos_ranks', [])

    ax = axes[0, 0]
    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]

    if len(pos_scores) > 0:
        ax.hist(neg_scores, bins=50, alpha=0.7, label=f'Negative (n={len(neg_scores)})',
                density=True, color='gray')

        for ps in pos_scores:
            ax.axvline(ps, color='red', linestyle='--', alpha=0.8, linewidth=2)
        ax.scatter(pos_scores, np.zeros_like(pos_scores), color='red', s=100,
                  label=f'Positive (n={len(pos_scores)})', zorder=10)

    ax.set_xlabel('Posterior Probability')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution by Class')
    ax.legend()
    ax.set_yscale('log')

    ax = axes[0, 1]
    sorted_scores = np.sort(scores)[::-1]
    cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)

    ax.plot(sorted_scores, cumulative, 'b-', linewidth=2, label='All edges')

    if len(pos_ranks) > 0:
        for idx, rank, percentile in pos_ranks:
            score = scores[idx]
            ax.scatter(score, rank/len(scores), color='red', s=100, zorder=10)
            ax.annotate(f'Rank {rank}\n({percentile:.1%})',
                       xy=(score, rank/len(scores)),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    ax.set_xlabel('Posterior Probability')
    ax.set_ylabel('Cumulative Fraction')
    ax.set_title('Cumulative Distribution with Positive Edge Positions')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if 'precision_curve' in metrics and 'recall_curve' in metrics:
        precision = metrics['precision_curve']
        recall = metrics['recall_curve']
        ax.plot(recall, precision, 'b-', linewidth=2)
        ax.fill_between(recall, precision, alpha=0.2)

        if 'topN_summary' in metrics:
            for N, summary in metrics['topN_summary'].items():
                if summary['recall'] > 0 and summary['precision'] > 0:
                    ax.scatter(summary['recall'], summary['precision'], s=100, zorder=10)
                    ax.annotate(f'Top-{N}',
                               xy=(summary['recall'], summary['precision']),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8)

        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve (AUC={metrics["auc_pr"]:.3f})')
        ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if 'topN_summary' in metrics:
        N_values = sorted(metrics['topN_summary'].keys())
        precisions = [metrics['topN_summary'][N]['precision'] for N in N_values]
        recalls = [metrics['topN_summary'][N]['recall'] for N in N_values]
        lifts = [metrics['topN_summary'][N]['lift'] for N in N_values]

        ax2 = ax.twinx()

        l1 = ax.plot(N_values, precisions, 'b-o', label='Precision', linewidth=2)
        l2 = ax.plot(N_values, recalls, 'g-s', label='Recall', linewidth=2)

        l3 = ax2.plot(N_values, lifts, 'r-^', label='Lift', linewidth=2)

        ax.set_xlabel('Top-N Edges')
        ax.set_ylabel('Precision / Recall')
        ax2.set_ylabel('Lift', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        ax.set_title('Performance at Different Cutoffs')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

        lns = l1 + l2 + l3
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='best')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return save_path

def plot_gp_prediction_analysis(test_edges, true_labels, predictions, embeddings, save_path):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('GP Model Prediction Analysis', fontsize=16, fontweight='bold')

    ax = axes[0, 0]
    pos_preds = predictions[true_labels == 1]
    neg_preds = predictions[true_labels == 0]

    if len(pos_preds) > 0:
        ax.hist(pos_preds, bins=50, alpha=0.7, label=f'Positive ({len(pos_preds)})', color='red', density=True)
    if len(neg_preds) > 0:
        ax.hist(neg_preds, bins=50, alpha=0.7, label=f'Negative ({len(neg_preds)})', color='blue', density=True)

    ax.set_xlabel('GP Prediction Probability')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Distribution by True Class')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if len(test_edges) > 0:

        emb_i = embeddings[test_edges[:, 0]]
        emb_j = embeddings[test_edges[:, 1]]
        distances = np.linalg.norm(emb_i - emb_j, axis=1)

        if len(distances) > 5000:
            sample_idx = np.random.choice(len(distances), 5000, replace=False)
            distances_sample = distances[sample_idx]
            predictions_sample = predictions[sample_idx]
            labels_sample = true_labels[sample_idx]
        else:
            distances_sample = distances
            predictions_sample = predictions
            labels_sample = true_labels

        pos_mask = labels_sample == 1
        neg_mask = labels_sample == 0

        if np.any(pos_mask):
            ax.scatter(distances_sample[pos_mask], predictions_sample[pos_mask],
                      alpha=0.6, s=20, c='red', label='Positive')
        if np.any(neg_mask):
            ax.scatter(distances_sample[neg_mask], predictions_sample[neg_mask],
                      alpha=0.6, s=20, c='blue', label='Negative')

        ax.set_xlabel('Embedding Distance')
        ax.set_ylabel('GP Prediction')
        ax.set_title('Prediction vs Embedding Distance')
        ax.legend()
        ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if len(predictions) > 0:
        order = np.argsort(predictions)[::-1]
        ordered_labels = true_labels[order]
        cumsum_pos = np.cumsum(ordered_labels)
        ranks = np.arange(1, len(ordered_labels) + 1)
        precision_curve = cumsum_pos / ranks

        ax.plot(ranks, precision_curve, 'b-', linewidth=2)
        ax.axhline(y=true_labels.mean(), color='r', linestyle='--',
                  label=f'Random (prevalence={true_labels.mean():.4f})')

        ax.set_xlabel('Rank')
        ax.set_ylabel('Precision')
        ax.set_title('Precision at Rank')
        ax.set_xscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if len(predictions) > 0:
        top_k = min(50, len(predictions))
        top_indices = np.argsort(predictions)[::-1][:top_k]
        top_preds = predictions[top_indices]
        top_labels = true_labels[top_indices]

        colors = ['red' if label == 1 else 'blue' for label in top_labels]
        bars = ax.bar(range(top_k), top_preds, color=colors, alpha=0.7)

        ax.set_xlabel(f'Top-{top_k} Ranked Edges')
        ax.set_ylabel('GP Prediction')
        ax.set_title(f'Top-{top_k} Predictions (Red=Positive, Blue=Negative)')
        ax.grid(True, alpha=0.3)

        precision_top_k = top_labels.sum() / top_k
        ax.axhline(y=precision_top_k, color='green', linestyle='--',
                  label=f'Top-{top_k} Precision={precision_top_k:.3f}')
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    return save_path

def create_gp_summary_report(metrics, save_path="gp_evaluation_summary.txt"):
    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("GP MODEL LINK PREDICTION EVALUATION SUMMARY\n")
        f.write("="*70 + "\n\n")

        f.write("DATASET OVERVIEW:\n")
        f.write("-"*40 + "\n")
        f.write(f"Total test edges: {len(metrics['true_labels']):,}\n")
        if metrics.get('downsampled', False):
            f.write(f"(Downsampled at rate: {metrics['downsample_rate']:.2%})\n")
        f.write(f"Positive edges: {metrics['num_positive']:,} ({metrics['num_positive']/len(metrics['true_labels'])*100:.4f}%)\n")
        f.write(f"Negative edges: {metrics['num_negative']:,} ({metrics['num_negative']/len(metrics['true_labels'])*100:.4f}%)\n")
        f.write(f"Class imbalance ratio: 1:{metrics['num_negative']/max(1, metrics['num_positive']):.0f}\n\n")

        f.write("POSITIVE EDGE RANKING ANALYSIS:\n")
        f.write("-"*40 + "\n")

        pos_ranks = []
        if 'pos_ranks_summary' in metrics and 'detailed_file' in metrics['pos_ranks_summary']:
            try:
                with open(metrics['pos_ranks_summary']['detailed_file'], 'r') as detail_f:
                    detailed_data = json.load(detail_f)
                pos_ranks = detailed_data['positive_edge_rankings']
            except:
                pass
        elif 'pos_ranks' in metrics:
            pos_ranks = metrics['pos_ranks']

        if pos_ranks:
            f.write(f"Number of positive edges: {len(pos_ranks)}\n")
            ranks = [r for _, r, _ in pos_ranks]
            percentiles = [p for _, _, p in pos_ranks]

            f.write(f"Mean percentile: {metrics['mean_pos_percentile']:.4f}\n")
            f.write(f"Best rank: {min(ranks)} (top {100*(1-max(percentiles)):.2f}%)\n")
            f.write(f"Worst rank: {max(ranks)} (top {100*(1-min(percentiles)):.2f}%)\n")
            f.write(f"Detailed rankings saved to separate file\n\n")
        else:
            f.write("No positive edges found in evaluation set.\n\n")

        if 'topN_summary' in metrics and metrics['topN_summary']:
            f.write("TOP-N PERFORMANCE ANALYSIS:\n")
            f.write("-"*40 + "\n")
            f.write(f"{'N':>6} {'TP':>6} {'Precision':>10} {'Recall':>10} {'Lift':>8}\n")
            f.write("-"*40 + "\n")

            for N in sorted(metrics['topN_summary'].keys()):
                summary = metrics['topN_summary'][N]
                f.write(f"{N:>6} {summary['tp']:>6} {summary['precision']:>10.4f} "
                       f"{summary['recall']:>10.4f} {summary['lift']:>8.2f}\n")
            f.write("\n")

        f.write("CLASSIFICATION METRICS:\n")
        f.write("-"*40 + "\n")
        f.write(f"AUC-ROC:         {metrics['auc_roc']:.4f}\n")
        f.write(f"AUC-PR:          {metrics['auc_pr']:.4f}\n")
        f.write("\n")

        f.write("PERFORMANCE ASSESSMENT:\n")
        f.write("-"*40 + "\n")

        if metrics['mean_pos_percentile'] >= 0.95:
            assessment = "EXCELLENT - GP model ranks positive edges in top 5%"
        elif metrics['mean_pos_percentile'] >= 0.90:
            assessment = "VERY GOOD - GP model ranks positive edges in top 10%"
        elif metrics['mean_pos_percentile'] >= 0.75:
            assessment = "GOOD - GP model ranks positive edges in top 25%"
        elif metrics['mean_pos_percentile'] >= 0.50:
            assessment = "MODERATE - GP model ranks positive edges in top 50%"
        else:
            assessment = "POOR - GP model ranks positive edges below median"

        f.write(f"Overall Assessment: {assessment}\n\n")

        f.write("RECOMMENDATIONS:\n")
        f.write("-"*40 + "\n")

        if metrics['auc_roc'] > 0.8:
            f.write("✓ Good discriminative ability (AUC-ROC > 0.8)\n")
        elif metrics['auc_roc'] > 0.7:
            f.write("~ Moderate discriminative ability (AUC-ROC > 0.7)\n")
        else:
            f.write("⚠ Poor discriminative ability (AUC-ROC <= 0.7)\n")

        if metrics['mean_pos_percentile'] > 0.8:
            f.write("✓ Positive edges are well-ranked by the GP model\n")
        else:
            f.write("⚠ Consider improving GP model architecture or hyperparameters\n")

        f.write("\n" + "="*70 + "\n")
        f.write("END OF GP EVALUATION REPORT\n")
        f.write("="*70 + "\n")

    return save_path

def plot_distribution(scores, labels, set_name, log_dir):
        plt.figure(figsize=(12, 6))
        probabilities = expit(scores)
        sns.kdeplot(probabilities[labels == 0], fill=True, color="skyblue", label="Negative", cut=0)
        sns.kdeplot(probabilities[labels == 1], fill=True, color="red", label="Positive", cut=0)

        plt.title(f'{set_name} Set Predicted Probabilities Distribution')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.legend()
        total = len(labels)
        pos_prop = np.sum(labels == 1) / total
        neg_prop = 1 - pos_prop
        plt.text(0.05, 0.95, f"Negative: {neg_prop:.2%}\nPositive: {pos_prop:.2%}",
                transform=plt.gca().transAxes, verticalalignment='top')
        plt.savefig(os.path.join(log_dir, f'{set_name.lower()}_distribution.png'))
        plt.close()

def append_results_row(output_dir, row):
    results_path = os.path.join(output_dir, "results.jsonl")
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

def build_results_row(metrics, split_name, context):
    row = {
        "schema_version": "1.0",
        "dataset_id": context.get("dataset_id"),
        "dataset_name": context.get("dataset_name"),
        "split": split_name,
        "run_index": context.get("run_index"),
        "run_seed": context.get("run_seed"),
        "embedding_type": context.get("embedding_type"),
        "use_snn": context.get("use_snn"),
        "evaluation_strategy": context.get("evaluation_strategy"),
        "model_type": context.get("model_type"),
        "train_ratio": context.get("train_ratio"),
        "input_dim": context.get("input_dim"),
        "output_dim": context.get("output_dim"),
        "num_epochs": context.get("num_epochs"),
        "batch_size": context.get("batch_size"),
        "learning_rate": context.get("learning_rate"),
        "negative_ratio": context.get("negative_ratio"),
        "temperature": context.get("temperature"),
        "neg_sampling_tag": context.get("neg_sampling_tag"),
        "neg_per_pos": context.get("neg_per_pos"),
        "auc_roc": metrics.get("auc_roc"),
        "auc_pr": metrics.get("auc_pr"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return row

def log_experiment(result):
    dataset_id = result['dataset_id']
    embedding_type = result['embedding_type']
    use_snn = result['use_snn']
    snn_tag = result.get('snn_tag', 'snn' if use_snn else 'no_snn')
    sampling_tag = result.get('neg_sampling_tag', 'default')
    sampling_tag_slug = re.sub(r"[^0-9A-Za-z_-]+", "_", sampling_tag).strip("_") or "default"
    base_dir = os.path.join(
        f'./logs/{embedding_type}_{snn_tag}_DS{dataset_id}',
        f'sampler_{sampling_tag_slug}'
    )
    os.makedirs(base_dir, exist_ok=True)

    all_entries = os.listdir(base_dir)
    version_entries = [d for d in all_entries if d.startswith('version_') and d.count('_') == 1]
    existing_versions = []
    for v in version_entries:
        try:
            version_num = int(v.split('_')[1])
            existing_versions.append(version_num)
        except (ValueError, IndexError):
            continue
    next_version = max(existing_versions + [0]) + 1
    log_dir = os.path.join(base_dir, f'version_{next_version}')
    os.makedirs(log_dir, exist_ok=True)

    params = dict(result)
    with open(os.path.join(log_dir, 'params.json'), 'w') as f:
        json.dump(params, f, indent=4)

    run_manifest = {
        "dataset_id": dataset_id,
        "dataset_name": result.get("dataset_name"),
        "embedding_type": embedding_type,
        "use_snn": use_snn,
        "evaluation_strategy": result.get("evaluation_strategy"),
        "model_type": result.get("model_type"),
        "train_ratio": result.get("train_ratio"),
        "neg_sampling_tag": sampling_tag_slug,
        "neg_sampling_config": result.get("neg_sampling_config"),
        "neg_sampling_weights": result.get("neg_sampling_weights"),
        "neg_sampling_preset": result.get("neg_sampling_preset"),
        "n_runs": result.get("n_runs"),
        "run_seed_base": result.get("run_seed_base"),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    with open(os.path.join(log_dir, "run_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)

    summary_payload = {
        'dataset_id': dataset_id,
        'dataset_name': result.get('dataset_name'),
        'evaluation_strategy': result.get('evaluation_strategy'),
        'embedding_type': embedding_type,
        'model_type': result.get('model_type'),
        'train_ratio': result.get('train_ratio'),
        'input_dim': result.get('input_dim'),
        'output_dim': result.get('output_dim'),
        'num_epochs': result.get('num_epochs'),
        'batch_size': result.get('batch_size'),
        'learning_rate': result.get('learning_rate'),
        'use_snn': use_snn,
        'snn_tag': result.get('snn_tag'),
        'negative_ratio': result.get('negative_ratio'),
        'temperature': result.get('temperature'),
        'n_runs': result.get('n_runs'),
        'run_seed_base': result.get('run_seed_base'),
        'sampling_tag': sampling_tag_slug,
        'neg_sampling_config': result.get('neg_sampling_config'),
        'neg_sampling_weights': result.get('neg_sampling_weights'),
        'neg_sampling_preset': result.get('neg_sampling_preset'),
        'context_adj_count': result.get('context_adj_count'),
        'train_auc': result.get('train_auc'),
        'train_pr_auc': result.get('train_pr_auc'),
        'valid_auc': result.get('valid_auc'),
        'valid_pr_auc': result.get('valid_pr_auc'),
        'test_auc': result.get('test_auc'),
        'test_pr_auc': result.get('test_pr_auc'),
        'test_time': result.get('test_time'),
    }

    with open(os.path.join(log_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(summary_payload, f, indent=4)

    summary_text_path = os.path.join(log_dir, 'metrics_summary.txt')
    with open(summary_text_path, 'w') as f:
        f.write(f"Sampling tag: {sampling_tag_slug}\n")
        f.write(f"Neg sampling weights: {summary_payload['neg_sampling_weights']}\n")
        f.write(f"Valid PR AUC: {summary_payload['valid_pr_auc']}\n")
        f.write(f"Test PR AUC: {summary_payload['test_pr_auc']}\n")

    overview_path = os.path.join(base_dir, 'sampler_overview.tsv')
    overview_header = [
        "timestamp",
        "dataset_id",
        "sampling_tag",
        "valid_pr_auc",
        "test_pr_auc",
        "valid_auc",
        "test_auc",
        "test_time",
    ]
    overview_exists = os.path.exists(overview_path)
    if overview_exists:
        with open(overview_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            existing_header = reader.fieldnames or []
            if existing_header != overview_header:
                existing_rows = []
                for row in reader:
                    existing_rows.append([row.get(col, "") for col in overview_header])
                with open(overview_path, "w", encoding="utf-8", newline="") as wf:
                    writer = csv.writer(wf, delimiter="\t")
                    writer.writerow(overview_header)
                    writer.writerows(existing_rows)
    with open(overview_path, 'a') as f:
        if not overview_exists:
            f.write("\t".join(overview_header) + "\n")
        f.write(
            f"{datetime.datetime.now().isoformat()}\t{dataset_id}\t{sampling_tag_slug}\t"
            f"{summary_payload['valid_pr_auc']}\t{summary_payload['test_pr_auc']}\t"
            f"{summary_payload['valid_auc']}\t"
            f"{summary_payload['test_auc']}\t{summary_payload['test_time']}\n"
        )

    print(f"Experiment logged at: {log_dir}")

    csv_header = [
        "Network", "CellType", "Dataset_ID",
        "EmbeddingType", "UseSNN",
        "Sampling_Tag", "Valid_PR", "Test_AUC", "Test_PR", "Test_Time"
    ]
    master_csv_path = "./master_results.csv"
    file_existed = os.path.isfile(master_csv_path)

    test_auc = result['test_auc']
    test_pr  = result['test_pr_auc']
    test_time = result['test_time']

    if dataset_id in DATASET_INFO_MAPPING:
        net_type, cell_type = DATASET_INFO_MAPPING[dataset_id]
    else:
        net_type, cell_type = ("UnknownNet", f"Dataset_{dataset_id}")

    csv_row = [
        net_type, cell_type, dataset_id,
        embedding_type, use_snn,
        sampling_tag_slug, result.get('valid_pr_auc'),
        test_auc, test_pr, test_time
    ]

    if file_existed:
        with open(master_csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_header = reader.fieldnames or []
            if existing_header != csv_header:
                existing_rows = []
                for row in reader:
                    existing_rows.append([row.get(col, "") for col in csv_header])
                with open(master_csv_path, "w", newline="", encoding="utf-8") as wf:
                    writer = csv.writer(wf)
                    writer.writerow(csv_header)
                    writer.writerows(existing_rows)

    with open(master_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_existed:
            writer.writerow(csv_header)
        writer.writerow(csv_row)

    temp_gp_dir = os.path.join(
        f'./logs/{embedding_type}_{snn_tag}_DS{dataset_id}',
        f'sampler_{sampling_tag_slug}',
        'temp_gp_reports'
    )

    def _move_temp_reports(temp_dir: str, target_name: str):
        if not os.path.exists(temp_dir):
            return
        destination = os.path.join(log_dir, target_name)
        if os.path.exists(destination):
            shutil.rmtree(destination)
        shutil.move(temp_dir, destination)
        print(f"Moved {target_name} to: {destination}")

    _move_temp_reports(temp_gp_dir, 'gp_reports')

    for root, _, files in os.walk(log_dir):
        for name in files:
            if name.startswith('._'):
                try:
                    os.remove(os.path.join(root, name))
                except FileNotFoundError:
                    pass

    print(f"Experiment logged at: {log_dir}")
    return log_dir

def save_split_data(dataset_id, split_name, expression_data, network_data, gene_names, cell_names,
                    base_dir='./data/splits', sampler_tag: Optional[str] = None):
    dataset_dir = os.path.join(base_dir, f'DS{dataset_id}')
    if sampler_tag:
        dataset_dir = os.path.join(dataset_dir, f'sampler_{sampler_tag}')
    split_dir = os.path.join(dataset_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    if gene_names is None:
        gene_names = [f'Gene_{i}' for i in range(expression_data.shape[0])]
    if cell_names is None:
        cell_names = [f'Cell_{i}' for i in range(expression_data.shape[1])]

    expression_df = pd.DataFrame(expression_data, index=gene_names, columns=cell_names)
    expression_df.to_csv(os.path.join(split_dir, 'ExpressionData.csv'))

    positive_edges = np.argwhere(network_data == 1)
    negative_edges = np.argwhere(network_data == 0)

    pos_edge_df = pd.DataFrame(positive_edges, columns=['Gene1', 'Gene2'])
    pos_edge_df['Gene1'] = pos_edge_df['Gene1'].apply(lambda x: gene_names[x])
    pos_edge_df['Gene2'] = pos_edge_df['Gene2'].apply(lambda x: gene_names[x])
    pos_edge_df.to_csv(os.path.join(split_dir, 'pos_refNetwork.csv'), index=False)

    neg_edge_df = pd.DataFrame(negative_edges, columns=['Gene1', 'Gene2'])
    neg_edge_df['Gene1'] = neg_edge_df['Gene1'].apply(lambda x: gene_names[x])
    neg_edge_df['Gene2'] = neg_edge_df['Gene2'].apply(lambda x: gene_names[x])
    neg_edge_df.to_csv(os.path.join(split_dir, 'neg_refNetwork.csv'), index=False)

    ref_edge_df = pd.DataFrame(positive_edges, columns=['Gene1', 'Gene2'])
    ref_edge_df['Gene1'] = ref_edge_df['Gene1'].apply(lambda x: gene_names[x])
    ref_edge_df['Gene2'] = ref_edge_df['Gene2'].apply(lambda x: gene_names[x])
    ref_edge_df.to_csv(os.path.join(split_dir, 'refNetwork.csv'), index=False)

    np.save(os.path.join(split_dir, 'expression.npy'), expression_data)
    np.save(os.path.join(split_dir, 'network.npy'), network_data)

def save_split_info(dataset_id, train_ratio, split_info, base_dir='./data/splits', sampler_tag: Optional[str] = None):
    dataset_dir = os.path.join(base_dir, f'DS{dataset_id}')
    if sampler_tag:
        dataset_dir = os.path.join(dataset_dir, f'sampler_{sampler_tag}')
    os.makedirs(dataset_dir, exist_ok=True)

    info_file = os.path.join(dataset_dir, f'split_info_{train_ratio:.2f}.json')
    with open(info_file, 'w') as f:
        json.dump(split_info, f, indent=4)

def plot_performance_curves(true_labels, pred_probs, save_dir, prefix=''):
    print("before: ", true_labels.shape, pred_probs.shape)
    valid_mask = true_labels != -1
    true_labels = true_labels[valid_mask]
    pred_probs = pred_probs[valid_mask]
    print("after: ", true_labels.shape, pred_probs.shape)

    os.makedirs(save_dir, exist_ok=True)

    fpr, tpr, _ = roc_curve(true_labels, pred_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{prefix} ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{prefix}_roc_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    precision, recall, _ = precision_recall_curve(true_labels, pred_probs)
    pr_auc = auc(recall, precision)

    baseline = np.sum(true_labels) / len(true_labels)

    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='darkorange', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    plt.axhline(y=baseline, color='navy', linestyle='--', label=f'Baseline ({baseline:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{prefix} Precision-Recall Curve')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{prefix}_pr_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(12, 8))
    sns.kdeplot(data=pred_probs[true_labels == 0], label='Negative Class', fill=True)
    sns.kdeplot(data=pred_probs[true_labels == 1], label='Positive Class', fill=True)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title(f'{prefix} Prediction Probability Distribution')
    plt.legend()
    plt.grid(True)

    total = len(true_labels)
    pos_prop = np.sum(true_labels == 1) / total
    neg_prop = 1 - pos_prop

    plt.text(0.02, 0.98,
             f'Class Distribution:\nNegative: {neg_prop:.1%}\nPositive: {pos_prop:.1%}',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.savefig(os.path.join(save_dir, f'{prefix}_prob_dist.png'), dpi=300, bbox_inches='tight')
    plt.close()

    stats = {
        'ROC_AUC': roc_auc,
        'PR_AUC': pr_auc,
        'Positive_Ratio': pos_prop,
        'Mean_Positive_Prob': np.mean(pred_probs[true_labels == 1]),
        'Mean_Negative_Prob': np.mean(pred_probs[true_labels == 0]),
        'Median_Positive_Prob': np.median(pred_probs[true_labels == 1]),
        'Median_Negative_Prob': np.median(pred_probs[true_labels == 0])
    }

    with open(os.path.join(save_dir, f'{prefix}_stats.txt'), 'w') as f:
        for key, value in stats.items():
            f.write(f'{key}: {value:.4f}\n')

def visualize_all_splits(train_labels, train_probs, valid_labels, valid_probs,
                        test_labels, test_probs, save_dir):
    for labels, probs, prefix in [
        (train_labels, train_probs, 'train'),
        (valid_labels, valid_probs, 'valid'),
        (test_labels, test_probs, 'test')
    ]:
        plot_performance_curves(labels, probs, save_dir, prefix)

def save_fold_split_data(split_dir: str, expression_data: np.ndarray, network_data: np.ndarray):

    genes = [f'Gene_{i}' for i in range(expression_data.shape[0])]
    cells = [f'Cell_{i}' for i in range(expression_data.shape[1])]
    expression_df = pd.DataFrame(expression_data, index=genes, columns=cells)
    expression_df.to_csv(os.path.join(split_dir, 'ExpressionData.csv'))

    positive_edges = np.argwhere(network_data == 1)
    negative_edges = np.argwhere(network_data == 0)

    pos_edge_df = pd.DataFrame(positive_edges, columns=['Gene1', 'Gene2'])
    pos_edge_df['Gene1'] = pos_edge_df['Gene1'].apply(lambda x: f'Gene_{x}')
    pos_edge_df['Gene2'] = pos_edge_df['Gene2'].apply(lambda x: f'Gene_{x}')
    pos_edge_df.to_csv(os.path.join(split_dir, 'pos_refNetwork.csv'), index=False)

    neg_edge_df = pd.DataFrame(negative_edges, columns=['Gene1', 'Gene2'])
    neg_edge_df['Gene1'] = neg_edge_df['Gene1'].apply(lambda x: f'Gene_{x}')
    neg_edge_df['Gene2'] = neg_edge_df['Gene2'].apply(lambda x: f'Gene_{x}')
    neg_edge_df.to_csv(os.path.join(split_dir, 'neg_refNetwork.csv'), index=False)

    edge_df = pd.DataFrame(positive_edges, columns=['Gene1', 'Gene2'])
    edge_df['Gene1'] = edge_df['Gene1'].apply(lambda x: f'Gene_{x}')
    edge_df['Gene2'] = edge_df['Gene2'].apply(lambda x: f'Gene_{x}')
    edge_df.to_csv(os.path.join(split_dir, 'refNetwork.csv'), index=False)

    np.save(os.path.join(split_dir, 'expression.npy'), expression_data)
    np.save(os.path.join(split_dir, 'network.npy'), network_data)

def run_experiment(gpu_id, dataset_id, train_ratio, output_dim, num_epochs, batch_size, learning_rate,
                   use_snn, embedding_type, negative_ratio, temperature,
                   result_queue, model_type, n_runs=3,
                   neg_sampling_config: Optional[NegSamplingConfig] = None,
                   neg_sampling_preset: Optional[str] = None,
                   sampling_tag: str = "default",
                   context_override: Optional[List[np.ndarray]] = None):
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except Exception:
        pass

    evaluation_strategy = 'gp'
    device = torch.device('cpu')

    datasets = get_datasets()
    dataset_info = next((dataset for dataset in datasets if dataset['dataset_id'] == dataset_id), None)
    print(f"Dataset Info: {dataset_info}")
    if dataset_info is None:
        raise ValueError(f"Dataset ID {dataset_id} not found in available datasets")
    dataset_id = dataset_info['dataset_id']
    dataset_name = dataset_info.get('dataset_name') or f"DS{dataset_id}"
    print(f"\nProcessing Dataset {dataset_id}...")
    snn_tag = "snn" if use_snn else "no_snn"
    sampling_tag = sampling_tag or "default"
    sampling_tag_slug = re.sub(r"[^0-9A-Za-z_-]+", "_", sampling_tag).strip("_") or "default"
    results_dir_base = f'./results/cl/DS{dataset_id}'
    os.makedirs(results_dir_base, exist_ok=True)
    sampler_dir = os.path.join(results_dir_base, f'sampler_{sampling_tag_slug}')
    os.makedirs(sampler_dir, exist_ok=True)
    ds_clean, ds_noisy, gene_names, cell_names = load_data(dataset_info)
    num_genes = ds_noisy.shape[0]
    num_cells = ds_noisy.shape[1]

    embedding_file = f'./results/cl/DS{dataset_id}/init_embeddings_{embedding_type}.npy'
    if os.path.exists(embedding_file):
        embeddings = np.load(embedding_file)
        print(f"Loaded cached {embedding_type} embeddings")
    else:
        embeddings = generate_embeddings(ds_noisy, embedding_type=embedding_type, n_components=64)
        print(f"Generated new {embedding_type} embeddings")
    print(embeddings[:5])
    input_dim = embeddings.shape[1]
    print(f"gene_names: {len(gene_names)}")
    gt_grn_file = dataset_info['network_file']
    H = load_ground_truth_grn(gt_grn_file, gene_names=gene_names)
    if not os.path.exists(f'./results/cl/DS{dataset_id}'):
        os.makedirs(f'./results/cl/DS{dataset_id}')
    np.save(f'./results/cl/DS{dataset_id}/init_embeddings_{embedding_type}.npy', embeddings)
    print(1)
    valid_ratio = (1 - train_ratio) / 2
    test_ratio = (1 - train_ratio) / 2
    print(2)

    context_adjs = context_override if context_override is not None else load_context_adjs_for_same_celltype(dataset_id, gene_names)

    if neg_sampling_config is not None:
        neg_cfg = replace(neg_sampling_config)
    elif neg_sampling_preset is not None and neg_sampling_preset in NEG_SAMPLING_PRESETS:
        neg_cfg = replace(NEG_SAMPLING_PRESETS[neg_sampling_preset])
    else:
        base_neg_weights = {
            "node_either": 0.35,
            "degree_balanced": 0.20,
            "distance_hard": 0.25,
            "context": 0.20,
        }
        if embeddings is None:
            base_neg_weights.pop("distance_hard", None)
        if not context_adjs:
            base_neg_weights.pop("context", None)
        neg_cfg = NegSamplingConfig(
            weights=dict(base_neg_weights) if base_neg_weights else {"random": 1.0},
            neg_per_pos=5,
            distance_topk=50,
            node_match="either",
            degree_bins=4,
            seed=42,
        )

    sanitized_weights = dict(neg_cfg.weights or {"random": 1.0})
    if embeddings is None:
        sanitized_weights.pop("distance_hard", None)
    if not context_adjs:
        sanitized_weights.pop("context", None)
    if not sanitized_weights:
        sanitized_weights["random"] = 1.0
    neg_cfg = replace(neg_cfg, weights=sanitized_weights)

    results_dir = sampler_dir
    manifest_path = os.path.join(results_dir, 'split_manifest.json')
    manifest_version = None
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, 'r') as f:
                manifest_version = json.load(f).get('sampler_version')
        except Exception:
            manifest_version = None

    g_train_path = os.path.join(results_dir, 'G_train.pkl')
    g_valid_path = os.path.join(results_dir, 'G_valid.pkl')
    g_test_path = os.path.join(results_dir, 'G_test.pkl')

    need_generation = (
        not os.path.exists(g_train_path)
        or not os.path.exists(g_valid_path)
        or not os.path.exists(g_test_path)
        or manifest_version != 'advanced_neg_sampling_v3'
    )

    if need_generation:
        print(f"Generating train/valid/test splits for dataset {dataset_id} (sampler={sampling_tag_slug}) with advanced sampling...")
        G_train, G_valid, G_test = build_train_valid_with_sampling(
            H=H,
            embeddings=embeddings,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            neg_cfg=neg_cfg,
            prior_adj_for_khop=None,
            context_adjs=context_adjs,
            test_ratio=test_ratio,
        )

        with open(g_train_path, 'wb') as f:
            pickle.dump(G_train, f)
        with open(g_valid_path, 'wb') as f:
            pickle.dump(G_valid, f)
        with open(g_test_path, 'wb') as f:
            pickle.dump(G_test, f)

        split_info = {
            'train_ratio': train_ratio,
            'valid_ratio': valid_ratio,
            'test_ratio': test_ratio,
            'num_genes': num_genes,
            'num_cells': num_cells,
            'timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            'generated': True,
            'sampler_version': 'advanced_neg_sampling_v3',
            'neg_sampling_weights': _normalize_weights(neg_cfg.weights or {"random": 1.0}),
            'neg_per_pos': neg_cfg.neg_per_pos,
            'context_adj_count': len(context_adjs),
            'train_pos': int(np.sum(G_train == 1)),
            'train_neg': int(np.sum(G_train == 0)),
            'valid_pos': int(np.sum(G_valid == 1)),
            'valid_neg': int(np.sum(G_valid == 0)),
            'test_pos': int(np.sum(G_test == 1)),
            'test_neg': int(np.sum(G_test == 0)),
        }
        with open(manifest_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        save_split_info(dataset_id, train_ratio, split_info, sampler_tag=sampling_tag_slug)

        for split_name, exp_data, net_data in [
            ('train', ds_noisy, G_train),
            ('valid', ds_noisy, G_valid),
            ('test', ds_noisy, G_test),
        ]:
            save_split_data(dataset_id, split_name, exp_data, net_data, gene_names, cell_names,
                            sampler_tag=sampling_tag_slug)
            split_dir = os.path.join('./data/splits', f'DS{dataset_id}', f'sampler_{sampling_tag_slug}', split_name)
            normalized_data = pd.DataFrame(exp_data, index=gene_names, columns=cell_names)
            normalized_data.to_csv(os.path.join(split_dir, 'bin-normalized-matrix.csv'))

        print("✓ Generated and saved new splits")
    else:
        print(f"Loading existing splits for dataset {dataset_id} (sampler={sampling_tag_slug})...")
        with open(g_train_path, 'rb') as f:
            G_train = pickle.load(f)
        with open(g_valid_path, 'rb') as f:
            G_valid = pickle.load(f)
        with open(g_test_path, 'rb') as f:
            G_test = pickle.load(f)
        print("✓ Loaded existing splits")

    print(f"\nSplit statistics (sampler={sampling_tag_slug}):")
    print(f"Train: {np.sum(G_train == 1)} positive, {np.sum(G_train == 0)} negative edges, {np.sum(G_train == -1)} ignored edges")
    print(f"Valid: {np.sum(G_valid == 1)} positive, {np.sum(G_valid == 0)} negative edges, {np.sum(G_valid == -1)} ignored edges")
    print(f"Test: {np.sum(G_test == 1)} positive, {np.sum(G_test == 0)} negative edges, {np.sum(G_test == -1)} ignored edges")
    print(f"Total positive edges: {np.sum(G_train == 1) + np.sum(G_valid == 1) + np.sum(G_test == 1)}")

    print(5)
    all_train_metrics = []
    all_valid_metrics = []
    all_test_metrics = []
    best_model = None
    best_valid_auc = -1
    run_times = []
    for run in range(n_runs):

        print(f"Run {run + 1}/{n_runs}")
        run_seed = 42 + run
        random.seed(run_seed)
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        torch.cuda.manual_seed_all(run_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        seed_everything(run_seed)
        model = None
        run_start = time.time()
        result_output_dir = None
        if use_snn:
            model = train_snn_directional(
                embeddings,
                G_train,
                input_dim,
                output_dim,
                num_epochs,
                batch_size,
                learning_rate,
                negative_ratio,
                temperature,
                device,
            )
            with torch.no_grad():
                embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
                projected_embeddings = model.get_embeddings(embeddings_tensor, combine_mode="avg").cpu().numpy()
        else:
            projected_embeddings = embeddings

        gp_model, likelihood, X_train, y_train = train_gp_model(
            projected_embeddings,
            G_train,
            device,
            model_type=model_type,
            direction_weight=1.0,
            inducing_points_num=500,
            num_epochs=50,
            batch_size=1024,
            run_seed=run_seed,
        )

        result_output_dir = os.path.join(
            f'./logs/{embedding_type}_{snn_tag}_DS{dataset_id}',
            f'sampler_{sampling_tag_slug}',
            'temp_gp_reports'
        )
        os.makedirs(result_output_dir, exist_ok=True)

        train_metrics = evaluate_bayesian_model_gp(
            gp_model,
            likelihood,
            projected_embeddings,
            G_train,
            H,
            device,
            create_visualizations=True,
            output_dir=os.path.join(result_output_dir, 'train'),
        )
        valid_metrics = evaluate_bayesian_model_gp(
            gp_model,
            likelihood,
            projected_embeddings,
            G_valid,
            H,
            device,
            create_visualizations=True,
            output_dir=os.path.join(result_output_dir, 'valid'),
        )
        test_metrics = evaluate_bayesian_model_gp(
            gp_model,
            likelihood,
            projected_embeddings,
            G_test,
            H,
            device,
            create_visualizations=True,
            output_dir=os.path.join(result_output_dir, 'test'),
        )
        print("train_metrics: ", train_metrics)
        print("valid_metrics: ", valid_metrics)
        print("test_metrics: ", test_metrics)
        model_save_path = os.path.join(results_dir, f'gp_model_{embedding_type}_{snn_tag}.pth')
        torch.save({
            'model_state_dict': gp_model.state_dict(),
            'likelihood_state_dict': likelihood.state_dict(),
            'inducing_points': gp_model.variational_strategy.inducing_points.detach(),
            'train_x': X_train,
            'train_y': y_train
        }, model_save_path)

        if result_output_dir:
            results_context = {
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "embedding_type": embedding_type,
                "use_snn": use_snn,
                "evaluation_strategy": evaluation_strategy,
                "model_type": model_type,
                "train_ratio": train_ratio,
                "input_dim": input_dim,
                "output_dim": output_dim,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "negative_ratio": negative_ratio,
                "temperature": temperature if use_snn else None,
                "neg_sampling_tag": sampling_tag_slug,
                "neg_per_pos": neg_cfg.neg_per_pos,
                "run_index": run,
                "run_seed": run_seed,
            }
            append_results_row(
                result_output_dir,
                build_results_row(train_metrics, "train", results_context),
            )
            append_results_row(
                result_output_dir,
                build_results_row(valid_metrics, "valid", results_context),
            )
            append_results_row(
                result_output_dir,
                build_results_row(test_metrics, "test", results_context),
            )

        run_end = time.time()
        one_run_time = run_end - run_start
        run_times.append(one_run_time)

        print("i")
        plt.figure(figsize=(12, 6))
        plt.plot(train_metrics['recall_curve'], train_metrics['precision_curve'], label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Train Precision-Recall curve')
        plt.legend()
        os.makedirs(results_dir, exist_ok=True)
        plt.savefig(os.path.join(results_dir, 'train_precision_recall_curve.png'))
        print("j")

        plt.figure(figsize=(12, 6))
        plt.plot(valid_metrics['recall_curve'], valid_metrics['precision_curve'], label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Validation Precision-Recall curve')
        plt.legend()
        plt.savefig(os.path.join(results_dir, 'valid_precision_recall_curve.png'))
        print("k")

        plt.figure(figsize=(12, 6))
        plt.plot(test_metrics['recall_curve'], test_metrics['precision_curve'], label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Test Precision-Recall curve')
        plt.legend()
        plt.savefig(os.path.join(results_dir, 'test_precision_recall_curve.png'))
        print(4)
        all_train_metrics.append(train_metrics)
        all_valid_metrics.append(valid_metrics)
        all_test_metrics.append(test_metrics)
        print(5)
        if valid_metrics['auc_roc'] > best_valid_auc:
            best_valid_auc = valid_metrics['auc_roc']
            best_model = model
        print(6)
        metric_keys = ['auc_roc', 'auc_pr']
        final_metrics = {
            'train': {'means': {}, 'stds': {}},
            'valid': {'means': {}, 'stds': {}},
            'test': {'means': {}, 'stds': {}}
        }
        print(7)
        for metric in metric_keys:

            values = [m[metric] for m in all_train_metrics]
            final_metrics['train']['means'][metric] = np.mean(values)
            final_metrics['train']['stds'][metric] = np.std(values)

            values = [m[metric] for m in all_valid_metrics]
            final_metrics['valid']['means'][metric] = np.mean(values)
            final_metrics['valid']['stds'][metric] = np.std(values)

            values = [m[metric] for m in all_test_metrics]
            final_metrics['test']['means'][metric] = np.mean(values)
            final_metrics['test']['stds'][metric] = np.std(values)
        print(8)

        if use_snn and best_model is not None:
            best_model.eval()
            with torch.no_grad():
                embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
                projected_embeddings = best_model.get_embeddings(embeddings_tensor, combine_mode="avg").cpu().numpy()
        else:
            projected_embeddings = embeddings
    time_mean = np.mean(run_times)
    time_std = np.std(run_times)
    time_str = f"{time_mean:.2f} +- {time_std:.2f}"
    print(f"Average run time: {time_str} seconds")

    result_queue.put({
        'dataset_id': dataset_id,
        'dataset_name': dataset_name,
        'evaluation_strategy': evaluation_strategy,
        'embedding_type': embedding_type,
        'model_type': model_type,
        'train_ratio': train_ratio,
        'input_dim': input_dim,
        'output_dim': output_dim,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'use_snn': use_snn,
        'snn_tag': snn_tag,
        'negative_ratio': negative_ratio,
        'temperature': temperature if use_snn else None,
        'n_runs': n_runs,
        'run_seed_base': 42,
        'train_auc': f"{final_metrics['train']['means']['auc_roc']:.4f} +- {final_metrics['train']['stds']['auc_roc']:.4f}",
        'valid_auc': f"{final_metrics['valid']['means']['auc_roc']:.4f} +- {final_metrics['valid']['stds']['auc_roc']:.4f}",
        'test_auc': f"{final_metrics['test']['means']['auc_roc']:.4f} +- {final_metrics['test']['stds']['auc_roc']:.4f}",
        'train_pr_auc': f"{final_metrics['train']['means']['auc_pr']:.4f} +- {final_metrics['train']['stds']['auc_pr']:.4f}",
        'valid_pr_auc': f"{final_metrics['valid']['means']['auc_pr']:.4f} +- {final_metrics['valid']['stds']['auc_pr']:.4f}",
        'test_pr_auc': f"{final_metrics['test']['means']['auc_pr']:.4f} +- {final_metrics['test']['stds']['auc_pr']:.4f}",
        'test_time': time_str,
        'neg_sampling_tag': sampling_tag_slug,
        'neg_sampling_config': asdict(neg_cfg),
        'neg_sampling_weights': _normalize_weights(neg_cfg.weights or {"random": 1.0}),
        'neg_sampling_preset': neg_sampling_preset,
        'context_adj_count': len(context_adjs),

    })
    print(9.01)

def logger_process(result_queue, num_experiments):
    experiments_logged = 0
    while experiments_logged < num_experiments:
        try:
            result = result_queue.get(timeout=1)
            print("Got result, logging...")
            log_dir = log_experiment(result)
            experiments_logged += 1
        except queue.Empty:
            continue

import random

def process_batch(batch):
    processes = []
    for args in batch:
        (
            gpu_id,
            embedding_type,
            use_snn,
            model_type,
            output_dim,
            num_epochs,
            train_ratio,
            dataset_id,
            batch_size,
            learning_rate,
            negative_ratio,
            temperature,
            sampling_tag,
            neg_sampling_preset,
            neg_sampling_config,
            n_runs_cfg,
        ) = args

        p = multiprocessing.Process(
            target=run_experiment,
            args=(
                gpu_id,
                dataset_id,
                train_ratio,
                output_dim,
                num_epochs,
                batch_size,
                learning_rate,
                use_snn,
                embedding_type,
                negative_ratio,
                temperature,
                result_queue,
                model_type,
                n_runs_cfg,
            ),
            kwargs={
                'sampling_tag': sampling_tag,
                'neg_sampling_preset': neg_sampling_preset,
                'neg_sampling_config': neg_sampling_config,
            },
        )

        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_everything(42)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    num_gpus = torch.cuda.device_count()

    embedding_types = ['fa']
    use_snn_options = [True]
    model_types = ['standard']
    output_dims = [16]
    num_epochs_list = [50]
    train_ratios = [0.8]
    n_runs_cfg = int(os.getenv("N_RUNS", "3"))

    dataset_ids = [1701]
    batch_sizes = [32]
    learning_rates = [1e-3]
    negative_ratios = [5]
    temperatures = [1]
    neg_sampling_options = [
        'random',
    ]

    experiments = []
    for embedding_type, use_snn, model_type, output_dim, num_epochs, train_ratio, dataset_id, batch_size, learning_rate, negative_ratio, sampling_option in product(
        embedding_types,
        use_snn_options,
        model_types,
        output_dims,
        num_epochs_list,
        train_ratios,
        dataset_ids,
        batch_sizes,
        learning_rates,
        negative_ratios,
        neg_sampling_options,
    ):
        sampling_tag, sampling_preset, sampling_config = _resolve_sampling_option(sampling_option)
        if use_snn:
            for temperature in temperatures:
                experiments.append((
                    embedding_type,
                    use_snn,
                    model_type,
                    output_dim,
                    num_epochs,
                    train_ratio,
                    dataset_id,
                    batch_size,
                    learning_rate,
                    negative_ratio,
                    temperature,
                    sampling_tag,
                    sampling_preset,
                    sampling_config,
                    n_runs_cfg,
                ))
        else:
            experiments.append((
                embedding_type,
                use_snn,
                model_type,
                output_dim,
                num_epochs,
                train_ratio,
                dataset_id,
                batch_size,
                learning_rate,
                negative_ratio,
                None,
                sampling_tag,
                sampling_preset,
                sampling_config,
                n_runs_cfg,
            ))

    print(f"Running {len(experiments)} experiments on {num_gpus} GPUs.")
    print(experiments)

    experiments_with_gpus = []
    for idx, exp in enumerate(experiments):
        if num_gpus > 0:
            gpu_id = idx % num_gpus
        else:
            gpu_id = None
        experiments_with_gpus.append((gpu_id,) + exp)

    result_queue = multiprocessing.Queue()
    logger = multiprocessing.Process(target=logger_process, args=(result_queue, len(experiments)))
    logger.start()

    batch_process_size = max(num_gpus * 2, 1)
    for i in range(0, len(experiments_with_gpus), batch_process_size):
        batch = experiments_with_gpus[i:i+batch_process_size]
        process_batch(batch)

    logger.join()
    print("All experiments completed and logged.")
