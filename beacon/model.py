"""Shared pair encoder and additive sparse variational GP."""
import copy
import hashlib
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import gpytorch
from sklearn.metrics import average_precision_score
ENCODER_EPOCHS = 100
GP_LEARNING_RATE = 0.01
GP_WEIGHT_DECAY = 0.01
GP_MAX_EPOCHS = 200
GP_CHECK_EVERY = 5
GP_PATIENCE = 20
GP_FALLBACK_EPOCHS = 25
VALIDATION_FRACTION = 0.1
VALIDATION_SEED = 42
MIN_VALIDATION_POSITIVES = 10

class Training:
    """One fit owns its split, epoch selection and training log."""
    def __init__(self, *, snn_weight=0.0):
        if not np.isfinite(snn_weight) or snn_weight < 0:
            raise ValueError("snn_weight must be finite and nonnegative")
        splits = {}
        self.training_log = {"mode": "refit", "fits": []}

        def matrix_key(adjacency):
            edges = np.argwhere(adjacency >= 0)
            labels = adjacency[tuple(edges.T)].astype(np.int8)
            return (hashlib.sha256(np.asarray(adjacency.shape, dtype=np.int64).tobytes() + edges.astype(np.int64).tobytes() + labels.tobytes()).hexdigest(), edges, labels)

        def internal_split(adjacency):
            """Stratified 10% hold-out of labeled training pairs; cached by the labeled entries of the matrix."""
            key, edges, labels = matrix_key(adjacency)
            if key in splits:
                return splits[key]
            rng = np.random.default_rng(VALIDATION_SEED)
            held = []
            for value in (1, 0):
                index = np.flatnonzero(labels == value)
                count = int(round(VALIDATION_FRACTION * len(index)))
                held.append(np.sort(rng.choice(index, count, replace=False)) if count else np.empty(0, dtype=int))
            held = np.concatenate(held).astype(int)
            info = {'labeled_pairs': int(len(labels)), 'labeled_positives': int((labels == 1).sum()), 'validation_pairs': int(len(held)), 'validation_positives': int((labels[held] == 1).sum()), 'key': key}
            if info['validation_positives'] < MIN_VALIDATION_POSITIVES:
                info.update(fallback=True, validation_pairs=0, validation_positives=0)
                result = {'train': adjacency, 'edges': None, 'labels': None, 'info': info}
            else:
                training = adjacency.copy()
                training[tuple(edges[held].T)] = -1
                info['fallback'] = False
                result = {'train': training, 'edges': edges[held], 'labels': labels[held], 'info': info, 'linear': np.sort(edges[held, 0].astype(np.int64) * adjacency.shape[1] + edges[held, 1])}
            splits[key] = result
            return result

        def check_disjoint(split, used_edges, where):
            if split is None or split['edges'] is None:
                return
            linear = used_edges[:, 0].astype(np.int64) * split['train'].shape[1] + used_edges[:, 1]
            overlap = np.intersect1d(linear, split['linear'], assume_unique=False)
            assert len(overlap) == 0, f'{len(overlap)} internal validation pairs entered {where}'

        def check_all_pairs(adjacency, used_edges, where):
            labeled = int((adjacency >= 0).sum())
            assert len(used_edges) == labeled, f'{where} used {len(used_edges)} of {labeled} labeled training pairs'

        class Encoder(nn.Module):

            def __init__(self, input_dim, output_dim, hidden_dim=256):
                super().__init__()
                self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim), nn.LayerNorm(output_dim))
                self.decoder = nn.Sequential(nn.Linear(4 * output_dim, 4 * output_dim), nn.ReLU(), nn.Linear(4 * output_dim, 1))

            def forward(self, source, target):
                return (self.encoder(source), self.encoder(target))

            def edge_logits(self, source, target):
                pair = torch.cat([source, target, source * target, (source - target).abs()], dim=-1)
                return self.decoder(pair).squeeze(-1)

            def get_embeddings(self, x, combine_mode='avg'):
                del combine_mode
                with torch.no_grad():
                    return self.encoder(x)

        def snn_loss(z, labels, temperature):
            """Soft nearest-neighbour loss with cosine similarity, excluding each point from its own neighbours."""
            z = F.normalize(z, dim=1)
            logits = (z @ z.T - 1) / temperature
            other = ~torch.eye(len(z), dtype=torch.bool, device=z.device)
            same = (labels[:, None] == labels[None, :]) & other
            valid = same.any(dim=1)
            if not valid.any():
                return z.sum() * 0
            everyone = torch.logsumexp(logits.masked_fill(~other, -torch.inf)[valid], dim=1)
            neighbours = torch.logsumexp(logits.masked_fill(~same, -torch.inf)[valid], dim=1)
            return (everyone - neighbours).mean()

        def fit_encoder(embeddings, training, input_dim, projection_dim, batch_size, learning_rate, negative_ratio, temperature, device, split):
            """Shared encoder training for ENCODER_EPOCHS on the labeled pairs of `training`."""
            model = Encoder(input_dim, projection_dim).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
            positives = np.argwhere(training == 1)
            unlabeled = np.argwhere(training == 0)
            if len(positives) == 0 or len(unlabeled) == 0:
                raise ValueError('Training needs supplied positive and sampled unlabeled pairs')
            check_disjoint(split, np.concatenate([positives, unlabeled]), 'encoder training pairs')
            features = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
            batches = int(np.ceil(len(positives) / batch_size))
            rng = np.random.default_rng(torch.initial_seed())
            model.train()
            for epoch in range(ENCODER_EPOCHS):
                positive_order = rng.permutation(len(positives))
                total = 0.0
                for batch in range(batches):
                    pos = positives[positive_order[batch * batch_size:(batch + 1) * batch_size]]
                    neg = unlabeled[rng.choice(len(unlabeled), len(pos) * int(negative_ratio), replace=False)]
                    edges = torch.as_tensor(np.concatenate([pos, neg]), dtype=torch.long, device=device)
                    labels = torch.cat([torch.ones(len(pos), device=device), torch.zeros(len(neg), device=device)])
                    source, target = model(features[edges[:, 0]], features[edges[:, 1]])
                    bce = F.binary_cross_entropy_with_logits(model.edge_logits(source, target), labels, pos_weight=torch.tensor(len(neg) / len(pos), device=device))
                    loss = bce + snn_weight * snn_loss(torch.cat([source, target]), torch.cat([labels, labels]), temperature)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total += float(loss.detach())
                print(f'Epoch {epoch + 1}/{ENCODER_EPOCHS}, Loss: {total / batches:.4f}')
            model.eval()
            return (model, np.concatenate([positives, unlabeled]))

        def train_encoder(embeddings, adjacency_matrix, input_dim, projection_dim, num_epochs, batch_size, learning_rate, negative_ratio, temperature, device):
            args = (input_dim, projection_dim, batch_size, learning_rate, negative_ratio, temperature, device)
            record = {'epochs': ENCODER_EPOCHS, 'requested_epochs': int(num_epochs), 'mode': 'refit'}
            seed = torch.initial_seed()
            split = internal_split(adjacency_matrix)
            if split['info']['fallback']:
                started = time.perf_counter()
                model, used = fit_encoder(embeddings, split['train'], *args, split)
                record.update(training_pairs=int(len(used)), seconds=time.perf_counter() - started, **split['info'])
            else:
                started = time.perf_counter()
                encoder90, used90 = fit_encoder(embeddings, split['train'], *args, split)
                step1_encoder_seconds = time.perf_counter() - started
                with torch.no_grad():
                    projected90 = encoder90.encoder(torch.as_tensor(embeddings, dtype=torch.float32, device=device)).cpu().numpy()
                started = time.perf_counter()
                _, _, _, _, selection = fit_gp(projected90, split['train'], device, run_seed=seed, split=split, max_epochs=GP_MAX_EPOCHS, early=True)
                selection['seconds'] = time.perf_counter() - started
                split['e_star'] = selection['best_epoch']
                split['selection'] = selection
                torch.manual_seed(seed)
                started = time.perf_counter()
                model, used = fit_encoder(embeddings, adjacency_matrix, *args, None)
                check_all_pairs(adjacency_matrix, used, 'refit encoder (step 2)')
                record.update(training_pairs=int(len(used)), step1_training_pairs=int(len(used90)), step1_encoder_seconds=step1_encoder_seconds, step1_gp_seconds=selection['seconds'], step2_encoder_seconds=time.perf_counter() - started, e_star=int(split['e_star']), step1_selection=selection, **split['info'])
            self.encoder, self.encoder_features = (model, embeddings)
            self.training_log['encoder'] = record
            model.train()
            return model

        class PairGP(gpytorch.models.ApproximateGP):

            def __init__(self, inducing_points):
                distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
                strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, distribution, learn_inducing_locations=True)
                super().__init__(strategy)
                self.mean_module = gpytorch.means.ConstantMean()

                def rbf(dims=None):
                    return gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(active_dims=dims))
                self.covar_module = rbf()
                d = inducing_points.size(-1) // 2
                self.covar_module = rbf(torch.arange(d)) + rbf(torch.arange(d, 2 * d)) + rbf()

            def forward(self, x):
                return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))

        def validation_ap(model, likelihood, x_valid, y_valid):
            """AP of the predictive probability on the internal validation pairs; random-number state unchanged."""
            cpu_state = torch.get_rng_state()
            cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            model.eval()
            likelihood.eval()
            scores = []
            with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(0.1):
                for offset in range(0, len(x_valid), 2048):
                    scores.append(likelihood(model(x_valid[offset:offset + 2048])).mean.cpu().numpy())
            model.train()
            likelihood.train()
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            return float(average_precision_score(y_valid, np.concatenate(scores)))

        def fit_gp(projected_embeddings, training, device, run_seed, split, max_epochs, early, inducing_points_num=500, batch_size=1024):
            """GP on the labeled pairs of `training`; with `early`, stop on AP of the split's held-out pairs."""
            torch.manual_seed(run_seed)
            edges = np.argwhere(training >= 0)
            check_disjoint(split, edges, 'GP training pairs')
            labels = training[tuple(edges.T)].astype(np.float32)
            z = torch.as_tensor(projected_embeddings, dtype=torch.float32)
            x_train = torch.cat([z[edges[:, 0]], z[edges[:, 1]]], dim=1).to(device)
            y_train = torch.as_tensor(labels, device=device)
            rng = np.random.default_rng(run_seed)
            count = min(inducing_points_num, len(labels))
            positive_count = min(int(labels.sum()), count // 2)
            chosen = np.concatenate([rng.choice(np.flatnonzero(labels == 1), positive_count, replace=False), rng.choice(np.flatnonzero(labels == 0), count - positive_count, replace=False)])
            check_disjoint(split, edges[chosen], 'inducing-point initialization')
            model = PairGP(x_train[torch.as_tensor(chosen, device=device)].clone()).to(device)
            likelihood = gpytorch.likelihoods.BernoulliLikelihood().to(device)
            objective = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(y_train))
            optimizer = torch.optim.AdamW([{'params': model.parameters()}, {'params': likelihood.parameters()}], lr=GP_LEARNING_RATE, weight_decay=GP_WEIGHT_DECAY)
            loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
            if early:
                x_valid = torch.cat([z[split['edges'][:, 0]], z[split['edges'][:, 1]]], dim=1).to(device)
                y_valid = split['labels'] == 1
            best_ap, best_epoch, best_state, checks, stop, done = (-np.inf, 0, None, [], None, 0)
            model.train()
            likelihood.train()
            for epoch in range(max_epochs):
                total = 0.0
                for x_batch, y_batch in loader:
                    optimizer.zero_grad()
                    with gpytorch.settings.cholesky_jitter(0.1):
                        loss = -objective(model(x_batch), y_batch)
                    loss.backward()
                    optimizer.step()
                    total += float(loss.detach())
                done = epoch + 1
                print(f'Epoch {done}/{max_epochs}, Loss: {total / len(loader):.4f}')
                if early and done % GP_CHECK_EVERY == 0:
                    value = validation_ap(model, likelihood, x_valid, y_valid)
                    checks.append({'epoch': done, 'validation_ap': value})
                    if value > best_ap:
                        best_ap, best_epoch = (value, done)
                        best_state = (copy.deepcopy(model.state_dict()), copy.deepcopy(likelihood.state_dict()))
                    if done - best_epoch >= GP_PATIENCE:
                        stop = done
                        break
            stop = stop or done
            if early and best_state is not None:
                model.load_state_dict(best_state[0])
                likelihood.load_state_dict(best_state[1])
            record = {'early_stopping': early, 'stop_epoch': int(stop), 'best_epoch': int(best_epoch if early else stop), 'best_validation_ap': float(best_ap) if early else None, 'validation_checks': checks, 'max_epochs': int(max_epochs), 'training_pairs': int(len(labels))}
            return (model, likelihood, x_train, y_train, record)

        def train_gp(projected_embeddings, adjacency_matrix, device, inducing_points_num=500, num_epochs=50, batch_size=1024, run_seed=42):
            started = time.perf_counter()
            common = {'inducing_points_num': inducing_points_num, 'batch_size': batch_size}
            split = internal_split(adjacency_matrix)
            if split['info']['fallback']:
                model, likelihood, x_train, y_train, record = fit_gp(projected_embeddings, adjacency_matrix, device, run_seed, None, GP_FALLBACK_EPOCHS, False, **common)
                record['stage'] = 'fallback'
            else:
                selection = split.get('selection')
                if selection is None:
                    _, _, _, _, selection = fit_gp(projected_embeddings, split['train'], device, run_seed, split, GP_MAX_EPOCHS, True, **common)
                    selection['selection_embeddings'] = 'provided'
                    split['e_star'], split['selection'] = (selection['best_epoch'], selection)
                model, likelihood, x_train, y_train, record = fit_gp(projected_embeddings, adjacency_matrix, device, run_seed, None, int(split['e_star']), False, **common)
                assert len(y_train) == int((adjacency_matrix >= 0).sum()), 'refit GP must use every labeled training pair'
                record.update(stage='refit', e_star=int(split['e_star']), step1_selection=selection)
            record.update(split['info'])
            record.update(mode='refit', requested_epochs=int(num_epochs), seconds=time.perf_counter() - started)
            self.training_log['fits'].append(record)
            print(f"BEACON GP ({'refit'}, {record['stage']}): stop epoch {record['stop_epoch']}, best epoch {record['best_epoch']}, validation AP {record['best_validation_ap']}, e* {record.get('e_star')}", flush=True)
            return (model, likelihood, x_train, y_train)

        self.fit_encoder = train_encoder
        self.fit_gp = train_gp
        self.internal_split = internal_split
        self.encoder_class = Encoder
        self.gp_class = PairGP
        self.snn_loss = snn_loss
