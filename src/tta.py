"""Online test-time adaptation (TTA) for PointMLP.

Methods (same definitions and defaults as the earlier DGCNN test-time adaptation study):
  source_only  no adaptation
  bn_adapt     BatchNorm uses test-batch statistics
  tent         entropy minimisation, BatchNorm affine parameters (Adam, lr 1e-3)
  shot_im      entropy minimisation + diversity, all parameters except the last layer (Adam, lr 1e-4)
  t3a          prototype classifier built from confident test features (100 supports per class)
  bn_t3a       bn_adapt + t3a
  sar          entropy minimisation on reliable samples with sharpness-aware updates of the BatchNorm
               affine parameters, and model reset when the loss collapses (Niu et al., ICLR 2023;
               SGD lr 2.5e-4, momentum 0.9, rho 0.05, entropy margin 0.4 ln C, reset at 0.2)

The test label space is a list of groups of ModelNet40 classes; the logit of a group is the
log-sum-exp of its members (e.g. ScanObjectNN "cabinet" = dresser, night_stand, wardrobe).
Hyperparameters can be changed with Adapter(..., hparams={...}) (configs/test.yaml, tta.hparams).
Each adapter works on its own copy of the model. One gradient step per batch; the prediction
for a batch comes from the forward pass before its update, as in TENT.
"""
import copy

import torch
import torch.nn as nn

METHODS = ["source_only", "bn_adapt", "tent", "shot_im", "t3a", "bn_t3a", "sar"]
HPARAMS = dict(tent_lr=1e-3, shot_lr=1e-4, t3a_filter_k=100,
               sar_lr=2.5e-4, sar_momentum=0.9, sar_rho=0.05, sar_margin=0.4, sar_reset=0.2)
_BN = (nn.BatchNorm1d, nn.BatchNorm2d)


def _bn_batch_stats(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, _BN):
            m.train()
            m.track_running_stats = False
            m.running_mean, m.running_var = None, None


def _entropy(logits):
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


class Adapter:
    def __init__(self, model, method, groups, hparams=None):
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}")
        unknown = set(hparams or {}) - set(HPARAMS)
        if unknown:
            raise ValueError(f"unknown hyperparameters {sorted(unknown)}")
        self.hp = {**HPARAMS, **(hparams or {})}
        self.method, self.groups = method, groups
        self.model = copy.deepcopy(model)
        self.head = self.model.classifier[-1]          # Linear(256, 40)
        self.opt = None
        if method in ("source_only", "t3a"):
            self.model.eval()
        else:
            _bn_batch_stats(self.model)
        if method in ("t3a", "bn_t3a"):
            self._init_t3a()
        if method in ("tent", "shot_im", "sar"):
            for p in self.model.parameters():
                p.requires_grad_(False)
            if method in ("tent", "sar"):
                params = [p for m in self.model.modules() if isinstance(m, _BN) for p in (m.weight, m.bias)]
                lr = self.hp["tent_lr"]
            else:
                head = set(map(id, self.head.parameters()))
                params = [p for p in self.model.parameters() if id(p) not in head]
                lr = self.hp["shot_lr"]
            for p in params:
                p.requires_grad_(True)
            if method == "sar":
                self.params = params
                self.opt = torch.optim.SGD(params, lr=self.hp["sar_lr"], momentum=self.hp["sar_momentum"])
                self.margin = self.hp["sar_margin"] * torch.log(torch.tensor(float(len(groups)))).item()
                self.ema = None
                self.initial = copy.deepcopy(self.model.state_dict())
                self.initial_opt = copy.deepcopy(self.opt.state_dict())
            else:
                self.opt = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))

    def group_logits(self, logits):
        return torch.stack([logits[:, g].logsumexp(1) for g in self.groups], 1)

    def _init_t3a(self):
        W = self.head.weight.detach()
        rows = [j for g in self.groups for j in g]
        labels = torch.tensor([k for k, g in enumerate(self.groups) for _ in g], device=W.device)
        self.support = W[rows].clone()
        self.support_labels = nn.functional.one_hot(labels, len(self.groups)).float()
        self.support_ent = _entropy(self.group_logits(self.head(self.support))).detach()
        self._feat = None
        self.head.register_forward_pre_hook(lambda mod, inp: setattr(self, "_feat", inp[0]))

    @torch.no_grad()
    def _t3a(self, x):
        logits = self.group_logits(self.model(x))
        z = self._feat
        y = nn.functional.one_hot(logits.argmax(1), logits.shape[1]).float()
        self.support = torch.cat([self.support, z])
        self.support_labels = torch.cat([self.support_labels, y])
        self.support_ent = torch.cat([self.support_ent, _entropy(logits)])
        cls = self.support_labels.argmax(1)
        keep = []
        for c in range(self.support_labels.shape[1]):
            idx = (cls == c).nonzero(as_tuple=True)[0]
            keep.append(idx[self.support_ent[idx].argsort()[:self.hp["t3a_filter_k"]]])
        keep = torch.cat(keep)
        self.support, self.support_labels, self.support_ent = (
            self.support[keep], self.support_labels[keep], self.support_ent[keep])
        protos = nn.functional.normalize(self.support, dim=1).T @ self.support_labels
        return z @ nn.functional.normalize(protos, dim=0)

    def _sar(self, x):
        logits = self.group_logits(self.model(x))
        ent = _entropy(logits)
        reliable = ent < self.margin
        if reliable.any():
            ent[reliable].mean().backward()
            # sharpness-aware step: move to the worst-case point within radius rho, then update
            grads = [p.grad for p in self.params if p.grad is not None]
            norm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2) + 1e-12
            eps = []
            with torch.no_grad():
                for p in self.params:
                    e = self.hp["sar_rho"] * p.grad / norm if p.grad is not None else torch.zeros_like(p)
                    p.add_(e)
                    eps.append(e)
            self.opt.zero_grad()
            ent2 = _entropy(self.group_logits(self.model(x)))
            reliable2 = ent2[reliable] < self.margin
            if reliable2.any():
                loss = ent2[reliable][reliable2].mean()
                loss.backward()
                self.ema = loss.item() if self.ema is None else 0.9 * self.ema + 0.1 * loss.item()
            with torch.no_grad():
                for p, e in zip(self.params, eps):
                    p.sub_(e)
            self.opt.step()
            self.opt.zero_grad()
            if self.ema is not None and self.ema < self.hp["sar_reset"]:   # model recovery
                self.model.load_state_dict(self.initial)
                self.opt.load_state_dict(self.initial_opt)
                self.ema = None
        return logits.detach()

    def __call__(self, x):
        """x [B,3,N] -> logits over the groups [B,G]."""
        if self.method == "sar":
            return self._sar(x)
        if self.method in ("t3a", "bn_t3a"):
            return self._t3a(x)
        if self.opt is None:
            with torch.no_grad():
                return self.group_logits(self.model(x))
        logits = self.group_logits(self.model(x))
        loss = _entropy(logits).mean()
        if self.method == "shot_im":
            pm = logits.softmax(1).mean(0)
            loss = loss + (pm * torch.log(pm + 1e-5)).sum()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return logits.detach()
