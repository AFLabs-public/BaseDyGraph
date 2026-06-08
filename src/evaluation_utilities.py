"""
Evaluation utilities for the discrete spatiotemporal graph model.

Import as a module:  from evaluation_utilities import *

Provides:
  - low-level metric helpers: off_diagonal_mask, masked_pearson, binary_auroc
  - graph-recovery: evaluate_regime_graph_recovery (diagonal-masked, regime-averaged,
    correlation + edge-AUROC; MSE intentionally omitted)
  - prediction baselines: persistence_baseline_accuracy,
    majority_class_baseline_accuracy, balanced_accuracy
  - full_evaluation_report: one pass over a loader; headline = lift_over_persistence

Assumes the model interface model(state_ids) -> {"next_state_logits", "graph_attn", ...}
and batches {"state_ids": (B,N,T), "regimes": (B,T)}. See the lab notes for the
rationale behind each metric choice (why correlation over MSE, why regime-averaging,
why lift-over-persistence is the headline).
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Low-level metric helpers
# ---------------------------------------------------------------------------

def off_diagonal_mask(n: int, device: torch.device) -> torch.Tensor:
    """Boolean (n, n) mask that is True off the diagonal."""
    return ~torch.eye(n, dtype=torch.bool, device=device)


def masked_pearson(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pearson correlation between two 1-D tensors (shift/scale invariant)."""
    a = a - a.mean()
    b = b - b.mean()
    denom = a.std() * b.std()
    if denom < 1e-12:
        return torch.tensor(0.0, device=a.device)
    return (a * b).mean() / denom


def binary_auroc(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    AUROC via the rank (Mann-Whitney U) formula.

    scores: continuous 1-D tensor (here: off-diagonal attention weights)
    labels: 1-D tensor in {0, 1} (here: whether a true edge exists)

    Returns P(score of a true edge > score of a non-edge). 1.0 = perfect,
    0.5 = chance, NaN if one class is absent. Tie-averaging is unnecessary for
    continuous attention scores.
    """
    labels = labels.bool()
    n_pos = labels.sum()
    n_neg = (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return torch.tensor(float("nan"), device=scores.device)

    order = torch.argsort(scores)
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=scores.dtype, device=scores.device)
    return (ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# ---------------------------------------------------------------------------
# Corrected graph-recovery evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_regime_graph_recovery(
    model,
    batch: Dict[str, torch.Tensor],
    true_regime_graphs: torch.Tensor,   # (R, N, N)
    device: Optional[torch.device] = None,
    regime_average: bool = True,
) -> Dict[str, float]:
    """
    Compare the model's inferred time-varying graph to the ground-truth
    regime graphs. Diagonal-masked, regime-averaged, correlation + edge-AUROC.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    state_ids = batch["state_ids"].to(device)
    if "regimes" not in batch:
        raise KeyError("batch must contain 'regimes' (B, T) for recovery evaluation")
    regimes = batch["regimes"].to(device)              # (B, T)

    out = model.backbone(state_ids)
    attn = out.get("graph_attn", None)
    if attn is None:
        return {"graph_corr": float("nan"), "graph_auroc": float("nan")}

    attn = attn.mean(dim=2)                             # (B, T, N, N): average over heads
    W = true_regime_graphs.to(device).float()          # (R, N, N)
    B, T, N, _ = attn.shape
    mask = off_diagonal_mask(N, device)

    def pair_metrics(a_graph: torch.Tensor, regime: int):
        a_off = a_graph[mask]
        t_off = W[regime][mask]
        corr = masked_pearson(a_off, t_off).item()
        auroc = binary_auroc(a_off, (t_off > 0).float()).item()
        return corr, auroc

    corrs, aurocs = [], []
    for b in range(B):
        rseq = regimes[b]                              # (T,)
        for r in torch.unique(rseq).tolist():
            steps = rseq == r
            if regime_average:
                a_graph = attn[b][steps].mean(dim=0)   # (N, N)
                c, au = pair_metrics(a_graph, r)
                corrs.append(c); aurocs.append(au)
            else:
                for a_graph in attn[b][steps]:         # per-timestep (noisier)
                    c, au = pair_metrics(a_graph, r)
                    corrs.append(c); aurocs.append(au)

    return {
        "graph_corr": float(np.nanmean(corrs)),
        "graph_auroc": float(np.nanmean(aurocs)),
    }


@torch.no_grad()
def evaluate_deviation_recovery(
    model,
    batch: Dict[str, torch.Tensor],
    regime_graphs: torch.Tensor,         # (R, N, N)
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """Recover the *regime-specific* structure, isolated from the shared component.

    The plain regime-averaged corr over-credits the part of the graph that is common
    to all regimes: a static graph (one matrix for every regime) matches the mean
    graph and scores high without representing any regime difference at all. This
    metric removes that confound by subtracting the mean-over-regimes from BOTH the
    inferred and the true graphs, then correlating the residuals (the deviations
    W^(r) - mean_r W). A static scorer's residual is ~0 -> ~0 here by construction;
    only a model whose graph varies by regime can score above chance. This is the
    metric that actually isolates the dynamic graph's contribution.
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    state_ids = batch["state_ids"].to(device)
    regimes = batch["regimes"].to(device)
    out = model.backbone(state_ids)
    attn = out.get("graph_attn", None)
    if attn is None:
        return {"deviation_corr": float("nan")}

    attn = attn.mean(dim=2)                              # (B, T, N, N)
    W = regime_graphs.to(device).float()                # (R, N, N)
    R, N, _ = W.shape
    mask = off_diagonal_mask(N, device)

    true_dev = W - W.mean(dim=0, keepdim=True)          # (R, N, N): true regime deviations

    # mean inferred graph per regime, pooled over the batch (so the per-model mean is
    # estimated across all sequences, mirroring how the static mean would form)
    sums = torch.zeros(R, N, N, device=device)
    counts = torch.zeros(R, device=device)
    B, T = attn.shape[0], attn.shape[1]
    for b in range(B):
        rseq = regimes[b]
        for r in torch.unique(rseq).tolist():
            steps = rseq == r
            sums[r] += attn[b][steps].sum(dim=0)
            counts[r] += int(steps.sum())
    seen = counts > 0
    inferred = torch.zeros_like(sums)
    inferred[seen] = sums[seen] / counts[seen].view(-1, 1, 1)
    inferred_dev = inferred - inferred[seen].mean(dim=0, keepdim=True)

    corrs = []
    for r in range(R):
        if not seen[r]:
            continue
        corrs.append(masked_pearson(inferred_dev[r][mask], true_dev[r][mask]).item())

    return {"deviation_corr": float(np.nanmean(corrs)) if corrs else float("nan")}


# ---------------------------------------------------------------------------
# Prediction baselines
# ---------------------------------------------------------------------------

def persistence_baseline_accuracy(state_ids: torch.Tensor) -> float:
    """Accuracy of predicting s_{t+1} = s_t. The key baseline to beat."""
    nxt = state_ids[:, :, 1:]
    cur = state_ids[:, :, :-1]
    return (nxt == cur).float().mean().item()


def majority_class_baseline_accuracy(state_ids: torch.Tensor, num_states: int) -> float:
    """Accuracy of always predicting the globally most frequent state."""
    counts = torch.bincount(state_ids.reshape(-1), minlength=num_states)
    majority = counts.argmax()
    target = state_ids[:, :, 1:]
    return (target == majority).float().mean().item()


def balanced_accuracy(preds: torch.Tensor, target: torch.Tensor, num_states: int) -> float:
    """Mean per-class recall over classes that actually appear."""
    recalls = []
    for k in range(num_states):
        in_class = target == k
        if in_class.any():
            recalls.append((preds[in_class] == k).float().mean().item())
    return float(np.mean(recalls)) if recalls else float("nan")


# ---------------------------------------------------------------------------
# Full evaluation report
# ---------------------------------------------------------------------------

@torch.no_grad()
def full_evaluation_report(
    model,
    loader,
    num_states: int,
    true_regime_graphs: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    # accumulators
    correct = total = 0
    persist_correct = persist_total = 0
    nll_sum = nll_tokens = 0.0
    per_class_correct = torch.zeros(num_states)
    per_class_total = torch.zeros(num_states)
    corrs, aurocs, dev_corrs = [], [], []

    for batch in loader:
        state_ids = batch["state_ids"].to(device)
        out = model(state_ids)
        logits = out["next_state_logits"]              # (B, N, T-1, K)
        target = state_ids[:, :, 1:]                   # (B, N, T-1)
        preds = logits.argmax(dim=-1)

        correct += (preds == target).sum().item()
        total += target.numel()

        nxt, cur = state_ids[:, :, 1:], state_ids[:, :, :-1]
        persist_correct += (nxt == cur).sum().item()
        persist_total += nxt.numel()

        flat_logits = logits.reshape(-1, num_states)
        flat_target = target.reshape(-1)
        nll_sum += F.cross_entropy(flat_logits, flat_target, reduction="sum").item()
        nll_tokens += flat_target.numel()

        p, t = preds.reshape(-1).cpu(), target.reshape(-1).cpu()
        for k in range(num_states):                    # confusion counts for balanced acc
            in_class = t == k
            per_class_total[k] += in_class.sum()
            per_class_correct[k] += (p[in_class] == k).sum()

        if true_regime_graphs is not None and "regimes" in batch:
            rec = evaluate_regime_graph_recovery(model, batch, true_regime_graphs, device)
            corrs.append(rec["graph_corr"]); aurocs.append(rec["graph_auroc"])
            dev = evaluate_deviation_recovery(model, batch, true_regime_graphs, device)
            dev_corrs.append(dev["deviation_corr"])

    seen = per_class_total > 0
    bacc = (per_class_correct[seen] / per_class_total[seen]).mean().item()

    report = {
        "persistence_acc": persist_correct / persist_total,
        "model_acc": correct / total,
        "lift_over_persistence": correct / total - persist_correct / persist_total,
        "balanced_acc": bacc,
        "nll": nll_sum / nll_tokens,
    }
    if corrs:
        report["graph_corr"] = float(np.nanmean(corrs))
        report["graph_auroc"] = float(np.nanmean(aurocs))
    if dev_corrs:
        report["deviation_corr"] = float(np.nanmean(dev_corrs))

    width = max(len(k) for k in report)
    print("=" * (width + 12))
    for k, v in report.items():
        print(f"{k:<{width}} : {v:+.4f}")
    print("=" * (width + 12))
    return report

