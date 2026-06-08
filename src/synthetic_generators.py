"""
Synthetic discrete spatiotemporal data generators.

Two generators share one base class:

  ContemporaneousGraphGenerator
      A node's next token depends on its neighbours' states at the same
      timestep. The recoverable structure is a regime-switched contemporaneous
      graph W^(r).

  LeadLagGraphGenerator
      Adds a lagged coupling term: a node's next token also depends on its
      neighbours' states k steps earlier, through a separate lagged graph
      W_lag^(r). A propagation-delay scorer can recover this; a contemporaneous
      scorer cannot.

Both emit the same dict so they are interchangeable downstream:
    state_ids      (B, N, T)  int64 tokens
    regimes        (B, T)     int64 regime path
    regime_graphs  (R, N, N)  contemporaneous ground-truth graph per regime
    (LeadLag also adds)
    lag_graphs     (R, N, N)  lagged ground-truth graph per regime
    lag            int        the lag k used

Generative law (logits for node i choosing next token k):

    contemporaneous : self_i[s_{i,t}, k]
                    + sum_j W^(r_t)[i,j] * B[s_{j,t}, k]
                    + regime_bias[r_t, i, k]
                    + shock

    lead-lag (adds) : + sum_j W_lag^(r_t)[i,j] * B_lag[s_{j,t-k}, k]

All randomness flows through a single torch.Generator for reproducibility.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

@dataclass
class SyntheticGraphDataConfig:
    num_samples: int = 512
    seq_len: int = 64
    num_nodes: int = 8
    num_states: int = 16
    num_regimes: int = 3
    self_transition_scale: float = 2.5
    spatial_scale: float = 1.25
    regime_scale: float = 0.75
    regime_stickiness: float = 0.92
    sparsity: float = 0.5
    shock_prob: float = 0.02
    shock_scale: float = 2.0
    seed: int = 7

    # lead-lag only (ignored by the contemporaneous generator)
    lag: int = 3                       # how many steps back the lagged coupling reaches
    lag_spatial_scale: float = 1.25    # strength of the lagged term
    lag_sparsity: float = 0.5          # sparsity of the lagged graph
    keep_contemporaneous: bool = True  # if False, coupling is purely lagged


# ------------------------------------------------------------------
# Shared base
# ------------------------------------------------------------------

class _BaseGraphGenerator:
    """Shared parameter construction and sampling scaffold.

    Subclasses implement `_coupling_logits(t, states, regime)` to add their
    neighbour-coupling contribution; everything else (self-transition, regime
    bias, shocks, regime path, batching) is shared.
    """

    def __init__(self, cfg: SyntheticGraphDataConfig) -> None:
        self.cfg = cfg
        self.g = torch.Generator().manual_seed(cfg.seed)
        self.N = cfg.num_nodes
        self.K = cfg.num_states
        self.R = cfg.num_regimes

        self.self_transitions = self._make_self_transitions()      # (N, K, K)
        self.spatial_kernel = self._make_spatial_kernel()          # (K, K)
        self.regime_bias = cfg.regime_scale * torch.randn(
            self.R, self.N, self.K, generator=self.g
        )                                                          # (R, N, K)
        self.regime_transition = self._make_regime_transition()    # (R, R)
        self.regime_graphs = self._make_graph(cfg.sparsity)        # (R, N, N)

    # -- parameter builders ----------------------------------------

    def _make_self_transitions(self) -> torch.Tensor:
        a = 0.25 * torch.randn(self.N, self.K, self.K, generator=self.g)
        return a + self.cfg.self_transition_scale * torch.eye(self.K).unsqueeze(0)

    def _make_spatial_kernel(self) -> torch.Tensor:
        b = 0.35 * torch.randn(self.K, self.K, generator=self.g)
        return self.cfg.spatial_scale * b

    def _make_regime_transition(self) -> torch.Tensor:
        off = (1.0 - self.cfg.regime_stickiness) / max(self.R - 1, 1)
        mat = torch.full((self.R, self.R), off)
        for r in range(self.R):
            mat[r, r] = self.cfg.regime_stickiness
        return mat

    def _make_graph(self, sparsity: float) -> torch.Tensor:
        """Row-stochastic, non-negative, zero-diagonal, asymmetric graph per regime."""
        graphs = []
        for _ in range(self.R):
            raw = torch.rand(self.N, self.N, generator=self.g)
            mask = (raw > sparsity).float()
            w = torch.rand(self.N, self.N, generator=self.g) * mask
            w.fill_diagonal_(0.0)
            w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            graphs.append(w)
        return torch.stack(graphs, dim=0)

    # -- to be provided by subclasses ------------------------------

    def _coupling_logits(self, t: int, states: torch.Tensor, regime: int) -> torch.Tensor:
        """Return the neighbour-coupling contribution to the logits, shape (N, K).

        `states` is the full (N, T) buffer sampled so far; only entries at indices
        <= t may be read (causality is the subclass's responsibility).
        """
        raise NotImplementedError

    def extra_outputs(self) -> Dict[str, torch.Tensor]:
        """Subclass hook to add ground-truth tensors to the output dict."""
        return {}

    @torch.no_grad()
    def oracle_decomposition(self, data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Upper bound on how much a dynamic graph can beat a static one on this data.

        Re-scores every realised transition under two graph choices, holding the
        self-transition, regime-bias, and kernel terms fixed:
          dynamic oracle : the true per-regime graph at each step
          static  oracle : the mean graph (averaged over regimes) at every step
        The gap is the largest advantage any dynamic model could have over any
        static one here; if it is near zero, dynamic structure cannot help on this
        data. Uses the contemporaneous regime graphs only; lead-lag structure is a
        separate ablation.
        """
        state_ids = data["state_ids"]                 # (B, N, T)
        regimes = data["regimes"]                      # (B, T)
        B, N, T = state_ids.shape
        node_idx = torch.arange(N)

        W = self.regime_graphs                         # (R, N, N) true per-regime graphs
        W_mean = W.mean(dim=0)                         # (N, N) best single fixed graph

        def score(use_true_graph: bool):
            acc_num = acc_den = 0
            nll_sum = nll_n = 0.0
            for b in range(B):
                s = state_ids[b]                       # (N, T)
                r = regimes[b]
                for t in range(T - 1):
                    rt = int(r[t])
                    cur = s[:, t]
                    self_eff = self.self_transitions[node_idx, cur]
                    G = W[rt] if use_true_graph else W_mean
                    coupling = G @ self.spatial_kernel[cur]
                    logits = self_eff + coupling + self.regime_bias[rt]
                    p = torch.softmax(logits, dim=-1)
                    realised = s[:, t + 1]
                    acc_num += int((p.argmax(-1) == realised).sum())
                    acc_den += N
                    pr = p.gather(-1, realised.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
                    nll_sum += float(-pr.log().sum())
                    nll_n += N
            return acc_num / acc_den, nll_sum / nll_n

        dyn_acc, dyn_nll = score(True)
        sta_acc, sta_nll = score(False)
        return {
            "dynamic_oracle_acc": dyn_acc,
            "static_oracle_acc": sta_acc,
            "dynamic_headroom_acc": dyn_acc - sta_acc,
            "dynamic_oracle_nll": dyn_nll,
            "static_oracle_nll": sta_nll,
            "dynamic_headroom_nll": sta_nll - dyn_nll,
        }

    # -- shared sampling -------------------------------------------

    def _sample_regime_path(self, seq_len: int) -> torch.Tensor:
        r = torch.zeros(seq_len, dtype=torch.long)
        r[0] = torch.randint(0, self.R, (1,), generator=self.g)
        for t in range(1, seq_len):
            r[t] = torch.multinomial(
                self.regime_transition[r[t - 1]], 1, generator=self.g
            ).squeeze(0)
        return r

    def _sample_sequence(self, return_probs: bool = False):
        cfg = self.cfg
        s = torch.zeros(self.N, cfg.seq_len, dtype=torch.long)
        s[:, 0] = torch.randint(0, self.K, (self.N,), generator=self.g)
        regimes = self._sample_regime_path(cfg.seq_len)
        node_idx = torch.arange(self.N)
        probs_seq = torch.zeros(self.N, cfg.seq_len - 1, self.K) if return_probs else None

        for t in range(cfg.seq_len - 1):
            r_t = int(regimes[t])
            current = s[:, t]                                       # (N,)

            self_effect = self.self_transitions[node_idx, current]  # (N, K)
            coupling = self._coupling_logits(t, s, r_t)             # (N, K)
            logits = self_effect + coupling + self.regime_bias[r_t]

            if cfg.shock_prob > 0:
                hit = (torch.rand(self.N, generator=self.g) < cfg.shock_prob).float().unsqueeze(-1)
                noise = cfg.shock_scale * torch.randn(self.N, self.K, generator=self.g)
                logits = logits + hit * noise

            probs = torch.softmax(logits, dim=-1)
            if return_probs:
                probs_seq[:, t] = probs                            # true next-token dist at step t
            s[:, t + 1] = torch.multinomial(probs, 1, generator=self.g).squeeze(-1)

        if return_probs:
            return s, regimes, probs_seq
        return s, regimes

    def generate(self, return_true_probs: bool = False) -> Dict[str, torch.Tensor]:
        seqs, paths, probs = [], [], []
        for _ in range(self.cfg.num_samples):
            if return_true_probs:
                seq, reg, pr = self._sample_sequence(return_probs=True)
                probs.append(pr)
            else:
                seq, reg = self._sample_sequence()
            seqs.append(seq)
            paths.append(reg)

        out = {
            "state_ids": torch.stack(seqs, dim=0),        # (B, N, T)
            "regimes": torch.stack(paths, dim=0),         # (B, T)
            "regime_graphs": self.regime_graphs.clone(),  # (R, N, N)
            "self_transitions": self.self_transitions.clone(),
            "spatial_kernel": self.spatial_kernel.clone(),
            "regime_bias": self.regime_bias.clone(),
        }
        if return_true_probs:
            out["true_probs"] = torch.stack(probs, dim=0)  # (B, N, T-1, K): true next-token dist
        out.update(self.extra_outputs())
        return out


# ------------------------------------------------------------------
# Contemporaneous generator (original law)
# ------------------------------------------------------------------

class ContemporaneousGraphGenerator(_BaseGraphGenerator):
    """Neighbour coupling uses neighbour states at the SAME timestep."""

    def _coupling_logits(self, t: int, states: torch.Tensor, regime: int) -> torch.Tensor:
        w_t = self.regime_graphs[regime]                # (N, N)
        source = self.spatial_kernel[states[:, t]]      # (N, K): effect of each j's state at t
        return w_t @ source                             # (N, K): aggregated over neighbours j


# ------------------------------------------------------------------
# Lead-lag generator (adds lagged coupling)
# ------------------------------------------------------------------

class LeadLagGraphGenerator(_BaseGraphGenerator):
    """Adds a lagged coupling term: neighbour states k steps in the past, through a
    separate lagged graph and kernel. Optionally drops the contemporaneous term so
    coupling is purely lead-lag.

    Recoverable ground truth:
        regime_graphs (R,N,N) contemporaneous  (present iff keep_contemporaneous)
        lag_graphs    (R,N,N) lagged
    """

    def __init__(self, cfg: SyntheticGraphDataConfig) -> None:
        super().__init__(cfg)
        self.lag = cfg.lag
        self.lag_graphs = self._make_graph(cfg.lag_sparsity)        # (R, N, N)
        b = 0.35 * torch.randn(self.K, self.K, generator=self.g)
        self.lag_kernel = cfg.lag_spatial_scale * b                 # (K, K)

    def _coupling_logits(self, t: int, states: torch.Tensor, regime: int) -> torch.Tensor:
        out = torch.zeros(self.N, self.K)

        if self.cfg.keep_contemporaneous:
            w_t = self.regime_graphs[regime]
            out = out + w_t @ self.spatial_kernel[states[:, t]]

        t_lag = t - self.lag
        if t_lag >= 0:                                  # causal: only read the real past
            w_lag = self.lag_graphs[regime]
            lagged_source = self.lag_kernel[states[:, t_lag]]   # (N, K) from states k steps back
            out = out + w_lag @ lagged_source

        return out

    def extra_outputs(self) -> Dict[str, torch.Tensor]:
        return {
            "lag_graphs": self.lag_graphs.clone(),
            "lag": torch.tensor(self.lag),
        }


# ------------------------------------------------------------------
# Convenience
# ------------------------------------------------------------------

def build_generator(cfg: SyntheticGraphDataConfig, lead_lag: bool = False) -> _BaseGraphGenerator:
    """Return the lead-lag generator if `lead_lag` else the contemporaneous one."""
    return LeadLagGraphGenerator(cfg) if lead_lag else ContemporaneousGraphGenerator(cfg)


# ------------------------------------------------------------------
# Data diagnostics
# ------------------------------------------------------------------

def data_diagnostics(data: Dict[str, torch.Tensor], num_states: Optional[int] = None,
                     generator: Optional["_BaseGraphGenerator"] = None,
                     verbose: bool = True) -> Dict[str, float]:
    """Quantify how (un)predictable the generated data is, so model accuracy can be
    read against the achievable ceiling rather than against 1.0.

    Pass a dict from generate(return_true_probs=True) to also get the oracle
    measures (ceiling, oracle NLL, true conditional entropy); without true_probs
    only the empirical measures are computed.

    Pass `generator` (the object that produced the data) to also get the
    static-vs-dynamic oracle decomposition. If dynamic_headroom_acc is near zero,
    time-varying structure cannot help on this data.

    Returns a dict and (if verbose) prints a short report. Key fields:
      achievable_acc_ceiling : best top-1 accuracy attainable (argmax of the true
                               distribution vs the realised token).
      oracle_nll             : lowest achievable cross-entropy (mean -log p_true).
      true_cond_entropy      : mean entropy of the true per-step distribution.
      H_next_given_current   : empirical H(s_{t+1} | s_t), predictability from the
                               current token alone (ignores graph/regime).
      structure_information  : H_next_given_current - true_cond_entropy; the
                               predictive information graph + regime add beyond the
                               current token. A static graph captures most of it;
                               the dynamic-only slice is the oracle gap below.
      dynamic_headroom_acc   : static vs dynamic oracle accuracy gap (needs `generator`).
      persistence_rate, state_entropy, state_balance_ratio, regime_balance, mean_regime_dwell
    """
    state_ids = data["state_ids"]                      # (B, N, T)
    B, N, T = state_ids.shape
    K = num_states or int(state_ids.max().item()) + 1
    rep: Dict[str, float] = {}

    nxt = state_ids[:, :, 1:]
    cur = state_ids[:, :, :-1]

    # persistence
    rep["persistence_rate"] = (nxt == cur).float().mean().item()

    # marginal state distribution
    counts = torch.bincount(state_ids.reshape(-1), minlength=K).float()
    freq = counts / counts.sum()
    nz = freq[freq > 0]
    rep["state_entropy"] = float(-(nz * nz.log()).sum())
    rep["uniform_entropy"] = float(torch.log(torch.tensor(float(K))))
    rep["state_balance_ratio"] = float(freq.max() / freq[freq > 0].min())

    # H(next | current) — sequence-only predictability (empirical)
    joint = torch.zeros(K, K)
    flat_c, flat_n = cur.reshape(-1), nxt.reshape(-1)
    joint.index_put_((flat_c, flat_n), torch.ones_like(flat_c, dtype=torch.float), accumulate=True)
    row = joint.sum(1, keepdim=True).clamp_min(1)
    p_n_given_c = joint / row
    p_c = joint.sum(1) / joint.sum()
    ent_per_c = -(torch.where(p_n_given_c > 0, p_n_given_c * p_n_given_c.log(),
                              torch.zeros_like(p_n_given_c))).sum(1)
    rep["H_next_given_current"] = float((p_c * ent_per_c).sum())

    # oracle measures (need the true per-step distribution)
    if "true_probs" in data:
        p = data["true_probs"]                          # (B, N, T-1, K)
        realised = nxt                                  # (B, N, T-1)
        rep["achievable_acc_ceiling"] = (p.argmax(-1) == realised).float().mean().item()
        p_realised = p.gather(-1, realised.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)
        rep["oracle_nll"] = float(-p_realised.log().mean())
        rep["true_cond_entropy"] = float(-(p * p.clamp_min(1e-12).log()).sum(-1).mean())
        rep["structure_information"] = rep["H_next_given_current"] - rep["true_cond_entropy"]

    # regimes
    if "regimes" in data:
        regimes = data["regimes"]
        R = int(regimes.max().item()) + 1
        rb = torch.bincount(regimes.reshape(-1), minlength=R).float()
        rep["regime_balance"] = (rb / rb.sum()).tolist()
        switches = (regimes[:, 1:] != regimes[:, :-1]).float().mean().item()
        rep["mean_regime_dwell"] = (1.0 / switches) if switches > 0 else float("inf")

    # static-vs-dynamic oracle decomposition (needs the generator's params)
    if generator is not None:
        rep.update(generator.oracle_decomposition(data))

    if verbose:
        print("=" * 56)
        print("DATA DIAGNOSTICS")
        print("=" * 56)
        order = ["achievable_acc_ceiling", "persistence_rate", "oracle_nll",
                 "true_cond_entropy", "H_next_given_current", "structure_information",
                 "state_entropy", "uniform_entropy", "state_balance_ratio",
                 "mean_regime_dwell"]
        for k in order:
            if k in rep:
                print(f"  {k:<24}: {rep[k]:+.4f}")
        if "regime_balance" in rep:
            print(f"  {'regime_balance':<24}: {[round(x,3) for x in rep['regime_balance']]}")
        if "dynamic_headroom_acc" in rep:
            print("-" * 56)
            print("  STATIC vs DYNAMIC oracle (the dynamic-only headroom):")
            print(f"  {'static_oracle_acc':<24}: {rep['static_oracle_acc']:+.4f}")
            print(f"  {'dynamic_oracle_acc':<24}: {rep['dynamic_oracle_acc']:+.4f}")
            print(f"  {'dynamic_headroom_acc':<24}: {rep['dynamic_headroom_acc']:+.4f}")
            print(f"  {'dynamic_headroom_nll':<24}: {rep['dynamic_headroom_nll']:+.4f}")
        print("=" * 56)
        if "achievable_acc_ceiling" in rep:
            print(f"  -> read model_acc against the {rep['achievable_acc_ceiling']:.3f} ceiling, not 1.0")
            print(f"  -> read model nll against the {rep['oracle_nll']:.3f} oracle, not 0.0")
        print(f"  -> structure_information = TOTAL info graph+regime add (static gets most of it)")
        if "dynamic_headroom_acc" in rep:
            print(f"  -> dynamic_headroom = what ONLY a dynamic graph can get "
                  f"(+{rep['dynamic_headroom_acc']:.3f} acc max)")
    return rep
