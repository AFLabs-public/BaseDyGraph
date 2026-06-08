"""
Propagation-delay (lead-lag) graph scorer.

Drop-in alternative to DynamicGraphScorer with the same contract:
    forward(h, state_ids) -> attn,  h: (B, T, N, D) -> (B, T, H, N, N)

The query comes from the current step h_t; the keys and values come from each
node's recent window h_{t-S+1 : t}. An edge score A_t[i, j] therefore compares
node i's present against node j's recent trajectory, giving directional, lagged
coupling that a contemporaneous Q_t K_t^T scorer cannot represent.

The window produces S scores per (i, j) pair, one per lag, collapsed to a single
edge weight by `lag_aggregation`:
    "softmax" : softmax-weighted sum over lags (soft lag selection)
    "max"     : strongest match over lags (hard lag selection)
    "mean"    : average over lags

Causal: lags only look backward and front padding is masked out.
"""

import math
import torch
import torch.nn as nn


class PropagationDelayGraphScorer(nn.Module):
    def __init__(self, cfg, window_size: int = 4, lag_aggregation: str = "softmax") -> None:
        super().__init__()
        if cfg.graph_hidden_dim % cfg.num_edge_heads != 0:
            raise ValueError("graph_hidden_dim must be divisible by num_edge_heads")
        if lag_aggregation not in ("softmax", "max", "mean"):
            raise ValueError("lag_aggregation must be one of: softmax, max, mean")

        self.cfg = cfg
        self.num_heads = cfg.num_edge_heads
        self.head_dim = cfg.graph_hidden_dim // cfg.num_edge_heads
        self.window_size = window_size
        self.lag_aggregation = lag_aggregation

        self.q_proj = nn.Linear(cfg.d_model, cfg.graph_hidden_dim)
        self.k_proj = nn.Linear(cfg.d_model, cfg.graph_hidden_dim)
        self.dropout = nn.Dropout(cfg.spatial_dropout)

    def _windowed_keys(self, k: torch.Tensor) -> torch.Tensor:
        """
        k: (B, T, H, N, hd) -> windowed keys (B, T, H, N, S, hd) where index s holds
        the key from timestep (t - s). Front-padded with zeros; pad mask returned too.
        """
        b, t, h, n, hd = k.shape
        s = self.window_size
        pad = torch.zeros(b, s - 1, h, n, hd, device=k.device, dtype=k.dtype)
        kpad = torch.cat([pad, k], dim=1)                       # (B, T+S-1, H, N, hd)

        # for output time t and lag s, source index in kpad is (t + (S-1) - s)
        base = torch.arange(s - 1, s - 1 + t, device=k.device)  # (T,)
        lags = torch.arange(s, device=k.device)                 # (S,)
        idx = base[:, None] - lags[None, :]                     # (T, S) indices into kpad
        kwin = kpad[:, idx]                                     # (B, T, S, H, N, hd)
        kwin = kwin.permute(0, 1, 3, 4, 2, 5)                   # (B, T, H, N, S, hd)

        valid = (idx >= (s - 1)).to(k.dtype)                    # (T, S): 1 if not padding
        return kwin, valid

    def forward(
        self,
        h: torch.Tensor,
        state_ids: torch.Tensor,
        return_lag_scores: bool = False,
    ) -> torch.Tensor:
        b, t, n, d = h.shape
        s = self.window_size

        q = self.q_proj(h).view(b, t, n, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = self.k_proj(h).view(b, t, n, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        kwin, valid = self._windowed_keys(k)                    # (B,T,H,N,S,hd), (T,S)

        # scores[b,t,h,i,j,s] = q[b,t,h,i] . kwin[b,t,h,j,s] / sqrt(hd)
        scores = torch.einsum("bthid,bthjsd->bthijs", q, kwin) / math.sqrt(self.head_dim)

        # mask padded lags (depend only on t, s) -> broadcast over i, j
        pad = (valid == 0).view(1, t, 1, 1, 1, s)
        scores = scores.masked_fill(pad, float("-inf"))

        attn_per_lag = scores

        if self.lag_aggregation == "max":
            logits = attn_per_lag.max(dim=-1).values                       # (B,T,H,N,N)
        elif self.lag_aggregation == "mean":
            safe = attn_per_lag.masked_fill(pad, 0.0)
            counts = valid.view(1, t, 1, 1, 1, s).sum(-1).clamp_min(1.0)    # (1,T,1,1,1)
            logits = safe.sum(-1) / counts.squeeze(-1)
        else:  # softmax over lags
            w = torch.softmax(attn_per_lag, dim=-1)                         # (B,T,H,N,N,S)
            vals = attn_per_lag.masked_fill(pad, 0.0)
            logits = (w * vals).sum(-1)

        if self.cfg.symmetric_graph:
            logits = 0.5 * (logits + logits.transpose(-1, -2))

        attn = torch.softmax(logits, dim=-1)                                # over neighbours j
        attn = self.dropout(attn)

        if self.cfg.add_self_loops:
            eye = torch.eye(n, device=attn.device, dtype=attn.dtype).view(1, 1, 1, n, n)
            attn = attn + eye
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        if return_lag_scores:
            # per-lag scores (B,T,H,N,N,S), padded lags at -inf; argmax over the
            # last axis identifies which lag carries each edge.
            return attn, attn_per_lag
        return attn
