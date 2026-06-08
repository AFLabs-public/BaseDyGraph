import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utilities import *
from typing import Optional, Dict, List, Tuple


def _sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax: Euclidean projection onto the probability simplex. Produces
    exact zeros (sparse rows) and is differentiable almost everywhere."""
    z = z - z.max(dim=dim, keepdim=True).values
    zs, _ = torch.sort(z, dim=dim, descending=True)
    rng = torch.arange(1, z.size(dim) + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.dim(); shape[dim] = -1
    rng = rng.view(shape)
    cssv = zs.cumsum(dim) - 1
    cond = (zs - cssv / rng) > 0
    k = cond.sum(dim=dim, keepdim=True)
    tau = cssv.gather(dim, (k - 1).clamp_min(0)) / k.to(z.dtype)
    return torch.clamp(z - tau, min=0)


def _entmax15(z: torch.Tensor, dim: int = -1, n_iter: int = 30) -> torch.Tensor:
    """1.5-entmax via bisection. Sits between softmax (dense) and sparsemax
    (sparse) in how aggressively it zeros small entries."""
    z = (z - z.max(dim=dim, keepdim=True).values) / 2.0
    tau_lo = z.max(dim=dim, keepdim=True).values - 1.0
    tau_hi = z.max(dim=dim, keepdim=True).values - (1.0 / z.size(dim)) ** 0.5
    for _ in range(n_iter):
        tau = (tau_lo + tau_hi) / 2
        p = torch.clamp(z - tau, min=0) ** 2
        Z = p.sum(dim=dim, keepdim=True)
        tau_lo = torch.where(Z < 1, tau, tau_lo)
        tau_hi = torch.where(Z >= 1, tau, tau_hi)
    p = torch.clamp(z - (tau_lo + tau_hi) / 2, min=0) ** 2
    return p / p.sum(dim=dim, keepdim=True).clamp_min(1e-12)


class IdentityTemporalModule(nn.Module):
    def forward(self, x):
        return x

class IdentitySpatialModule(nn.Module):
    def forward(
        self,
        h: torch.Tensor,
        attn: Optional[torch.Tensor] = None,
        e: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return h

def build_temporal_module(cfg: ModelConfig) -> nn.Module:
    if cfg.temporal_module_type == "none":
        return IdentityTemporalModule()
    elif cfg.temporal_module_type == "transformer":
        return PerNodeTemporalEncoder(cfg)
    else:
        raise ValueError(f"Unknown temporal_module_type: {cfg.temporal_module_type}")


def build_spatial_components(
    cfg: ModelConfig,
) -> tuple[Optional[nn.Module], nn.Module]:
    if cfg.spatial_module_type == "none":
        return None, IdentitySpatialModule()
    elif cfg.spatial_module_type == "dynamic_graph":
        return DynamicGraphScorer(cfg), SpatialMessagePassing(cfg)
    elif cfg.spatial_module_type == "static_graph":
        return StaticGraphScorer(cfg), SpatialMessagePassing(cfg)
    elif cfg.spatial_module_type == "dynamic_base":
        return DynamicBaseGraphScorer(cfg), SpatialMessagePassing(cfg)
    else:
        raise ValueError(f"Unknown spatial_module_type: {cfg.spatial_module_type}")



# ------------------------------------------------------------
# Temporal encoder
# ------------------------------------------------------------

class PerNodeTemporalEncoder(nn.Module):
    """
    Shared causal transformer over node sequences.
    Input:  (B, N, T, D)
    Output: (B, N, T, D)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, cfg.max_seq_len)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.ff_mult * cfg.d_model,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_temporal_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, t, d = x.shape
        x = x.reshape(b * n, t, d)
        x = self.pos_enc(x)
        mask = causal_mask(t, x.device)
        x = self.encoder(x, mask=mask)
        x = x.reshape(b, n, t, d)
        return x


# ------------------------------------------------------------
# Dynamic graph inference at each time t
# ------------------------------------------------------------

class DynamicGraphScorer(nn.Module):
    """
    Builds A_t from contextual node embeddings H_t.

    Input:
        h: (B, T, N, D)
        state_ids: (B, N, T)
    Output:
        attn: (B, T, H, N, N)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.num_edge_heads
        head_dim = cfg.graph_hidden_dim // cfg.num_edge_heads
        if cfg.graph_hidden_dim % cfg.num_edge_heads != 0:
            raise ValueError("graph_hidden_dim must be divisible by num_edge_heads")
        self.head_dim = head_dim

        self.q_proj = nn.Linear(cfg.d_model, cfg.graph_hidden_dim)
        self.k_proj = nn.Linear(cfg.d_model, cfg.graph_hidden_dim)
        self.dropout = nn.Dropout(cfg.spatial_dropout)

        if cfg.use_state_pair_bias:
            self.state_pair_bias = nn.Parameter(
                torch.zeros(cfg.num_edge_heads, cfg.num_states, cfg.num_states)
            )
            nn.init.normal_(self.state_pair_bias, std=0.02)
        else:
            self.state_pair_bias = None

        # Independent per-edge gate for graph_activation="gated":
        #   A[i,j] = sigmoid((score - theta) / tau)
        # Edges are scored independently rather than competing through a softmax
        # over neighbours. theta is a learnable per-head threshold, tau a fixed
        # temperature; gate_row_normalise rescales rows afterwards.
        self.gate_theta = nn.Parameter(torch.zeros(cfg.num_edge_heads))
        self.gate_tau = getattr(cfg, "gate_tau", 0.5)
        self.gate_row_normalise = getattr(cfg, "gate_row_normalise", True)

        # Optional learnable base graph (H, N, N) added to the QK^T logits before
        # normalisation. It holds a fixed per-edge adjacency while QK^T learns the
        # time-varying deviation from it. Enabled by DynamicBaseGraphScorer; off here.
        self.use_base_graph = False
        self.base_graph = None

        # Residual gate for DynamicBaseGraphScorer. When enabled, the logits are
        #     base_logits + alpha * dynamic_logits
        # instead of base_logits + dynamic_logits. A small initial alpha starts
        # near the base graph and adds dynamic deviation over training. Mode
        # "none" leaves alpha = 1.0.
        self.dynamic_residual_gate = getattr(cfg, "dynamic_residual_gate", "none")
        self.dynamic_residual_init = float(getattr(cfg, "dynamic_residual_init", 1.0))
        self.dynamic_residual_learnable = bool(getattr(cfg, "dynamic_residual_learnable", True))
        # How alpha is applied:
        #   "logit"  : A = normalise(base_logits + alpha * dynamic_logits)
        #   "convex" : A = (1 - alpha) * normalise(base_logits)
        #                  + alpha * normalise(base_logits + dynamic_logits)
        # In convex mode alpha is the mixture weight between the base-only and
        # full dynamic graphs.
        self.dynamic_residual_mix = getattr(cfg, "dynamic_residual_mix", "logit")
        self.dynamic_residual_raw = None

    @staticmethod
    def _alpha_to_raw(alpha: float) -> float:
        # Map alpha in (0, 1) to the pre-sigmoid parameter. Clamp away from the
        # exact boundary for numerical safety.
        eps = 1e-6
        alpha = min(max(float(alpha), eps), 1.0 - eps)
        return math.log(alpha / (1.0 - alpha))

    def _make_dynamic_residual_parameter(self) -> None:
        """Create the alpha parameter. Called from DynamicBaseGraphScorer.__init__;
        kept separate so the plain DynamicGraphScorer carries no gate parameters."""
        mode = self.dynamic_residual_gate
        if mode not in {"none", "scalar", "per_head"}:
            raise ValueError(
                f"Unknown dynamic_residual_gate={mode!r}; expected 'none', 'scalar', or 'per_head'"
            )
        if self.dynamic_residual_mix not in {"logit", "convex"}:
            raise ValueError(
                f"Unknown dynamic_residual_mix={self.dynamic_residual_mix!r}; expected 'logit' or 'convex'"
            )
        if mode == "none":
            self.dynamic_residual_raw = None
            return

        raw_init = self._alpha_to_raw(self.dynamic_residual_init)
        shape = (1,) if mode == "scalar" else (self.num_heads,)
        raw = torch.full(shape, raw_init, dtype=torch.float32)
        if self.dynamic_residual_learnable:
            self.dynamic_residual_raw = nn.Parameter(raw)
        else:
            self.register_buffer("dynamic_residual_raw", raw, persistent=True)

    def dynamic_residual_alpha(self) -> torch.Tensor:
        """Return alpha for logging/use. Shape: scalar or (H,)."""
        if self.dynamic_residual_gate == "none" or self.dynamic_residual_raw is None:
            return torch.tensor(1.0, device=self.q_proj.weight.device)
        return torch.sigmoid(self.dynamic_residual_raw)

    def _alpha_view(self, logits: torch.Tensor) -> torch.Tensor:
        """Broadcast alpha to (1,1,H,1,1), or scalar-compatible shape."""
        alpha = self.dynamic_residual_alpha().to(device=logits.device, dtype=logits.dtype)
        if alpha.ndim == 0 or alpha.numel() == 1:
            return alpha.view(1, 1, 1, 1, 1)
        return alpha.view(1, 1, self.num_heads, 1, 1)

    def _base_logits_like(self, logits: torch.Tensor) -> torch.Tensor:
        return self.base_graph.view(1, 1, self.num_heads, logits.size(-1), logits.size(-1))

    def _combine_base_and_dynamic(self, dynamic_logits: torch.Tensor) -> torch.Tensor:
        """Combine base and dynamic logits into attention.

        Plain dynamic_graph: normalise(dynamic_logits).
        dynamic_base:
            gate='none'  -> normalise(base + dynamic)
            mix='logit'  -> normalise(base + alpha * dynamic)
            mix='convex' -> (1 - alpha) * normalise(base)
                            + alpha * normalise(base + dynamic)
        """
        if not self.use_base_graph or self.base_graph is None:
            return self._normalise(dynamic_logits)

        base_logits = self._base_logits_like(dynamic_logits)

        if self.dynamic_residual_gate == "none":
            return self._normalise(base_logits + dynamic_logits)

        alpha = self._alpha_view(dynamic_logits)

        if self.dynamic_residual_mix == "logit":
            return self._normalise(base_logits + alpha * dynamic_logits)

        if self.dynamic_residual_mix == "convex":
            a_base = self._normalise(base_logits.expand_as(dynamic_logits))
            a_dyn = self._normalise(base_logits + dynamic_logits)
            return (1.0 - alpha) * a_base + alpha * a_dyn

        raise RuntimeError(f"Unhandled dynamic_residual_mix={self.dynamic_residual_mix!r}")

    def _normalise(self, logits: torch.Tensor) -> torch.Tensor:
        """Normalise edge logits into per-row attention over neighbours (last dim).

        graph_activation (default 'softmax'):
          'softmax'   : dense; every neighbour gets non-zero weight.
          'sparsemax' : simplex projection with exact zeros, for a sparse graph.
          'entmax15'  : between softmax and sparsemax.
          'gated'     : independent per-edge sigmoid gate, no competition across
                        neighbours; row-normalised afterwards if gate_row_normalise.

        The sparse activations exist because the target graphs are mostly zeros,
        which softmax cannot represent.
        """
        act = getattr(self.cfg, "graph_activation", "softmax")
        if act == "softmax":
            return torch.softmax(logits, dim=-1)
        elif act == "sparsemax":
            return _sparsemax(logits, dim=-1)
        elif act == "entmax15":
            return _entmax15(logits, dim=-1)
        elif act == "gated":
            # independent per-edge gate; no competition across neighbours.
            # logits: (B,T,H,N,N); theta broadcast over the head axis.
            theta = self.gate_theta.view(1, 1, -1, 1, 1)
            gate = torch.sigmoid((logits - theta) / self.gate_tau)
            if self.gate_row_normalise:
                # rescale row mass without re-imposing competition between edges
                gate = gate / gate.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            return gate
        else:
            raise ValueError(f"Unknown graph_activation: {act}")

    def forward(self, h: torch.Tensor, state_ids: torch.Tensor) -> torch.Tensor:
        b, t, n, d = h.shape
        q = self.q_proj(h).view(b, t, n, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = self.k_proj(h).view(b, t, n, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)  # (B,T,H,N,N)

        if self.state_pair_bias is not None:
            # state_ids: (B, N, T) -> (B, T, N)
            s = state_ids.permute(0, 2, 1)
            s_i = s.unsqueeze(-1).expand(b, t, n, n)
            s_j = s.unsqueeze(-2).expand(b, t, n, n)
            bias = self.state_pair_bias[:, s_i, s_j]  # (H,B,T,N,N)
            bias = bias.permute(1, 2, 0, 3, 4)
            logits = logits + bias

        if self.cfg.symmetric_graph:
            logits = 0.5 * (logits + logits.transpose(-1, -2))

        attn = self._combine_base_and_dynamic(logits)
        attn = self.dropout(attn)

        if self.cfg.add_self_loops:
            eye = torch.eye(n, device=attn.device, dtype=attn.dtype).view(1, 1, 1, n, n)
            attn = attn + eye
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        return attn



class StaticGraphScorer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.num_edge_heads
        self.logits = nn.Parameter(
            torch.zeros(cfg.num_edge_heads, cfg.num_nodes, cfg.num_nodes)
        )
        nn.init.normal_(self.logits, std=0.02)

    def forward(self, h: torch.Tensor, state_ids: torch.Tensor) -> torch.Tensor:
        # h: (B, T, N, D), ignored except for batch/time shape
        b, t, n, _ = h.shape
        logits = self.logits

        if self.cfg.symmetric_graph:
            logits = 0.5 * (logits + logits.transpose(-1, -2))

        attn = torch.softmax(logits, dim=-1)  # (H, N, N)
        attn = attn.unsqueeze(0).unsqueeze(0).expand(b, t, -1, -1, -1).contiguous()
        return attn


class DynamicBaseGraphScorer(DynamicGraphScorer):
    """Dynamic scorer with an added learnable base graph.

        A_t = normalise( base[h, i, j] + QK^T(h_t) / sqrt(d) )

    The base is a raw (H, N, N) parameter (as in StaticGraphScorer); QK^T then
    learns the time-varying deviation from it. With QK^T -> 0 this reduces to the
    static graph.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(cfg)
        self.use_base_graph = True
        self.base_graph = nn.Parameter(
            torch.zeros(cfg.num_edge_heads, cfg.num_nodes, cfg.num_nodes)
        )
        nn.init.normal_(self.base_graph, std=0.02)
        self._make_dynamic_residual_parameter()


# ------------------------------------------------------------
# Spatial mixing using inferred graph A_t
# ------------------------------------------------------------

class _SpatialMPBlock(nn.Module):
    """
    A single message-passing block (the original SpatialMessagePassing body).
    Input:
        h:    (B, T, N, D)
        attn: (B, T, H, N, N)
    Output:
        out:  (B, T, N, D)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_heads = cfg.num_edge_heads
        if cfg.d_model % cfg.num_edge_heads != 0:
            raise ValueError("d_model must be divisible by num_edge_heads")
        self.head_dim = cfg.d_model // cfg.num_edge_heads

        # Value mixed over the graph:
        #   "hidden"          -> v_proj(h)         (contextualised hidden state)
        #   "state_embedding" -> v_proj(e)         (raw current-state embedding)
        #   "concat"          -> v_proj([h ; e])   (both)
        self.spatial_value = getattr(cfg, "spatial_value", "hidden")
        in_dim = cfg.d_model * 2 if self.spatial_value == "concat" else cfg.d_model
        self.v_proj = nn.Linear(in_dim, cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.norm_mix = nn.LayerNorm(cfg.d_model)
        self.norm_ff = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.ff_mult * cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ff_mult * cfg.d_model, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, h: torch.Tensor, attn: torch.Tensor,
                e: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, n, d = h.shape
        if self.spatial_value == "state_embedding":
            if e is None:
                raise ValueError("spatial_value='state_embedding' needs e")
            val_in = e
        elif self.spatial_value == "concat":
            if e is None:
                raise ValueError("spatial_value='concat' needs e")
            val_in = torch.cat([h, e], dim=-1)
        else:
            val_in = h
        v = self.v_proj(val_in).view(b, t, n, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        mixed = torch.matmul(attn, v)  # (B,T,H,N,Hd)
        mixed = mixed.permute(0, 1, 3, 2, 4).reshape(b, t, n, d)
        mixed = self.out_proj(mixed)

        h = self.norm_mix(h + mixed)
        h = self.norm_ff(h + self.ff(h))
        return h


class SpatialMessagePassing(nn.Module):
    """
    Stack of message-passing blocks that reuse the same graph attn (the scorer
    computes it once; it is then propagated for num_spatial_layers hops).
    num_spatial_layers defaults to 1.

    Input:
        h:    (B, T, N, D)
        attn: (B, T, H, N, N)
    Output:
        out:  (B, T, N, D)
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        num_layers = getattr(cfg, "num_spatial_layers", 1)
        self.layers = nn.ModuleList([_SpatialMPBlock(cfg) for _ in range(num_layers)])

    def forward(self, h: torch.Tensor, attn: torch.Tensor,
                e: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            h = layer(h, attn, e=e)
        return h
