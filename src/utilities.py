
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

@dataclass
class ModelConfig:
    num_states: int
    num_nodes: int
    d_model: int = 128
    nhead: int = 4
    num_temporal_layers: int = 3
    num_spatial_layers: int = 1
    dropout: float = 0.1
    ff_mult: int = 4
    max_seq_len: int = 512
    num_edge_heads: int = 4
    graph_hidden_dim: int = 128
    spatial_dropout: float = 0.1
    use_node_embedding: bool = True
    use_state_pair_bias: bool = False
    add_self_loops: bool = False
    symmetric_graph: bool = False
    predict_next_state: bool = True
    temporal_module_type: str = "transformer"
    spatial_module_type: str = "dynamic_graph"
    spatial_value: str = "hidden"   # "hidden" | "state_embedding" | "concat"
    graph_activation: str = "softmax"   # "softmax" | "sparsemax" | "entmax15" | "gated"
    gate_tau: float = 0.5            # temperature for graph_activation="gated"
    gate_row_normalise: bool = True  # row-normalise after gating (controls message scale)

    # Residual gate for spatial_module_type="dynamic_base": blends the dynamic
    # logits against the learned base graph by a weight alpha.
    #   "none"     -> alpha = 1.0 (base + dynamic, ungated)
    #   "scalar"   -> one learnable alpha shared across edge heads
    #   "per_head" -> one learnable alpha per edge head
    # dynamic_residual_init is alpha's initial value in [0, 1]; a small value
    # starts near the base graph and adds dynamic deviation during training.
    dynamic_residual_gate: str = "none"       # "none" | "scalar" | "per_head"
    dynamic_residual_init: float = 1.0         # e.g. 0.05 for a conservative start
    dynamic_residual_learnable: bool = True    # if False, alpha is fixed at init
    dynamic_residual_mix: str = "logit"        # "logit" | "convex"

    # Interlaced spatio-temporal stack. Default is a single temporal encoder
    # followed by one graph scorer / spatial block. With interlaced_st_blocks=True
    # or num_st_blocks > 1, the backbone runs repeated
    # temporal -> graph scorer -> spatial blocks, so later scorers condition on
    # representations that have already mixed cross-node information.
    interlaced_st_blocks: bool = False
    num_st_blocks: int = 1
    first_spatial_module_type: str | None = None  # optional override for block 0
    st_block_post_norm: bool = True               # LayerNorm after each ST block

    # Graph diagnostics for interlaced stacks. graph_eval_layer selects which
    # block's graph is exposed as out["graph_attn"] for evaluation: -1 for the
    # last non-None graph, 0/1/... for a specific block. graph_log_all_layers
    # logs recovery metrics for every block graph under layer-tagged names.
    graph_eval_layer: int = -1
    graph_log_all_layers: bool = True


    # Graph regularisation, off by default and intended for learned dynamic
    # graphs. graph_reg_layer = -1 targets the final non-None graph, 0/1/... a
    # specific block.
    #   graph_entropy_reg         : minimise row entropy directly
    #   graph_target_entropy_reg  : match a target row entropy (sharpen without collapse)
    #   graph_temporal_smooth_reg : penalise frame-to-frame change
    graph_reg_layer: int = -1
    graph_reg_warmup_epochs: int = 0
    graph_entropy_reg: float = 0.0
    graph_target_entropy: float | None = None
    graph_target_entropy_reg: float = 0.0
    graph_temporal_smooth_reg: float = 0.0


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, T, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        t = x.size(1)
        return x + self.pe[:, :t]


def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)
