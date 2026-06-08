"""
Model: backbone + next-state head + Lightning module.

Extracted from the research notebook so notebooks can import it:
    from model import DiscreteSTGraphBackbone, NextStateHead, DiscreteSTGraphLightningModule

DEPENDENCY NOTE
---------------
This module relies on your existing code for:
    ModelConfig, build_temporal_module, build_spatial_components
which live in `modules.py` / `utilities.py`. Make sure those are importable
(same directory on sys.path). The `from modules import *` / `from utilities import *`
lines below mirror the original notebook; adjust if your module names differ.

If you wire the new PropagationDelayGraphScorer into the spatial stage, do it inside
`build_spatial_components` in your modules.py (add a `spatial_module_type ==
"propagation_delay"` branch). This file does not modify your factory.
"""

from typing import Any, Dict, Optional
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from typing import Dict, List, Optional, Tuple

# Your existing code (ModelConfig, build_temporal_module, build_spatial_components, ...)
from utilities import *      # noqa: F401,F403
from modules import *        # noqa: F401,F403


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class InterlacedSTBlock(nn.Module):
    """One interlaced spatio-temporal block.

    Flow:
        h -> temporal block -> graph scorer -> spatial message passing

    h is always represented as (B, T, N, D) at the block boundary. The temporal
    module internally expects (B, N, T, D), so we permute in/out locally.
    """

    def __init__(self, cfg: "ModelConfig", spatial_module_type: str) -> None:  # noqa: F821
        super().__init__()
        self.cfg = cfg
        self.spatial_module_type = spatial_module_type

        self.temporal_module = build_temporal_module(cfg)  # noqa: F821

        if spatial_module_type == "oracle_graph":
            self.graph_scorer = None
            self.spatial_module = SpatialMessagePassing(cfg)  # noqa: F821
        else:
            # Build this block's scorer with a shallow config override so each
            # block can use a different spatial module type while sharing all
            # other graph/normalisation/gate knobs.
            try:
                from dataclasses import replace
                block_cfg = replace(cfg, spatial_module_type=spatial_module_type)
            except Exception:
                block_cfg = cfg
                block_cfg.spatial_module_type = spatial_module_type
            self.graph_scorer, self.spatial_module = build_spatial_components(block_cfg)  # noqa: F821

        self.post_norm = nn.LayerNorm(cfg.d_model) if getattr(cfg, "st_block_post_norm", True) else nn.Identity()

    def _oracle_attn(
        self,
        h_btnd: torch.Tensor,
        regimes: Optional[torch.Tensor],
        oracle_regime_graphs: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if regimes is None or oracle_regime_graphs is None:
            raise RuntimeError("oracle_graph block needs regimes and oracle_regime_graphs")
        b, t, n, _ = h_btnd.shape
        G = oracle_regime_graphs.to(h_btnd.device)  # (R, N, N)
        A = G[regimes.long()]                       # (B, T, N, N)
        row_sum = A.sum(dim=-1, keepdim=True)
        eye = torch.eye(n, device=A.device, dtype=A.dtype).view(1, 1, n, n)
        A = torch.where(row_sum > 1e-6, A, eye.expand_as(A))
        return A.unsqueeze(2).expand(b, t, self.cfg.num_edge_heads, n, n)

    def forward(
        self,
        h_btnd: torch.Tensor,
        state_ids: torch.Tensor,
        e_btnd: torch.Tensor,
        regimes: Optional[torch.Tensor] = None,
        oracle_regime_graphs: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Temporal module expects (B, N, T, D); block boundary is (B, T, N, D).
        h_bntd = h_btnd.permute(0, 2, 1, 3).contiguous()
        h_bntd = self.temporal_module(h_bntd)
        h_btnd = h_bntd.permute(0, 2, 1, 3).contiguous()

        if self.spatial_module_type == "oracle_graph":
            attn = self._oracle_attn(h_btnd, regimes, oracle_regime_graphs)
            h_btnd = self.spatial_module(h_btnd, attn, e=e_btnd)
        elif self.graph_scorer is not None:
            attn = self.graph_scorer(h_btnd, state_ids)
            h_btnd = self.spatial_module(h_btnd, attn, e=e_btnd)
        else:
            attn = None
            h_btnd = self.spatial_module(h_btnd, None, e=e_btnd)

        return self.post_norm(h_btnd), attn


class DiscreteSTGraphBackbone(nn.Module):
    """State IDs -> embeddings -> temporal/spatial backbone -> head-ready reps.

    Default path is the original architecture. Set either
    `interlaced_st_blocks=True` or `num_st_blocks > 1` to use repeated
    temporal -> graph -> spatial blocks.
    """

    def __init__(self, cfg: "ModelConfig") -> None:  # noqa: F821 (from modules/utilities)
        super().__init__()
        self.cfg = cfg
        self.use_interlaced = bool(getattr(cfg, "interlaced_st_blocks", False)) or int(getattr(cfg, "num_st_blocks", 1)) > 1

        self.state_embedding = nn.Embedding(cfg.num_states, cfg.d_model)
        self.node_embedding = (
            nn.Embedding(cfg.num_nodes, cfg.d_model) if cfg.use_node_embedding else None
        )
        self.pre_norm = nn.LayerNorm(cfg.d_model)
        self.post_norm = nn.LayerNorm(cfg.d_model)
        self.oracle_regime_graphs = None

        if self.use_interlaced:
            num_blocks = int(getattr(cfg, "num_st_blocks", 1))
            if num_blocks < 1:
                raise ValueError("num_st_blocks must be >= 1")

            first_type = getattr(cfg, "first_spatial_module_type", None)
            block_types = []
            for i in range(num_blocks):
                if i == 0 and first_type not in {None, "", "same"}:
                    block_types.append(first_type)
                else:
                    block_types.append(cfg.spatial_module_type)

            self.st_blocks = nn.ModuleList([InterlacedSTBlock(cfg, stype) for stype in block_types])

            # Backwards-compatible handles used by logging/evaluation code. They
            # point to the last block, which is usually the refined graph scorer.
            last = self.st_blocks[-1]
            self.graph_scorer = getattr(last, "graph_scorer", None)
            self.spatial_module = getattr(last, "spatial_module", None)
            self.temporal_module = None
        else:
            self.temporal_module = build_temporal_module(cfg)              # noqa: F821
            if cfg.spatial_module_type == "oracle_graph":
                self.graph_scorer = None
                self.spatial_module = SpatialMessagePassing(cfg)           # noqa: F821
            else:
                self.graph_scorer, self.spatial_module = build_spatial_components(cfg)  # noqa: F821
            self.st_blocks = None

    def _initial_embedding_bntd(self, state_ids: torch.Tensor) -> torch.Tensor:
        b, n, t = state_ids.shape
        x = self.state_embedding(state_ids)
        if self.node_embedding is not None:
            node_ids = torch.arange(n, device=state_ids.device)
            x = x + self.node_embedding(node_ids).view(1, n, 1, self.cfg.d_model)
        return self.pre_norm(x)

    def temporal_output(self, state_ids: torch.Tensor) -> torch.Tensor:
        """Representation fed to the final graph scorer, shape (B, T, N, D).

        In the original path this is the post-temporal representation. In the
        interlaced path it is the representation immediately before the final
        block's spatial graph scorer is applied.
        """
        x = self._initial_embedding_bntd(state_ids)
        if not self.use_interlaced:
            h = self.temporal_module(x)
            return h.permute(0, 2, 1, 3).contiguous()

        h_btnd = x.permute(0, 2, 1, 3).contiguous()
        e_btnd = self.state_embedding_btnd(state_ids)
        # Run all but the final block, then run only the final block's temporal
        # part so the returned tensor matches what the final graph scorer sees.
        for block in self.st_blocks[:-1]:
            h_btnd, _ = block(h_btnd, state_ids, e_btnd)
        final = self.st_blocks[-1]
        h_bntd = h_btnd.permute(0, 2, 1, 3).contiguous()
        h_bntd = final.temporal_module(h_bntd)
        return h_bntd.permute(0, 2, 1, 3).contiguous()

    def state_embedding_btnd(self, state_ids: torch.Tensor) -> torch.Tensor:
        """Raw current-state embedding e (B, T, N, D), before temporal mixing."""
        e = self.state_embedding(state_ids)                            # (B, N, T, D)
        return e.permute(0, 2, 1, 3).contiguous()                      # (B, T, N, D)

    def _select_graph_attn(self, block_attns: List[Optional[torch.Tensor]]) -> Optional[torch.Tensor]:
        """Select the graph exposed as out["graph_attn"] for legacy evaluation.

        graph_eval_layer=-1 selects the last non-None graph. Non-negative values
        select that exact interlaced block index. This makes it explicit whether
        recovery metrics are being computed from layer 0, layer 1, etc.
        """
        if not block_attns:
            return None
        layer = int(getattr(self.cfg, "graph_eval_layer", -1))
        if layer >= 0:
            if layer >= len(block_attns):
                raise ValueError(f"graph_eval_layer={layer} out of range for {len(block_attns)} ST blocks")
            return block_attns[layer]
        for attn in reversed(block_attns):
            if attn is not None:
                return attn
        return None

    def _oracle_attn(self, h_btnd: torch.Tensor, regimes: Optional[torch.Tensor]) -> torch.Tensor:
        if regimes is None or getattr(self, "oracle_regime_graphs", None) is None:
            raise RuntimeError("oracle_graph mode needs regimes and oracle_regime_graphs")
        b, t, n, _ = h_btnd.shape
        G = self.oracle_regime_graphs.to(h_btnd.device)                # (R, N, N)
        A = G[regimes.long()]                                          # (B, T, N, N)
        row_sum = A.sum(dim=-1, keepdim=True)
        eye = torch.eye(n, device=A.device, dtype=A.dtype).view(1, 1, n, n)
        A = torch.where(row_sum > 1e-6, A, eye.expand_as(A))
        return A.unsqueeze(2).expand(b, t, self.cfg.num_edge_heads, n, n)

    def forward(self, state_ids: torch.Tensor,
                regimes: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        b, n, t = state_ids.shape
        if n != self.cfg.num_nodes:
            raise ValueError(f"Expected num_nodes={self.cfg.num_nodes}, got {n}")

        e_btnd = self.state_embedding_btnd(state_ids)                  # (B, T, N, D)

        if self.use_interlaced:
            h_btnd = self._initial_embedding_bntd(state_ids).permute(0, 2, 1, 3).contiguous()
            block_attns: List[Optional[torch.Tensor]] = []
            for block in self.st_blocks:
                h_btnd, attn = block(
                    h_btnd,
                    state_ids,
                    e_btnd,
                    regimes=regimes,
                    oracle_regime_graphs=getattr(self, "oracle_regime_graphs", None),
                )
                block_attns.append(attn)
            z = self.post_norm(h_btnd)
            selected_attn = self._select_graph_attn(block_attns)
            return {
                "temporal_repr": h_btnd,
                "spatial_repr": z,
                "graph_attn": selected_attn,
                "block_graph_attns": block_attns,
            }

        h_btnd = self.temporal_output(state_ids)                       # (B, T, N, D)

        if self.cfg.spatial_module_type == "oracle_graph":
            attn = self._oracle_attn(h_btnd, regimes)
            z = self.spatial_module(h_btnd, attn, e=e_btnd)
        elif self.graph_scorer is not None:
            attn = self.graph_scorer(h_btnd, state_ids)                # (B, T, H, N, N)
            z = self.spatial_module(h_btnd, attn, e=e_btnd)            # (B, T, N, D)
        else:
            attn = None
            z = self.spatial_module(h_btnd, None, e=e_btnd)            # identity passthrough

        z = self.post_norm(z)
        return {"temporal_repr": h_btnd, "spatial_repr": z, "graph_attn": attn}

# ---------------------------------------------------------------------------
# Next-state head
# ---------------------------------------------------------------------------

class NextStateHead(nn.Module):
    """Predict s_{i,t+1} from the representation at time t. (B,T,N,D) -> (B,N,T-1,K)."""

    def __init__(self, d_model: int, num_states: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, num_states)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.proj(h[:, :-1])                                  # (B, T-1, N, K)
        return logits.permute(0, 2, 1, 3).contiguous()                # (B, N, T-1, K)


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class DiscreteSTGraphLightningModule(pl.LightningModule):
    def __init__(
        self,
        cfg: "ModelConfig",  # noqa: F821
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        true_regime_graphs: Optional[torch.Tensor] = None,
        scheduler_t_max: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["cfg", "true_regime_graphs"])
        self.cfg = cfg
        self.backbone = DiscreteSTGraphBackbone(cfg)
        self.next_state_head = NextStateHead(cfg.d_model, cfg.num_states)
        self.lr = lr
        self.weight_decay = weight_decay
        # cosine horizon; should match trainer max_epochs. None -> resolved at
        # configure_optimizers from trainer.max_epochs (falls back to 100).
        self.scheduler_t_max = scheduler_t_max

        if true_regime_graphs is not None:
            self.register_buffer("true_regime_graphs", true_regime_graphs.float(), persistent=False)
        else:
            self.true_regime_graphs = None

        # for the oracle_graph diagnostic rung: hand the true graphs to the backbone
        if cfg.spatial_module_type == "oracle_graph":
            if true_regime_graphs is None:
                raise ValueError("oracle_graph mode requires true_regime_graphs")
            self.backbone.oracle_regime_graphs = true_regime_graphs.float()

    def forward(self, state_ids: torch.Tensor,
                regimes: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        out = self.backbone(state_ids, regimes=regimes)
        out["next_state_logits"] = self.next_state_head(out["spatial_repr"])
        return out

    def _compute_graph_recovery_metrics(self, graph_attn, regimes) -> Dict[str, torch.Tensor]:
        if self.true_regime_graphs is None:
            return {}
        attn = graph_attn.mean(dim=2)                                  # (B, T, N, N)
        true_graph = self.true_regime_graphs[regimes.long()]
        n = attn.size(-1)
        off = (~torch.eye(n, device=attn.device, dtype=torch.bool)).view(1, 1, n, n)
        attn_off = attn.masked_select(off).view(attn.size(0), attn.size(1), -1)
        true_off = true_graph.masked_select(off).view(true_graph.size(0), true_graph.size(1), -1)
        mse = F.mse_loss(attn_off, true_off)
        pc = attn_off - attn_off.mean(-1, keepdim=True)
        tc = true_off - true_off.mean(-1, keepdim=True)
        denom = pc.std(-1) * tc.std(-1)
        corr = ((pc * tc).mean(-1) / denom.clamp_min(1e-6)).mean()

        # AUROC is scale-fair: does the attention RANK true edges above non-edges?
        # Unlike Pearson corr, it doesn't penalise dense softmax for failing to match
        # the true graph's exact-zero sparsity — it only asks about ordering. This is
        # the right recovery metric while the activation is dense (softmax).
        a = attn_off.reshape(-1)
        lbl = (true_off.reshape(-1) > 0)
        pos, neg = a[lbl], a[~lbl]
        if pos.numel() > 0 and neg.numel() > 0:
            cap = 20000
            if pos.numel() > cap:
                pos = pos[torch.randperm(pos.numel(), device=a.device)[:cap]]
            if neg.numel() > cap:
                neg = neg[torch.randperm(neg.numel(), device=a.device)[:cap]]
            allv = torch.cat([pos, neg])
            ranks = allv.argsort().argsort().float() + 1.0
            r_pos = ranks[:pos.numel()].sum()
            auroc = (r_pos - pos.numel() * (pos.numel() + 1) / 2) / (pos.numel() * neg.numel())
        else:
            auroc = torch.tensor(float("nan"), device=a.device)
        return {"graph_mse": mse, "graph_corr": corr, "graph_auroc": auroc}

    def _log_graph_mix(self, stage: str, step: bool) -> None:
        """Log residual/convex graph-mix alphas with explicit layer ids.

        All alpha logs live under graph_mix/... rather than train/... or val/...
        to avoid name collisions with ordinary training metrics. In interlaced
        mode each block gets graph_mix/{stage}/layer_{i:02d}/...
        """
        blocks = getattr(self.backbone, "st_blocks", None)
        if blocks is not None:
            for bi, block in enumerate(blocks):
                scorer = getattr(block, "graph_scorer", None)
                if scorer is None or not hasattr(scorer, "dynamic_residual_alpha"):
                    continue
                alpha = scorer.dynamic_residual_alpha().detach()
                prefix = f"graph_mix/{stage}/layer_{bi:02d}"
                self.log(f"{prefix}/alpha_mean", alpha.mean(), on_step=step, on_epoch=True)
                if alpha.numel() > 1:
                    self.log(f"{prefix}/alpha_min", alpha.min(), on_step=step, on_epoch=True)
                    self.log(f"{prefix}/alpha_max", alpha.max(), on_step=step, on_epoch=True)
            return

        scorer = getattr(self.backbone, "graph_scorer", None)
        if scorer is not None and hasattr(scorer, "dynamic_residual_alpha"):
            alpha = scorer.dynamic_residual_alpha().detach()
            prefix = f"graph_mix/{stage}/layer_00"
            self.log(f"{prefix}/alpha_mean", alpha.mean(), on_step=step, on_epoch=True)
            if alpha.numel() > 1:
                self.log(f"{prefix}/alpha_min", alpha.min(), on_step=step, on_epoch=True)
                self.log(f"{prefix}/alpha_max", alpha.max(), on_step=step, on_epoch=True)

    def _log_one_graph_recovery(
        self,
        attn: Optional[torch.Tensor],
        regimes: torch.Tensor,
        stage: str,
        step: bool,
        prefix: str,
    ) -> None:
        if attn is None or self.true_regime_graphs is None:
            return
        gm = self._compute_graph_recovery_metrics(attn, regimes.to(attn.device))
        if not gm:
            return
        self.log(f"{prefix}/{stage}/mse", gm["graph_mse"], on_step=step, on_epoch=True)
        self.log(f"{prefix}/{stage}/corr", gm["graph_corr"], on_step=step, on_epoch=True)
        if "graph_auroc" in gm:
            self.log(f"{prefix}/{stage}/auroc", gm["graph_auroc"], on_step=step, on_epoch=True)

    def _log_graph_diagnostics(self, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], stage: str, step: bool) -> None:
        """Log graph entropy/recovery with explicit selected/all-layer names."""
        graph_attn = out.get("graph_attn", None)
        with torch.no_grad():
            # Selected graph: this is the graph exposed as out["graph_attn"].
            # In interlaced mode the selected layer is controlled by cfg.graph_eval_layer.
            if graph_attn is not None:
                a = graph_attn.clamp_min(1e-8)
                entropy = -(a * a.log()).sum(dim=-1).mean()
                self.log(f"graph_selected/{stage}/entropy", entropy, on_step=step, on_epoch=True)

                if "regimes" in batch and self.true_regime_graphs is not None:
                    self._log_one_graph_recovery(
                        graph_attn,
                        batch["regimes"],
                        stage,
                        step,
                        prefix="graph_selected",
                    )

                    # Backwards-compatible summary keys for existing notebook tables.
                    # These are for the selected graph only; layer-specific names below
                    # make it clear which interlaced block is being evaluated.
                    gm = self._compute_graph_recovery_metrics(graph_attn, batch["regimes"].to(graph_attn.device))
                    if gm:
                        self.log(f"{stage}/graph_mse", gm["graph_mse"], on_step=step, on_epoch=True)
                        self.log(f"{stage}/graph_corr", gm["graph_corr"], on_step=step, on_epoch=True)
                        if "graph_auroc" in gm:
                            self.log(f"{stage}/graph_auroc", gm["graph_auroc"], on_step=step, on_epoch=True)

            # Every interlaced layer gets its own diagnostics if requested.
            if bool(getattr(self.cfg, "graph_log_all_layers", True)) and "regimes" in batch:
                block_attns = out.get("block_graph_attns", None)
                if block_attns is not None:
                    for bi, attn_b in enumerate(block_attns):
                        if attn_b is None:
                            continue
                        a = attn_b.clamp_min(1e-8)
                        entropy = -(a * a.log()).sum(dim=-1).mean()
                        layer_prefix = f"graph_layers/layer_{bi:02d}"
                        self.log(f"{layer_prefix}/{stage}/entropy", entropy, on_step=step, on_epoch=True)
                        self._log_one_graph_recovery(attn_b, batch["regimes"], stage, step, prefix=layer_prefix)

            self._log_graph_mix(stage, step)

    def _select_graph_for_regularisation(self, out: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """Return the graph attention tensor to regularise.

        Defaults to the same selected/final graph used for evaluation. In an
        interlaced stack, cfg.graph_reg_layer=-1 selects the last non-None graph;
        non-negative values select a specific block. For the old single-block
        path, this is just out["graph_attn"].
        """
        block_attns = out.get("block_graph_attns", None)
        if block_attns is None:
            return out.get("graph_attn", None)

        layer = int(getattr(self.cfg, "graph_reg_layer", -1))
        if layer >= 0:
            if layer >= len(block_attns):
                return None
            return block_attns[layer]

        for attn in reversed(block_attns):
            if attn is not None:
                return attn
        return out.get("graph_attn", None)

    def _graph_reg_warmup_scale(self) -> float:
        warmup = int(getattr(self.cfg, "graph_reg_warmup_epochs", 0) or 0)
        if warmup <= 0:
            return 1.0
        # current_epoch is 0-indexed; use +1 so the first epoch gets non-zero
        # pressure but still ramps gently.
        return float(min(1.0, max(0.0, (self.current_epoch + 1) / warmup)))

    def _compute_graph_regularisation(
        self,
        out: Dict[str, torch.Tensor],
        stage: str,
        step: bool,
    ) -> torch.Tensor:
        """Optional unsupervised graph-shape regularisation.

        Returns a scalar tensor. All terms are disabled by default, so existing
        runs are exactly unchanged unless the corresponding coefficients are set.
        Regularisation is logged under graph_reg/{stage}/... and is applied to
        the training objective only by _shared_step.
        """
        attn = self._select_graph_for_regularisation(out)
        device = out["next_state_logits"].device
        zero = torch.zeros((), device=device)
        if attn is None:
            return zero

        entropy_coef = float(getattr(self.cfg, "graph_entropy_reg", 0.0) or 0.0)
        target_entropy_coef = float(getattr(self.cfg, "graph_target_entropy_reg", 0.0) or 0.0)
        smooth_coef = float(getattr(self.cfg, "graph_temporal_smooth_reg", 0.0) or 0.0)
        if entropy_coef == 0.0 and target_entropy_coef == 0.0 and smooth_coef == 0.0:
            return zero

        warmup_scale = self._graph_reg_warmup_scale()
        a = attn.clamp_min(1e-8)
        row_entropy = -(a * a.log()).sum(dim=-1)  # (B, T, H, N)
        entropy = row_entropy.mean()

        reg = zero
        if entropy_coef != 0.0:
            ent_loss = entropy
            reg = reg + entropy_coef * ent_loss
            self.log(f"graph_reg/{stage}/entropy_loss", ent_loss.detach(), on_step=step, on_epoch=True)

        if target_entropy_coef != 0.0:
            target = getattr(self.cfg, "graph_target_entropy", None)
            if target is None:
                # If no explicit target is supplied, use the mean true graph
                # entropy when available. This is synthetic-diagnostic friendly;
                # for real data, set an explicit target or leave the coefficient 0.
                if self.true_regime_graphs is not None:
                    G = self.true_regime_graphs.to(device).float()
                    G = G / G.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                    target_tensor = -(G.clamp_min(1e-12) * G.clamp_min(1e-12).log()).sum(dim=-1).mean()
                else:
                    target_tensor = entropy.detach()
            else:
                target_tensor = torch.as_tensor(float(target), device=device, dtype=entropy.dtype)
            target_loss = (entropy - target_tensor.detach()).pow(2)
            reg = reg + target_entropy_coef * target_loss
            self.log(f"graph_reg/{stage}/target_entropy", target_tensor.detach(), on_step=False, on_epoch=True)
            self.log(f"graph_reg/{stage}/target_entropy_loss", target_loss.detach(), on_step=step, on_epoch=True)

        if smooth_coef != 0.0 and attn.size(1) > 1:
            smooth_loss = (attn[:, 1:] - attn[:, :-1]).pow(2).mean()
            reg = reg + smooth_coef * smooth_loss
            self.log(f"graph_reg/{stage}/temporal_smooth_loss", smooth_loss.detach(), on_step=step, on_epoch=True)

        reg = reg * warmup_scale
        self.log(f"graph_reg/{stage}/warmup_scale", torch.as_tensor(warmup_scale, device=device), on_step=step, on_epoch=True)
        self.log(f"graph_reg/{stage}/loss", reg.detach(), on_step=step, on_epoch=True)
        return reg

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        state_ids = batch["state_ids"].long()
        regimes = batch.get("regimes", None)
        out = self(state_ids, regimes=regimes)
        logits = out["next_state_logits"]                              # (B, N, T-1, K)
        target = state_ids[:, :, 1:]
        pred_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

        preds = logits.argmax(dim=-1)
        acc = (preds == target).float().mean()

        # Training has both step and epoch curves; validation/test are epoch-only.
        step = (stage == "train")
        graph_reg = self._compute_graph_regularisation(out, stage, step)
        loss = pred_loss + graph_reg if stage == "train" else pred_loss

        self.log(f"{stage}/pred_loss", pred_loss, prog_bar=False, on_step=step, on_epoch=True)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=step, on_epoch=True)
        self.log(f"{stage}/acc", acc, prog_bar=True, on_step=step, on_epoch=True)

        self._log_graph_diagnostics(out, batch, stage, step)
        return loss

    def training_step(self, batch, batch_idx):  # noqa: D401
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # match the cosine horizon to the actual training length so the LR anneals.
        t_max = self.scheduler_t_max
        if t_max is None:
            t_max = getattr(self.trainer, "max_epochs", None) or 100
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max)
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
