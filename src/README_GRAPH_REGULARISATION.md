# Graph regularisation patch

Adds optional unsupervised graph-shape regularisation to the selected graph, usually the final dynamic graph in an interlaced stack.

New ModelConfig knobs:

```python
GRAPH_REG_LAYER = -1                 # -1 = last non-None/final graph; 0/1/... = exact block
GRAPH_REG_WARMUP_EPOCHS = 20         # 0 disables warmup
GRAPH_ENTROPY_REG = 0.0              # minimise entropy directly; use cautiously
GRAPH_TARGET_ENTROPY = 2.2           # None => use true graph entropy if available
GRAPH_TARGET_ENTROPY_REG = 1e-4      # match graph entropy to target
GRAPH_TEMPORAL_SMOOTH_REG = 1e-4     # discourage graph flicker over time
```

Add to `BASE_CFG_KWARGS`:

```python
graph_reg_layer=GRAPH_REG_LAYER,
graph_reg_warmup_epochs=GRAPH_REG_WARMUP_EPOCHS,
graph_entropy_reg=GRAPH_ENTROPY_REG,
graph_target_entropy=GRAPH_TARGET_ENTROPY,
graph_target_entropy_reg=GRAPH_TARGET_ENTROPY_REG,
graph_temporal_smooth_reg=GRAPH_TEMPORAL_SMOOTH_REG,
```

Recommended first setting:

```python
GRAPH_REG_LAYER = -1
GRAPH_REG_WARMUP_EPOCHS = 20
GRAPH_ENTROPY_REG = 0.0
GRAPH_TARGET_ENTROPY = 2.2
GRAPH_TARGET_ENTROPY_REG = 1e-4
GRAPH_TEMPORAL_SMOOTH_REG = 1e-4
```

Logs are under `graph_reg/{train,val}/...`.
