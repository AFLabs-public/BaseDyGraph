# Logging / graph evaluation cleanup patch

## Main changes

1. Residual/convex alpha logs are now under `graph_mix/...`, never under `train/...` or `val/...`.

Examples:

- `graph_mix/train/layer_00/alpha_mean`
- `graph_mix/val/layer_01/alpha_mean`
- `graph_mix/val/layer_01/alpha_min`
- `graph_mix/val/layer_01/alpha_max`

2. Graph recovery logs have explicit layer IDs for interlaced models:

- `graph_layers/layer_00/val/corr`
- `graph_layers/layer_01/val/corr`
- `graph_layers/layer_01/val/auroc`

3. A selected graph is also logged:

- `graph_selected/val/corr`
- `graph_selected/val/auroc`

The selected graph is controlled by `graph_eval_layer`.

4. Backwards-compatible keys are preserved for notebook tables:

- `val/graph_corr`
- `val/graph_auroc`
- `val/graph_mse`

These refer to the selected graph only.

## New config knobs

```python
graph_eval_layer = -1       # -1 = last non-None graph; 0/1/... = exact ST block index
graph_log_all_layers = True # log graph recovery for every interlaced block
```

Add to notebook config:

```python
GRAPH_EVAL_LAYER = -1
GRAPH_LOG_ALL_LAYERS = True
```

and to `BASE_CFG_KWARGS`:

```python
graph_eval_layer=GRAPH_EVAL_LAYER,
graph_log_all_layers=GRAPH_LOG_ALL_LAYERS,
```

## Which graph is evaluated?

For an interlaced two-block model with block 0 static and block 1 dynamic_base:

```python
FIRST_SPATIAL_MODULE_TYPE = "static_graph"
SPATIAL_MODULE_TYPE = "dynamic_base"
NUM_ST_BLOCKS = 2
GRAPH_EVAL_LAYER = -1
```

The selected graph is block 1, the final dynamic_base graph.

Set:

```python
GRAPH_EVAL_LAYER = 0
```

to evaluate the first block graph instead.

