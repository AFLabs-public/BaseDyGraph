# Interlaced spatio-temporal patch

This patch includes the previous dynamic_base residual gate / convex gate changes and adds an interlaced ST block stack.

New `ModelConfig` fields:

```python
interlaced_st_blocks: bool = False
num_st_blocks: int = 1
first_spatial_module_type: str | None = None
st_block_post_norm: bool = True
```

Default behaviour is unchanged when `interlaced_st_blocks=False` and `num_st_blocks=1`.

Interlaced mode runs:

```text
embedding -> ST block 0 -> ST block 1 -> ... -> head

ST block = temporal module -> graph scorer -> spatial message passing
```

`first_spatial_module_type` lets block 0 use a different graph scorer, for example:

```python
interlaced_st_blocks=True
num_st_blocks=2
first_spatial_module_type="static_graph"
spatial_module_type="dynamic_base"
```

This gives:

```text
Temporal -> static spatial -> Temporal -> dynamic_base spatial
```

Residual gate logging:
- Legacy/final scorer: `{stage}/dynamic_residual_alpha`
- Per interlaced block: `{stage}/block{i}_dynamic_residual_alpha`
