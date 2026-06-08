<p align="center">
  <img src="docs/fox_banner_long.png" alt="Autonomous-Fox Laboratories" width="100%">
</p>

<h1 align="center">BaseDyGraph</h1>

<p align="center">
  <em>A controlled benchmark and interlaced spatio-temporal model for learning time-varying graph structure.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/framework-PyTorch%20%2B%20Lightning-ee4c2c" alt="PyTorch + Lightning">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="License: Apache-2.0">
  <img src="https://img.shields.io/badge/status-research-informational" alt="Research">
</p>

---

**BaseDyGraph** is a teaching benchmark for dynamic graph learning. Discrete node states evolve under a hidden regime process, where each regime carries its own sparse directed graph; the model must predict each node's next state from the state sequences alone, and — where possible — recover the active graph. Because the data generator is known, it exposes exact **static** and **dynamic oracles**, so every result can be read against the *dynamic headroom*: the performance available only to a model that infers time-varying structure.

It is built for learning the craft of model experimentation — running end-to-end training, designing controlled ablations and sweeps, and seeing how architecture choices change both prediction and graph recovery.

## Start here

- 📄 **[Technical note (PDF)](docs/basedygraph_technical_note.pdf)** — the primary reference. Read this first. It covers the mental model, the data-generating process, the architecture (temporal encoder, graph scorers, interlaced spatio-temporal blocks, the dynamic-base gate, regularisation), the diagnostics, and the ablation protocol. LaTeX source: [`docs/basedygraph_technical_note.tex`](docs/basedygraph_technical_note.tex).
- 📓 **[`notebooks/BaseDyGraph_experiments.ipynb`](notebooks/BaseDyGraph_experiments.ipynb)** — the runnable experiments notebook. Generate data once, then walk the graph-type ladder (no-graph → static → dynamic → dynamic-base → oracle).

## The task in one line

You get only the observed state sequences; the regime path and per-regime graphs are **hidden** (used solely to build the oracles and score recovery). Predict next state at every node, optionally expose a per-step adjacency that recovers the active graph, and aim to capture as much dynamic headroom as possible while keeping the recovered graph faithful and selective. A recommended configuration is deliberately not provided — finding one is the exercise.

## Repository layout

```
src/
  synthetic_generators.py       # data-generating process + static/dynamic oracles
  data_module.py                # Lightning DataModule
  modules.py                    # temporal encoder, graph scorers, spatial message passing
  model.py                      # backbone, next-state head, Lightning module
  utilities.py                  # ModelConfig + helpers
  evaluation_utilities.py       # metrics / recovery scoring
  propagation_delay_scorer.py   # optional lead-lag scorer (not wired into the factory by default)
docs/
  basedygraph_technical_note.pdf
  basedygraph_technical_note.tex
  fox_banner_long.png
notebooks/
  BaseDyGraph_experiments.ipynb
LICENSE
NOTICE
requirements.txt                # you provide this (see Installation)
```

## Installation

Requires **Python 3.10+**. Set up an isolated virtual environment and install into it.

```bash
# 1. clone
git clone <your-repo-url>
cd basedygraph

# 2. create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\Activate.ps1       # Windows (PowerShell)
# .venv\Scripts\activate.bat       # Windows (cmd)

# 3. install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. (optional) register the venv as a Jupyter kernel so the notebook can find it
python -m ipykernel install --user --name basedygraph --display-name "Python (BaseDyGraph)"

# 5. launch
jupyter lab                        # or: jupyter notebook
```

Then open `notebooks/BaseDyGraph_experiments.ipynb` and select the **Python (BaseDyGraph)** kernel. `src/utilities.py:ModelConfig` is the single source of truth for runnable model settings.

> **`requirements.txt`** is not included — add one pinning your dependencies (at minimum `torch`, `lightning`, `numpy`, plus `jupyterlab` and `ipykernel` for the notebook, and `wandb` if you use it for logging). To deactivate the environment when finished, run `deactivate`.

## License

Code in this repository is licensed under the [Apache License 2.0](LICENSE).

The "Autonomous-Fox" name, the Autonomous-Fox Laboratories wordmark, and the fox logo (including `docs/fox_banner_long.png`) are trademarks of Autonomous-Fox Laboratories and are **not** covered by the Apache license (see Section 6 and the [NOTICE](NOTICE) file); all rights to those marks are reserved.
