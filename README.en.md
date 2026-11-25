# Temporal Debt Theory (TDT)

Independent computational research on irreversibility, memory, information, and control in finite physical systems.

**Research status:** an exploratory theoretical framework accompanied by classical and quantum simulations. TDT is not presented as an established physical law or a universally validated theory.

## The central question

This work investigates whether operational descriptions of irreversibility and memory can help characterize the cost of transforming, controlling, or recovering physical states when relevant information remains distributed across the environment.

The idea has been examined in different models: two-level systems, Ising chains, finite baths, quantum models with explicit modes, and Markovian and non-Markovian processes. Each model tests only part of the hypothesis; no single result establishes the entire theory.

## Research directions

- **Information thermodynamics:** Landauer bounds, erasure, and side information.
- **Finite baths and recovery:** Ising and Creutz-demon models with stochastic simulations.
- **Quantum memory:** explicit harmonic modes, reduced dynamics, and backflow witnesses.
- **Optimal control:** numerical protocols, time-step refinement, and comparison with conventional references.

## Selected results

- A two-level benchmark numerically solves the master-equation dynamics and checks the energy balance with residuals close to numerical precision. At the same time, the operational quantity called Pi changes with the reference scale and does not generally coincide with recoverable dissipation. This identifies an important mathematical boundary.
- A model with two qubits and two truncated harmonic modes has a 36-dimensional Hilbert space and exhibits trace-distance backflow for the selected state pair and parameters. This characterizes memory in that specific model; it is not a TDT-specific signature.
- A line of finite-bath stochastic simulations runs up to 100,000 trajectories per condition to compare control, evolution, and recovery strategies in Ising and Creutz-demon models.
- An exact calculation on a four-bit Ising ring studies the ideal erasure bound when the final magnetization is available as side information.

See [docs/EXPERIMENT_CATALOG.md](docs/EXPERIMENT_CATALOG.md) for methods, output files, and limitations.

## Laboratory contents

- `*.py` — Python simulations and analyses.
- `*.cpp` — compiled simulators for stochastic experiments.
- `*.json` — structured outputs from specific runs.
- `*.txt` and `*.log` — reports and execution records.
- `*.npz`, `*.csv`, and `*.png` — numerical and visual artifacts from selected experiments.

The labels `v2`, `v4`, and `v5` identify historical research iterations, not releases of a consolidated theory.

## Environment

- Python 3.12 or newer.
- NumPy and SciPy for numerical simulations.
- Matplotlib for plotting scripts.
- A compatible C++ compiler for the `*.cpp` experiments.

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

**Reproducibility:** read the experiment catalog first. Some historical scripts use absolute local paths or write outputs to the working directory. Run them in a clean copy of the laboratory to preserve existing results.

## Scientific integrity

- Negative, inconclusive, and limiting results are part of the research record.
- Backflow alone does not imply a new thermodynamic law.
- SPSA optimization does not prove that a global minimum has been found.
- Markovian and explicit-mode models can only be compared when their physical assumptions and parameters are compatible.
- A numerical identity or a correct energy balance does not show that TDT adds a new prediction.
- The central mathematical formulation still requires consistent operational definitions, dimensional analysis, and independent tests.

## Current status

TDT remains an exploratory research program. The repository's value is its collection of models, code, traceable results, and questions that did — or did not — survive testing. No claim is made of experimental validation, scientific consensus, or universal applicability.

---

Leia também: [Português](README.md) · [Español](README.es.md) · [Experiment catalog](docs/EXPERIMENT_CATALOG.md)