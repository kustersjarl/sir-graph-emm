# Stochastic SIR Epidemics on Contact Networks

This repository contains a stochastic Susceptible--Infectious--Recovered (SIR) epidemic simulator on contact networks, implemented mainly using NumPy.

The project is designed for studying subgroup discovery in epidemic simulations. In particular, it supports contact-adjusted Exceptional Model Mining (EMM) methods for distinguishing intrinsic susceptibility and infectiousness from differences caused by realised contact opportunities.

The simulator supports:

- weighted undirected contact networks;
- individual-level susceptibility and infectiousness parameters;
- stochastic transmission and recovery;
- tracking of infection times, recovery times, and infectors;
- repeated outbreak simulations;
- exposure- and opportunity-based epidemic analysis;
- subgroup discovery using contact-adjusted EMM.

## Installation

The repository uses the [`uv`](https://docs.astral.sh/uv/) package manager. Using `uv` is recommended because it creates the virtual environment and installs the correct dependencies automatically.

Clone the repository and install the environment:

```bash
git clone <url>
cd sir-graph-emm
uv sync
```

If `uv` is not available, the required dependencies are listed in `pyproject.toml` and can be installed manually, although this is not recommended.

## Quick Start

The main simulation and experimental workflow is available in the notebook included with the repository.

After installing the environment, the notebook can be opened in VS Code or Jupyter and run from top to bottom.

The notebook demonstrates the complete workflow:

1. generation of the synthetic population;
2. construction of the contact network;
3. calibration of the epidemic model;
4. stochastic SIR simulation;
5. construction of exposure and transmission-opportunity observations;
6. Exceptional Model Mining with Models A and B;
7. subgroup-recovery and stress-test experiments;
8. generation of the figures and final results.

A more elaborate user guide can be added in the future.

## Testing

The repository includes unit tests for the main simulation and analysis components.

With the `uv` environment installed, run:

```bash
uv run pytest
```

For detailed output:

```bash
uv run pytest -v
```

Alternatively, if the virtual environment is already activated:

```bash
pytest -v
```

All tests should pass before running or modifying the experimental pipeline.

## Project Structure

The main package is organised as follows:

```text
sir_graph/
├── __init__.py
├── analysis.py
├── dynamics.py
├── emm.py
├── graph.py
├── parameters.py
└── states.py
```

The modules have the following roles:

- `graph.py` -- graph representation and basic network operations;
- `parameters.py` -- individual susceptibility, infectiousness, and recovery parameters;
- `states.py` -- epidemic states and seed selection;
- `dynamics.py` -- stochastic SIR transmission and recovery dynamics;
- `analysis.py` -- extraction of exposure and transmission-opportunity observations;
- `emm.py` -- subgroup-description generation, scoring, and EMM search.

The `tests/` directory contains the corresponding unit tests.

## Epidemic Model

For an infectious individual \(u\) and susceptible neighbour \(v\), transmission is sampled independently according to

```text
P(u -> v) = min(1, p * tau_u * sigma_v * w_uv)
```

where:

- `p` is the global transmission parameter;
- `tau_u` is the infectiousness of the source;
- `sigma_v` is the susceptibility of the recipient;
- `w_uv` is the transmission weight of the contact.

Recovery is stochastic and is sampled independently at each simulation step.

## Exceptional Model Mining

Two contact-adjusted EMM models are implemented:

**Model A -- Exposure-adjusted susceptibility**

Model A evaluates whether members of a subgroup become infected more readily after accounting for their realised exposure to infectious neighbours.

**Model B -- Opportunity-adjusted infectiousness**

Model B evaluates whether infected members of a subgroup cause more secondary infections after accounting for their realised opportunities to transmit to susceptible neighbours.

Both models compare a candidate subgroup with its complement and combine the estimated transmission effect with subgroup support.

## Implementation Notes

Graphs are stored using a compressed sparse row (CSR) representation rather than dense adjacency matrices. This keeps network operations memory-efficient while allowing fast access to the neighbours of individual nodes.

Node-level epidemic parameters are stored separately from the graph representation. This keeps network structure independent from susceptibility, infectiousness, recovery, and other individual attributes.

The simulation and analysis code is separated so that epidemic dynamics can be used independently of the EMM experiments.

## Data

The experiments use a synthetic population of 10,000 individuals. Household-size frequencies are calibrated to Dutch population statistics, while the remaining demographic, behavioural, and contact characteristics are generated synthetically.

The experimental setup uses multiple network realisations and repeated stochastic outbreak simulations to evaluate subgroup recovery under known planted transmission effects.

## Reproducibility

Random seeds are explicitly controlled throughout network generation and epidemic simulation. Experimental comparisons use matched seeds where appropriate so that planted and control conditions can be compared under the same stochastic realisations.

The notebook contains the complete experimental pipeline used to generate the reported results and figures.