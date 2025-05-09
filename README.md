# SuperpositionGNN

**Code accompanying the paper "Superposition in GNNs"**
*Lukas Pertl, supervised by Pietro Lio, May 6, 2025*

A framework implementing the synthetic graph experiments, GNN architectures, and geometric analyses described in the paper. This repository contains all scripts and utilities to reproduce the key findings:

* Superposition via compression and the impact of pooling functions
* Topology-driven superposition without explicit bottlenecks
* Metastable low-rank minima in GIN models and hyperplane phenomena
* Empirical studies on real-world binary classification tasks

## Table of Contents

* [Paper]
* [Introduction](#introduction)
* [Features](#features)
* [Installation](#installation)
* [Quickstart with `main.py`](#quickstart-with-mainpy)
* [Experimental Pipeline](#experimental-pipeline)
* [Project Structure](#project-structure)
* [Key Components](#key-components)

  * [GraphGeneration.py](#graphgenerationpy)
  * [Model.py](#modelpy)
  * [Trainer.py](#trainerpy)
  * [Runner.py](#runnerpy)
  * [ExperimentalPipeline1.py](#experimentalpipeline1py)
* [Visualization & Animation](#visualization--animation)
* [Configuration Files](#configuration-files)
* [Logging & TensorBoard](#logging--tensorboard)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)

## Paper

The full write-up of these experiments is available in the PDF:

```
???
```

It covers the definitions, datasets, theoretical background, and results (Figures, Tables) for:

1. Compression-induced superposition and pooling effects
2. Topology-driven superposition in motif tasks
3. Metastable low-rank minima and hyperplane transitions
4. Practical implications for real-world graph classification

## Introduction

**SuperpositionGNN** provides an end-to-end setup—from synthetic graph generation through model training, evaluation, and geometric analysis—ideal for studying GNN behavior under controlled conditions.

## Features

* Synthetic graph generators for chain, correlated, motif, combined, and count modes
* Flexible GCN and GIN model definitions with custom pooling and equiangular frame initialization
* Weighted loss handling and two-phase training (freeze/unfreeze final layer)
* Configurable experimental pipelines via Excel or YAML inputs
* Automated logging to TensorBoard and JSON result exports
* Geometry analysis: singular-value decomposition, collapsed embedding detection
* Visualization helpers for static plots and animations

## Installation

```bash
git clone https://github.com/LukasPertl1/SuperpositionGNN.git
cd SuperpositionGNN
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Quickstart with `main.py`

`main.py` provides two entry points:

1. **Motif-focused experiment** (default):

   ```bash
   python main.py
   ```

   Uses `ExperimentalPipeline1.main()` to run specified rows and mode (e.g., motif).

2. **Simple-mode training demo**:

   * Uncomment and rename the second `__main__` block tag to `__main__ == '__main__'` (SIMPLE).
   * Generates synthetic "simple" chain graphs, trains a GCN model for 5 epochs, evaluates, and plots 3D embeddings via `Visualizer.plot_avg_hidden_embeddings_3d()`.

Modify parameters like `specific_rows` and `Mode` at the top of `main.py` as needed.

## Experimental Pipeline

For large-scale, reproducible experiments, use `ExperimentalPipeline1.py`:

```bash
python ExperimentalPipeline1.py
```

This script:

1. Reads combinations from Excel (e.g., `ExperimentList/combinations.xlsx`)
2. Builds configurations (`simple`, `motif`, `count`) based on `specific_rows` and `Mode`
3. Calls `run_single_experiment(config)` for each
4. Saves JSON summaries under `experiment_results/` when `save=True`

## Project Structure

```
SuperpositionGNN/
├─ ExperimentList/          # Excel sheets with parameter combinations
├─ Extras/                  # Auxiliary scripts
├─ GraphsForReport/         # Figures for publications
├─ experiment_results/      # JSON outputs of experiments
├─ Annimation.py            # Animation utilities
├─ ExperimentalPipeline1.py # Main pipeline driver
├─ GraphGeneration.py       # Synthetic graph generators
├─ Model.py                 # GNNModel and pooling helpers
├─ PipelineUtils.py         # Helper functions (e.g., get_hidden_dims)
├─ ReaderWriter.py          # Logging & JSON writers
├─ Runner.py                # Batch experiment runner
├─ Trainer.py               # Training, evaluation, geometry analysis
├─ Visualizer.py            # Plotting utilities
├─ main.py                  # Quick-start script
├─ requirements.txt
└─ README.md
```

## Key Components

### GraphGeneration.py

* **SyntheticGraphDataGenerator**: Modes: `simple`, `correlated`, `motif`, `combined`, `count`.
* Chain, motif, and random-edge generators.
* Feature computation (one-hot, correlated, count-based).
* `sparcity_calculator` for initial embedding sparsity estimation.

### Model.py

* **GNNModel**: Choose `model_type` (`GCN` or `GIN`), specify `hidden_dims`, `pooling` (`mean`, `max`, `gm`).
* Equiangular frame initialization for final layer.
* Supports freezing/unfreezing final layer for two-phase training.

### Trainer.py

* **Trainer** class handles epoch loops, loss weighting, evaluation, and logging.
* Two-phase training: freeze final layer, then unfreeze for fine-tuning.
* `evaluate()` returns loss, accuracy, prediction/embedding dicts, and empty-graph stats.
* Geometry analysis: `svd_analysis`, `geometry_of_representation`, summarization utilities.

### Runner.py

* **run\_multiple\_experiments()**: Wrapper to loop over experiments, instantiate generator, model, and trainer.
* Saves per-experiment results, model parameters, and average embeddings.
* Supports BCE or MSE losses based on config.

### ExperimentalPipeline1.py

* Builds `base_config_*` dictionaries for each mode.
* Reads Excel sheets (`combinations.xlsx`, `motif_combinations.xlsx`, `count_combinations.xlsx`).
* Generates and adjusts configs via `get_hidden_dims` and other helpers.
* Calls `run_single_experiment()` and saves JSON outputs.

## Visualization & Animation

* Static plots via `Visualizer.py` (training curves, embedding scatter).
* Animations via `Annimation.py` (graph evolution, embedding trajectories).

## Configuration Files

Place experiment definitions in:

* `ExperimentList/combinations.xlsx`
* `ExperimentList/motif_combinations.xlsx`
* `ExperimentList/count_combinations.xlsx`

Use `PipelineUtils.py` hooks to customize dimension lookups and naming.

## Logging & TensorBoard

* Logs saved under `runs/<mode>/<architecture>/...`
* Scalars: loss, accuracy, singular values.
* Graph visualizations, embeddings logged if enabled in config.

## Contributing

1. Fork and clone the repo.
2. Create a feature branch: `git checkout -b feature/YourFeature`.
3. Commit and push changes.
4. Open a pull request.

## License

This project is MIT licensed. See [LICENSE](LICENSE) for details.

## Contact

Questions or issues? Open a GitHub issue or reach out to Lukas Pertl (@LukasPertl1) on GitHub.





