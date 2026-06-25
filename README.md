# SpArX-healthcare-explainability

Bachelor thesis project exploring group-wise SpArX explanations for neural networks trained on tabular healthcare datasets.

## Overview

This project trains a multilayer perceptron (MLP) on three binary classification healthcare datasets and evaluates whether group-wise SpArX explanation models — built separately for patient subgroups — are more faithful than a single global SpArX model.

The pipeline is split across two notebooks:

- **Phase 1** — data preparation, model training, evaluation, and artifact saving
- **Phase 2 & 3** — patient grouping via KMeans clustering, global SpArX baseline construction, and group-wise SpArX explanation evaluation

## Datasets

Three publicly available datasets are included in the `data/` folder:

| Dataset | File | Source |
|---|---|---|
| Heart Disease (Cleveland) | `processed.cleveland.data` | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/45/heart+disease) |
| Pima Indians Diabetes | `diabetes.csv` | [Kaggle / UCI](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) |
| Breast Cancer WDBC | `wdbc.data` | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) |

## Repository Structure

```text
SpArX-healthcare-explainability/
├── data/
│   ├── processed.cleveland.data
│   ├── diabetes.csv
│   └── wdbc.data
├── 01_phase1_model_training.ipynb
├── 02_phase2_3_groupwise_sparx.ipynb
├── requirements.txt
├── README.md
└── LICENSE
```

Artifact folders are generated at runtime and are not committed to the repository. After running the notebooks, the expected generated structure is:

```text
artifacts/
├── heart_disease/
│   ├── phase1_model/
│   ├── phase2_candidate_groups/
│   └── phase3_sparx_evaluation/
├── diabetes/
│   ├── phase1_model/
│   ├── phase2_candidate_groups/
│   └── phase3_sparx_evaluation/
└── breast_cancer_wdbc/
    ├── phase1_model/
    ├── phase2_candidate_groups/
    └── phase3_sparx_evaluation/
```

## Requirements

- Python 3.9+
- PyTorch
- scikit-learn
- pandas
- numpy
- matplotlib
- joblib
- TensorFlow / Keras (required by SpArX, installed automatically in Phase 2)
- SpArX (cloned and patched automatically in Phase 2, no manual installation needed)

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Environment Setup

At the top of each notebook there is an environment setup cell. Set `BASE_DIR` to point to the project folder depending on where you are running:

**Google Colab with Google Drive:**
```python
BASE_DIR = Path("/content/gdrive/MyDrive/path/to/SpArX-healthcare-explainability")
DATA_DIR = BASE_DIR / "data"
```

**Local machine (Jupyter, PyCharm, VSCode):**
```python
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
```

In Colab, also run the Drive mount cell directly below the setup cell before proceeding.

## Notebook Run Order

The notebooks must be run in order because Phase 2 and Phase 3 depend on artifacts saved during Phase 1.

1. Run `01_phase1_model_training.ipynb` for each dataset by changing `dataset_name`:
   - `heart_disease`
   - `diabetes`
   - `breast_cancer_wdbc`

2. After Phase 1 artifacts have been generated for all three datasets, run `02_phase2_3_groupwise_sparx.ipynb` for each dataset in the same way.

## How to Run

### Option A — Google Colab

1. Upload the repository folder to Google Drive.
2. Open `01_phase1_model_training.ipynb` in Colab.
3. In the environment setup cell, set `BASE_DIR` to your Drive path (see above).
4. Run the Drive mount cell below it.
5. Run the notebook once for each dataset by changing `dataset_name`.
6. Open `02_phase2_3_groupwise_sparx.ipynb`, set the same `BASE_DIR`, and run for each dataset.

### Option B — Local (Jupyter, PyCharm, VSCode)

1. Clone the repository:

   ```bash
   git clone https://github.com/rsarbu/SpArX-healthcare-explainability.git
   cd SpArX-healthcare-explainability
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. In the environment setup cell of each notebook, set `BASE_DIR = Path(".")` (see above).
4. Skip the Google Drive mount cell.
5. Run `01_phase1_model_training.ipynb` first for all three datasets, then run `02_phase2_3_groupwise_sparx.ipynb`.

> **Important:** Phase 2 and Phase 3 depend on artifacts saved by Phase 1. Always run the notebooks in order. If the `artifacts/` folder is deleted, Phase 1 must be rerun.

## Outputs

The notebooks save the following under `artifacts/`:

- Trained MLP model weights and configuration
- Fitted preprocessing pipeline
- Processed train/test arrays and labels
- Sample identifiers
- KMeans clustering summaries and assignments
- PCA embeddings
- SpArX faithfulness and complexity evaluation tables

These outputs reproduce the results reported in the thesis, including MLP classification performance, Phase 2 clustering summaries, global SpArX baseline results, and group-wise SpArX faithfulness comparisons.

## Reproducibility

Random seeds are fixed at 42 where applicable, including PyTorch, NumPy, Python `random`, scikit-learn `train_test_split`, KMeans, and PCA. Running the notebooks from top to bottom with the same dataset files and package versions will reproduce the reported results.

Small numerical differences may occur across hardware, operating systems, or library versions.

## License

MIT License — see [LICENSE](LICENSE) for details.
