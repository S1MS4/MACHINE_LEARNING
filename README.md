# MACHINE_LEARNING (LD1–LD6)

Repository with multiple machine learning mini-projects:
- **LD1_SCRAPING**: data scraping/preparation
- **LD2_STATISTICS**: descriptive/inferential statistics + plots
- **LD3_NEURAL_NETWORK**: neural network training/visualization
- **LD4_CLASSIFICATION**: classification model(s) + evaluation results
- **LD5_CLUSTERING**: clustering (e.g., k-means / hierarchical) + images
- **LD6_IMAGE_RECOGNITION**: image recognition pipeline (train/test) + evaluation

## How to upload everything to one GitHub repo
From the repository root:
```bash
git init
git add .
git commit -m "Initial commit: add LD1–LD6 project folders"
git branch -M main
git remote add origin <YOUR_REPO_URL>
git push -u origin main
```

## Project structure

### LD1_SCRAPING
**What it does**
LD1_SCRAPING collects data from a car listing website and prepares it for later analysis.
It cleans/sorts the raw data and stores it as CSV files in `data/`.

**What it contains**
- `data/` – scraped datasets (CSV files)
- `scripts/` – scripts for preparing/cleaning/sorting data

**Entry scripts**
- `scripts/script.py`
- `scripts/sort.py`


### LD2_STATISTICS
**What it does**
LD2_STATISTICS explores the dataset and computes descriptive/inferential statistics.
It also generates plots in `images/` for histograms, boxplots, and correlations.

**What it contains**
- `data/` – datasets used in statistical analysis
- `images/` – generated plots (histograms, boxplots, correlation figures)
- `sources/` – supporting material (including stat terminology notes)
- `scripts/` – statistical computations / data preparation


**Entry scripts**
- `scripts/main.py`
- `scripts/populate.py`
- `scripts/K1.py`, `scripts/K2.py`

### LD3_NEURAL_NETWORK
**What it does**
LD3_NEURAL_NETWORK trains a simple neural network on the prepared dataset.
It saves/loads artifacts and visualizes training curves and the learned network.

**What it contains**
- `Sources/` – dataset and saved artifacts (e.g., `weights.json`, diagrams)
- `Scripts/` – implementation + training/evaluation/visualization

**Entry scripts**
- `Scripts/neural_network.py`
- `Scripts/plot_training.py`
- `Scripts/visualize_neural_network.py`


### LD4_CLASSIFICATION
**What it does**
LD4_CLASSIFICATION trains a classification model and evaluates it on the dataset.
It generates evaluation outputs like confusion matrices and learning/accuracy plots in `results/`.

**What it contains**
- `data/` – classification dataset
- `scripts/` – training/evaluation and visualization
- `results/` – saved evaluation outputs (confusion matrices, learning curves, plots)


**Entry scripts**
- `scripts/main.py`
- `scripts/visualize.py`
- `scripts/kaggle.py`
- `scripts/unique.py`

### LD5_CLUSTERING
**What it does**
LD5_CLUSTERING applies clustering algorithms (like k-means and hierarchical) to the datasets.
It produces visualizations of cluster assignments and results in `pictures/`.

**What it contains**
- `duomenys/` – clustering datasets
- `scripts/` – scraping/selection + clustering experiments
- `pictures/` – generated clustering visuals (k-means/hierarchical, samples)


**Entry scripts**
- `scripts/main.py`
- `scripts/scrape.py`
- `scripts/parinkimas.py`
- `scripts/fix_space.py`

### LD6_IMAGE_RECOGNITION
**What it does**
LD6_IMAGE_RECOGNITION trains an image recognition model using a learning set and evaluates it on a test set.
It saves evaluation visuals (like a confusion matrix and learning curve) into `images/`.

**What it contains**
- `data/learning_set/` – training images grouped by class
- `data/testing_set/` – test images grouped by class
- `images/` – evaluation results (confusion matrix, learning curve)
- `scripts/` – training/inference script

**Entry scripts**
- `scripts/main.py`


## Running projects
This repo contains multiple independent sub-projects (LD1–LD6). Run the entry script inside the relevant folder, for example:
- `python LD3_NEURAL_NETWORK/Scripts/neural_network.py`
- `python LD4_CLASSIFICATION/scripts/main.py`
- `python LD6_IMAGE_RECOGNITION/scripts/main.py`

(Exact dependencies/commands depend on the code inside each sub-project.)

