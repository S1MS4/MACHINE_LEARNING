# MACHINE_LEARNING (LD1–LD6)

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Selenium](https://img.shields.io/badge/Selenium-43B02A?style=flat&logo=selenium&logoColor=white)
![ChromeDriver](https://img.shields.io/badge/ChromeDriver-4285F4?style=flat&logo=googlechrome&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![Pillow](https://img.shields.io/badge/Pillow-3776AB?style=flat&logo=python&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=flat&logo=scipy&logoColor=white)

Repository with multiple machine learning mini-projects:
- **LD1_SCRAPING**: data scraping/preparation
- **LD2_STATISTICS**: descriptive/inferential statistics + plots
- **LD3_NEURAL_NETWORK**: neural network training/visualization
- **LD4_CLASSIFICATION**: classification model(s) + evaluation results
- **LD5_CLUSTERING**: clustering (e.g., k-means / hierarchical) + images
- **LD6_IMAGE_RECOGNITION**: image recognition pipeline (train/test) + evaluation

## Project structure


### LD1_SCRAPING

**What it does**
LD1_SCRAPING collects data from a car listing website and prepares it for later analysis.
It cleans/sorts the raw data and stores it as CSV files in `data/`.

**Example CSV first rows (CSV header + sample rows)**

<table>
  <tr>
    <th align="left">File</th>
    <th align="left">Pavadinimas</th>
    <th align="left">Metai</th>
    <th align="left">Kėbulo tipas</th>
    <th align="left">Kaina</th>
    <th align="left">Kuras</th>
    <th align="left">Pavarų dėžė</th>
    <th align="left">Variklis</th>
    <th align="left">Rida</th>
    <th align="left">Miestas</th>
    <th align="left">Nuoroda</th>
  </tr>

  <tr>
    <td><code>LD1_SCRAPING/data/BMW 520.csv</code></td>
    <td>BMW 520</td>
    <td>1999-01</td>
    <td>Sedanas</td>
    <td>1 950 €</td>
    <td>Benzinas</td>
    <td>Mechaninė</td>
    <td>2.0 l., 110 kW</td>
    <td>209 441 km</td>
    <td>Marijampolė</td>
    <td><a href="https://autoplius.lt/skelbimai/bmw-520-2-0-l-sedanas-1999-benzinas-28760593.html">link</a></td>
  </tr>
  <tr>
    <td><code>LD1_SCRAPING/data/BMW 520.csv</code></td>
    <td>BMW 520</td>
    <td>2003-09</td>
    <td>Sedanas</td>
    <td>1 998 €</td>
    <td>Benzinas</td>
    <td>Mechaninė</td>
    <td>2.2 l., 125 kW</td>
    <td>261 000 km</td>
    <td>Klaipėda</td>
    <td><a href="https://autoplius.lt/skelbimai/bmw-520-2-2-l-sedanas-2003-benzinas-29887384.html">link</a></td>
  </tr>
  <tr>
    <td><code>LD1_SCRAPING/data/BMW 520.csv</code></td>
    <td>BMW 520</td>
    <td>1995</td>
    <td>Universalas</td>
    <td>1 900 €</td>
    <td>Benzinas</td>
    <td>Mechaninė</td>
    <td>2.0 l., 110 kW</td>
    <td>300 000 km</td>
    <td>Marijampolė</td>
    <td><a href="https://autoplius.lt/skelbimai/bmw-520-2-0-l-universalas-1995-benzinas-29874262.html">link</a></td>
  </tr>
  <tr>
    <td><code>LD1_SCRAPING/data/BMW 520.csv</code></td>
    <td>BMW 520</td>
    <td>2000-05</td>
    <td>Sedanas</td>
    <td>1 550 €</td>
    <td>Dyzelinas</td>
    <td>Mechaninė</td>
    <td>2.0 l., 100 kW</td>
    <td>366 000 km</td>
    <td>Klaipėda</td>
    <td><a href="https://autoplius.lt/skelbimai/bmw-520-2-0-l-sedanas-2000-dyzelinas-29487798.html">link</a></td>
  </tr>
  <tr>
    <td><code>LD1_SCRAPING/data/BMW 520.csv</code></td>
    <td>BMW 520</td>
    <td>1982</td>
    <td>Sedanas</td>
    <td>2 000 €</td>
    <td>Benzinas / dujos</td>
    <td>Mechaninė</td>
    <td>2.0 l.,</td>
    <td>123 456 km</td>
    <td>Alytus</td>
    <td><a href="https://autoplius.lt/skelbimai/bmw-520-2-0-l-sedanas-1982-benzinas-dujos-30010351.html">link</a></td>
  </tr>

  <tr>
    <td><code>LD1_SCRAPING/data/Volkswagen Golf.csv</code></td>
    <td>Volkswagen Golf</td>
    <td>2011-05</td>
    <td>Universalas</td>
    <td>4 700 €</td>
    <td>Dyzelinas</td>
    <td>Mechaninė</td>
    <td>2.0 l., 103 kW</td>
    <td>307 000 km</td>
    <td>Vilnius</td>
    <td><a href="https://autoplius.lt/skelbimai/volkswagen-golf-2-0-l-universalas-2011-dyzelinas-30041115.html">link</a></td>
  </tr>
  <tr>
    <td><code>LD1_SCRAPING/data/Volkswagen Golf.csv</code></td>
    <td>Volkswagen Golf</td>
    <td>2000-06</td>
    <td>Hečbekas</td>
    <td>1 700 €</td>
    <td>Dyzelinas</td>
    <td>Automatinė</td>
    <td>1.9 l., 81 kW</td>
    <td>289 543 km</td>
    <td>Vilnius</td>
    <td><a href="https://autoplius.lt/skelbimai/volkswagen-golf-1-9-l-hecbekas-2000-dyzelinas-30047155.html">link</a></td>
  </tr>
  <tr>
    <td><code>LD1_SCRAPING/data/Volkswagen Golf.csv</code></td>
    <td>Volkswagen Golf</td>
    <td>2009-09</td>
    <td>Hečbekas</td>
    <td>3 600 €</td>
    <td>Benzinas</td>
    <td>Mechaninė</td>
    <td>1.4 l., 90 kW</td>
    <td>226 000 km</td>
    <td>Vilnius</td>
    <td><a href="https://autoplius.lt/skelbimai/volkswagen-golf-1-4-l-hecbekas-2009-benzinas-29939205.html">link</a></td>
  </tr>
  <tr>
    <td><code>LD1_SCRAPING/data/Volkswagen Golf.csv</code></td>
    <td>Volkswagen Golf</td>
    <td>2014-11</td>
    <td>Hečbekas</td>
    <td>6 650 €</td>
    <td>Benzinas</td>
    <td>Automatinė</td>
    <td>1.4 l., 90 kW</td>
    <td>200 800 km</td>
    <td>Trakai</td>
    <td><a href="https://autoplius.lt/skelbimai/volkswagen-golf-1-4-l-hecbekas-2014-benzinas-30008467.html">link</a></td>
  </tr>
  <tr>
    <td><code>LD1_SCRAPING/data/Volkswagen Golf.csv</code></td>
    <td>Volkswagen Golf</td>
    <td>2016</td>
    <td>Universalas</td>
    <td>6 999 €</td>
    <td>Dyzelinas</td>
    <td>Automatinė</td>
    <td>1.6 l., 81 kW</td>
    <td>259 508 km</td>
    <td>Vilnius</td>
    <td><a href="https://autoplius.lt/skelbimai/volkswagen-golf-1-6-l-universalas-2016-dyzelinas-30042425.html">link</a></td>
  </tr>

</table>




**Useful visuals**

- Rate limiting / scraping throttling example: `LD1_SCRAPING/rate_limit.png`
- Sample scraping/parsed output: `LD1_SCRAPING/ND1.png`

![LD1 example image](LD1_SCRAPING/ND1.png)
![LD1 rate limit image](LD1_SCRAPING/rate_limit.png)

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

**All images (LD2_STATISTICS)**

![LD2 overview](LD2_STATISTICS/ND2.png)

**plots in `LD2_STATISTICS/images/`**

![LD2 histogramos](LD2_STATISTICS/images/histogramos.png)
![LD2 boxplots](LD2_STATISTICS/images/boxplots.png)
![LD2 ttest histograms](LD2_STATISTICS/images/ttest_histograms.png)
![LD2 ttest boxplots](LD2_STATISTICS/images/ttest_boxplots.png)

**correlations**

![LD2 koreliacija 1](LD2_STATISTICS/images/koreliacijos/1_hours_with_ai_assistance_daily.png)
![LD2 koreliacija 2](LD2_STATISTICS/images/koreliacijos/2_ai_replaces_vs_burnout.png)
![LD2 koreliacija 3](LD2_STATISTICS/images/koreliacijos/3_satisfaction_vs_experience.png)
![LD2 koreliacija 4](LD2_STATISTICS/images/koreliacijos/4_salary_vs_satisfaction.png)

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

**Useful visuals**

- Network architecture diagram: `LD3_NEURAL_NETWORK/Sources/network_architecture.png`
- Training curves: `LD3_NEURAL_NETWORK/Sources/training_curves.png`
- Quick artifact preview: `LD3_NEURAL_NETWORK/ND3.png`

![LD3 neural network artifact](LD3_NEURAL_NETWORK/ND3.png)

**All images (LD3_NEURAL_NETWORK)**

![LD3 network architecture](LD3_NEURAL_NETWORK/Sources/network_architecture.png)
![LD3 training curves](LD3_NEURAL_NETWORK/Sources/training_curves.png)

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

**Useful visuals**

- Confusion matrices + learning curves are saved in `LD4_CLASSIFICATION/results/`
- Quick artifact preview: `LD4_CLASSIFICATION/ND4.png`

![LD4 classification artifact](LD4_CLASSIFICATION/ND4.png)

**All images (LD4_CLASSIFICATION)**

![LD4 confusion matrices](LD4_CLASSIFICATION/results/confusion_matrices.png)
![LD4 learning curves](LD4_CLASSIFICATION/results/mokymosi_kreives.png)
![LD4 decision boundary](LD4_CLASSIFICATION/results/sprendimu_ribos.png)
![LD4 accuracy comparison](LD4_CLASSIFICATION/results/tikslumo_palyginimas.png)

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

**Useful visuals**

- Example cluster visualization: `LD5_CLUSTERING/ND5.png`
- More clustering visuals: `LD5_CLUSTERING/pictures/`

![LD5 clustering artifact](LD5_CLUSTERING/ND5.png)

**All images (LD5_CLUSTERING)**
![LD5 hierarchical methods](LD5_CLUSTERING/pictures/hierarchiniai_metodai.png)

<table>
  <tr>
    <td align="left">
      <img src="LD5_CLUSTERING/pictures/sample20/klasteriai_voronoi20.png" alt="LD5 sample20 clusters voronoi" />
    </td>
    <td align="right">
      <img src="LD5_CLUSTERING/pictures/sample20/klasterizavimas_rezultatai20.png" alt="LD5 sample20 clustering results" />
    </td>
  </tr>
</table>

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

**Useful visuals**

- Quick artifact preview: `LD6_IMAGE_RECOGNITION/ND6.png`
- Confusion matrix: `LD6_IMAGE_RECOGNITION/images/confusion_matrix.png`
- Learning curve: `LD6_IMAGE_RECOGNITION/images/learning_curve.png`

![LD6 image recognition artifact](LD6_IMAGE_RECOGNITION/ND6.png)

**All images (LD6_IMAGE_RECOGNITION)**

![LD6 confusion matrix](LD6_IMAGE_RECOGNITION/images/confusion_matrix.png)
![LD6 learning curve](LD6_IMAGE_RECOGNITION/images/learning_curve.png)

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

