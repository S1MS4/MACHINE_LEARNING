"""
ND4 – Klasifikavimo algoritmų efektyvumo tyrimas
Duomenys: student_dropout_dataset.csv
Tikslas: numatyti label_name (active / at-risk / dropped)
Algoritmai: KNN, SVM, Gradient Boosted RF
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.utils import resample

# 1. Duomenų nuskaitymas

df = pd.read_csv("../data/student_dropout_dataset.csv")
print(f"Įkelta {df.shape[0]} eilučių, {df.shape[1]} stulpelių")
print(df['label_name'].value_counts().to_string())

# 2. Požymių paruošimas

# region ir exam_season yra kategoriniai – verčiam į skaičius
le_region = LabelEncoder()
df["region_encoded"] = le_region.fit_transform(df["region"])

# pasirenkam požymius – skaitiniai + diskretus (region_encoded, exam_season)
# enroll_date ir student_id praleidžiam – jie nėra informatyvūs
features = [
    "age",
    "courses_enrolled",
    "completed_assignments",
    "completion_rate",
    "login_frequency",
    "last_activity_days_ago",
    "forum_posts_count",
    "exam_season",        # diskretus
    "region_encoded",     # diskretus
]

X = df[features].values
y = df["label_name"].values

# tikslinį kintamąjį verčiam į skaičius
le_cat = LabelEncoder()
y_enc = le_cat.fit_transform(y)

print(f"\nKlasės: {le_cat.classes_}")

# 3. Normalizavimas – MinMaxScaler į [0, 1] (K.1.2)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 4. Balansavimas – oversampling (K.1.3)

# klasės jau gana subalansuotos (~1630-1700), bet vis tiek sulyginam
max_dydis = np.bincount(y_enc).max()
X_parts, y_parts = [], []

for klase in np.unique(y_enc):
    mask = y_enc == klase
    X_k, y_k = X_scaled[mask], y_enc[mask]
    if len(X_k) < max_dydis:
        X_k, y_k = resample(X_k, y_k, replace=True, n_samples=max_dydis, random_state=67)
    X_parts.append(X_k)
    y_parts.append(y_k)

X_bal = np.vstack(X_parts)
y_bal = np.concatenate(y_parts)

print(f"Po balansavimo: {np.bincount(y_bal)} (kiekviena klasė)")

# 5. Kryžminė patikra – StratifiedKFold k=10 (K.1.4)

CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=67)

# 6. Modeliai ir hyperparametrų optimizavimas – GridSearchCV (K.1.5, K.1.6)

modeliai = {
    "KNN": GridSearchCV(
        KNeighborsClassifier(),
        {"n_neighbors": [3, 5, 7, 9, 11], "weights": ["uniform", "distance"], "metric": ["euclidean", "manhattan"]},
        cv=CV, scoring="accuracy", n_jobs=-1,
    ),
    "SVM": GridSearchCV(
        SVC(probability=True, random_state=67),
        {"C": [0.1, 1, 10, 100], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]},
        cv=CV, scoring="accuracy", n_jobs=-1,
    ),
    "Gradient Boosted RF": GridSearchCV(
        GradientBoostingClassifier(random_state=67),
        {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1, 0.2], "subsample": [0.8, 1.0]},
        cv=CV, scoring="accuracy", n_jobs=-1,
    ),
}

# 7. Apmokymas ir tikslumo skaičiavimas (K.1.7)

rezultatai = {}

for pavadinimas, grid in modeliai.items():
    grid.fit(X_bal, y_bal)
    cv_scores = cross_val_score(grid.best_estimator_, X_bal, y_bal, cv=CV, scoring="accuracy")
    rezultatai[pavadinimas] = {
        "modelis": grid.best_estimator_,
        "best_params": grid.best_params_,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "cv_visi": cv_scores,
    }
    print(f"{pavadinimas}: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}  |  {grid.best_params_}")

# 8. Tikslumo diagrama (K.2.3)

pavadinimai = list(rezultatai.keys())
vidurkiai = [rezultatai[p]["cv_mean"] for p in pavadinimai]
std_r = [rezultatai[p]["cv_std"] for p in pavadinimai]
spalvos = ["#4C72B0", "#DD8452", "#55A868"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Studentų dropout – klasifikavimo algoritmų palyginimas", fontweight="bold")

bars = axes[0].bar(pavadinimai, vidurkiai, yerr=std_r, capsize=8, color=spalvos, edgecolor="black")
axes[0].set_ylim(0, 1.1)
axes[0].set_ylabel("Tikslumas (10-fold CV)")
axes[0].set_title("Vidurkiai su paklaidos juostomis")
for bar, val in zip(bars, vidurkiai):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")

bp = axes[1].boxplot([rezultatai[p]["cv_visi"] for p in pavadinimai],
                     labels=pavadinimai, patch_artist=True)
for patch, spalva in zip(bp["boxes"], spalvos):
    patch.set_facecolor(spalva)
    patch.set_alpha(0.7)
axes[1].set_ylabel("Tikslumas")
axes[1].set_title("Tikslumo pasiskirstymas (10 fold)")
axes[1].grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("../results/tikslumo_palyginimas.png", dpi=150, bbox_inches="tight")
plt.close()

# 9. Confusion matricos (K.2.4)

X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=67, stratify=y_bal
)

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Klasifikavimo matricos – geriausi modeliai", fontweight="bold")

for ax, (pavadinimas, d) in zip(axes, rezultatai.items()):
    d["modelis"].fit(X_train, y_train)
    y_pred = d["modelis"].predict(X_test)
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred), display_labels=le_cat.classes_).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{pavadinimas}\nTikslumas: {accuracy_score(y_test, y_pred):.4f}")

plt.tight_layout()
plt.savefig("../results/confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()


# 10. Ataskaita (K.2.1, K.2.2, K.2.5)

print("Normalizavimas: MinMaxScaler [0,1] | Balansavimas: oversampling | CV: StratifiedKFold k=10\n")

for pavadinimas, d in rezultatai.items():
    print(f"{pavadinimas:<22} vidurkis={d['cv_mean']:.4f}  std={d['cv_std']:.4f}")

geriausias = max(rezultatai, key=lambda p: rezultatai[p]["cv_mean"])
print(f"\nGeriausias: {geriausias} ({rezultatai[geriausias]['cv_mean']:.4f})")

geriausias_modelis = rezultatai[geriausias]["modelis"]
geriausias_modelis.fit(X_train, y_train)
print(classification_report(y_test, geriausias_modelis.predict(X_test), target_names=le_cat.classes_))
