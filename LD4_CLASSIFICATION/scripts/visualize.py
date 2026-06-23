"""
visualize.py – papildomos vizualizacijos prie main.py
Paleidimas: python visualize.py
Reikalavimai: main.py turi būti tame pačiame aplanke
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, GridSearchCV, learning_curve
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import resample
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# 1. Duomenų paruošimas (identiška main.py logika)

df = pd.read_csv("../data/student_dropout_dataset.csv")

le_region = LabelEncoder()
df["region_encoded"] = le_region.fit_transform(df["region"])

features = [
    "age", "courses_enrolled", "completed_assignments", "completion_rate",
    "login_frequency", "last_activity_days_ago", "forum_posts_count",
    "exam_season", "region_encoded",
]

X = df[features].values
y = df["label_name"].values

le_cat = LabelEncoder()
y_enc = le_cat.fit_transform(y)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

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

CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=67)

# Geriausi parametrai iš main.py rezultatų
modeliai = {
    "KNN": KNeighborsClassifier(n_neighbors=11, weights="distance", metric="manhattan"),
    "SVM": SVC(C=100, kernel="rbf", gamma="scale", probability=True, random_state=67),
    "Gradient Boosted RF": GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.2, subsample=0.8, random_state=67
    ),
}

klasių_spalvos = {0: "#4C72B0", 1: "#DD8452", 2: "#55A868"}  # active, at-risk, dropped
klasių_pavadinimai = le_cat.classes_  # ['active', 'at-risk', 'dropped']

# 2. PCA – sumažinam iki 2D sprendimų ribų vizualizacijai

pca = PCA(n_components=2, random_state=67)
X_2d = pca.fit_transform(X_bal)
var_exp = pca.explained_variance_ratio_

print(f"PCA: PC1={var_exp[0]:.1%}, PC2={var_exp[1]:.1%}, iš viso={sum(var_exp):.1%} variacijos")

# Sprendimų ribų tinklelis
x_min, x_max = X_2d[:, 0].min() - 0.3, X_2d[:, 0].max() + 0.3
y_min, y_max = X_2d[:, 1].min() - 0.3, X_2d[:, 1].max() + 0.3
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
tinklelis = np.c_[xx.ravel(), yy.ravel()]

# 3. Sprendimų ribų diagrama

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle(
    f"Sprendimų ribos – PCA 2D projekcija\n"
    f"(PC1={var_exp[0]:.1%} + PC2={var_exp[1]:.1%} = {sum(var_exp):.1%} variacijos)",
    fontweight="bold", fontsize=13
)

fono_spalvos = ["#AEC6E8", "#F5C9A0", "#A8D5B0"]  # švelnios klasių spalvos fonui

for ax, (pavadinimas, modelis) in zip(axes, modeliai.items()):
    # Apmokome 2D erdvėje
    modelis.fit(X_2d, y_bal)
    Z = modelis.predict(tinklelis).reshape(xx.shape)

    # Spalvotas fonas pagal prognozę
    for klase_id, spalva in enumerate(fono_spalvos):
        ax.contourf(xx, yy, Z == klase_id, levels=[0.5, 1.5],
                    colors=[spalva], alpha=0.45)

    # Ribų linijos
    ax.contour(xx, yy, Z, levels=[0.5, 1.5], colors="white", linewidths=1.2, alpha=0.8)

    # Duomenų taškai (imame 300 atsitiktinių iš kiekvienos klasės – geriau matomumas)
    rng = np.random.default_rng(42)
    for klase_id in np.unique(y_bal):
        mask = y_bal == klase_id
        idx = rng.choice(np.where(mask)[0], size=min(300, mask.sum()), replace=False)
        ax.scatter(X_2d[idx, 0], X_2d[idx, 1],
                   c=klasių_spalvos[klase_id], s=8, alpha=0.5,
                   edgecolors="none", label=klasių_pavadinimai[klase_id])

    ax.set_title(pavadinimas, fontweight="bold", fontsize=11)
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1%})", fontsize=9)
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1%})", fontsize=9)
    ax.tick_params(labelsize=8)

    # Legenda tik pirmame grafike
    if ax == axes[0]:
        legendos = [mpatches.Patch(color=klasių_spalvos[i], label=klasių_pavadinimai[i])
                    for i in range(3)]
        ax.legend(handles=legendos, fontsize=8, loc="upper right",
                  framealpha=0.85, edgecolor="gray")

plt.tight_layout()
plt.savefig("../results/sprendimu_ribos.png", dpi=150, bbox_inches="tight")
plt.close()
print("Išsaugota: sprendimu_ribos.png")

# 4. Mokymosi kreivės

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Mokymosi kreivės – tikslumas priklausomai nuo mokymo duomenų kiekio",
             fontweight="bold", fontsize=13)

# Dydžiai: nuo 10% iki 100% balansavimo duomenų
train_sizes = np.linspace(0.10, 1.0, 10)

for ax, (pavadinimas, modelis), spalva in zip(axes, modeliai.items(),
                                               ["#4C72B0", "#DD8452", "#55A868"]):
    train_sz, train_sc, val_sc = learning_curve(
        modelis, X_bal, y_bal,
        train_sizes=train_sizes,
        cv=CV,
        scoring="accuracy",
        n_jobs=-1,
        shuffle=True,
        random_state=67,
    )

    train_mean = train_sc.mean(axis=1)
    train_std  = train_sc.std(axis=1)
    val_mean   = val_sc.mean(axis=1)
    val_std    = val_sc.std(axis=1)

    # Mokymo kreivė
    ax.plot(train_sz, train_mean, "o-", color=spalva, linewidth=2,
            markersize=5, label="Mokymas")
    ax.fill_between(train_sz, train_mean - train_std, train_mean + train_std,
                    alpha=0.15, color=spalva)

    # Validavimo kreivė
    ax.plot(train_sz, val_mean, "s--", color=spalva, linewidth=2,
            markersize=5, alpha=0.7, label="Validavimas (CV)")
    ax.fill_between(train_sz, val_mean - val_std, val_mean + val_std,
                    alpha=0.10, color=spalva)

    # Galutinis CV tikslumas (horizontali punktyrinė linija)
    galutinis = val_mean[-1]
    ax.axhline(galutinis, color="gray", linewidth=0.8, linestyle=":",
               label=f"Galutinis: {galutinis:.4f}")

    ax.set_title(pavadinimas, fontweight="bold", fontsize=11)
    ax.set_xlabel("Mokymo imties dydis (įrašai)", fontsize=9)
    ax.set_ylabel("Tikslumas", fontsize=9)
    ax.set_ylim(0.60, 1.05)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, framealpha=0.85, edgecolor="gray")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("../results/mokymosi_kreives.png", dpi=150, bbox_inches="tight")
plt.close()
print("Išsaugota: mokymosi_kreives.png")

print("\nVizualizacija baigta. Failai:")
print("  sprendimu_ribos.png")
print("  mokymosi_kreives.png")