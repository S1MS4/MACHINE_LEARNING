import csv
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import Voronoi
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage


# ==============================================================
# K.1 – DUOMENYS
# ==============================================================

# K.1.1 + K.1.2 – Užkraunam CSV failą su lietuviškais tekstais.
# Duomenų aibę sudaro bent 20 dokumentų, keturių temų (bent 5 kiekviena).
print("Užkraunam duomenis...")

tekstai = []
temos = []
pavadinimai = []

with open("../duomenys/duomenys160.csv", encoding="utf-8") as f:
    for eilute in csv.DictReader(f):
        tekstai.append(eilute["tekstas"])
        temos.append(eilute["tema"])
        pavadinimai.append(eilute["pavadinimas"][:40])

print(f"Užkrauta {len(tekstai)} dokumentų\n")


# ==============================================================
# K.2 – TEKSTO APDOROJIMO METODAI
# ==============================================================

# K.2.2 – Lietuviški stop words sąrašas: dažni žodžiai, kurie nieko nereiškia analizei.
STOP_WORDS = {
    "ir", "bei", "ar", "kad", "į", "iš", "su", "be", "po", "per", "prie",
    "nuo", "iki", "už", "ant", "pas", "tai", "yra", "buvo", "bus", "būtų",
    "jis", "ji", "jie", "jos", "mes", "jūs", "tu", "aš", "jo", "jos",
    "jam", "ją", "jų", "juos", "mums", "jums", "tave", "save",
    "kuris", "kuri", "kurie", "kurios", "kuriame", "kurioje",
    "bet", "tačiau", "arba", "nei", "nors", "nes", "jei", "jeigu",
    "kaip", "taip", "tiek", "labai", "dar", "jau", "net", "tik", "vis",
    "savo", "šis", "ši", "šie", "šios", "tas", "ta", "tie", "tos",
    "to", "tą", "tuo", "tame", "šio", "šią", "šiuo", "šiame",
    "vienas", "viena", "visi", "visas", "visa", "pats", "pati",
    "apie", "dėl", "prieš", "pagal", "tarp", "virš", "šalia",
    "jog", "kad", "ką", "ko", "kam", "kuo", "kur", "kada", "kaip",
    "na", "o", "a", "nu", "gi", "va", "ne", "taigi", "pvz", "t",
    "d", "g", "s", "m", "n", "j", "k", "l", "p", "r", "v",
    "balandžio", "gegužės", "kovo", "vasario", "sausio", "birželio",
    "liepos", "rugpjūčio", "rugsėjo", "spalio", "lapkričio", "gruodžio",
    "metų", "metu", "metais", "dienos", "dieną", "dienomis",
    "sakė", "teigė", "pasakė", "pridūrė", "pažymėjo", "kalbėjo",
    "nurodė", "informavo", "pranešė", "pabrėžė", "komentavo",
}


# K.2.1 – Trys teksto apdorojimo filtrai: mažosios raidės, simbolių šalinimas, trumpų žodžių šalinimas.
# K.2.2 – Taikomas stop words sąrašas.
# K.2.3 – Išmetamos lietuviškos kabutės ir kiti netipiniai ženklai.
def apdoroti_teksta(tekstas):
    # Filtras 1 (K.2.1): viską paverčiam mažosiomis raidėmis
    tekstas = tekstas.lower()

    # Filtras 2 (K.2.3): išmetam lietuviškas ir kitas kabutes
    tekstas = re.sub(r'[„""\'«»]', '', tekstas)

    # Filtras 3 (K.2.1): paliekam tik raides ir tarpus, šalinam skaičius ir skyrybą
    tekstas = re.sub(r'[^a-ząčęėįšųūž\s]', ' ', tekstas)

    # Filtras 4 (K.2.1): šalinam žodžius trumpesnius nei 3 raidės
    zodziai = tekstas.split()
    zodziai = [z for z in zodziai if len(z) >= 3]

    # Filtras 5 (K.2.2): šalinam stop words iš sąrašo
    zodziai = [z for z in zodziai if z not in STOP_WORDS]

    return ' '.join(zodziai)


print("Apdorojam tekstus...")
apdoroti = [apdoroti_teksta(t) for t in tekstai]

print("Prieš:", tekstai[0][:100])
print("Po:   ", apdoroti[0][:100])
print()


# ==============================================================
# TF-IDF VEKTORIZACIJA (paruošiamasis žingsnis prieš K.3)
# ==============================================================

# TF-IDF paverčia tekstus į skaičių vektorius – retesnis žodis konkrečiame dokumente gauna didesnį svorį.
print("Vektorizuojam (TF-IDF)...")

# min_df=2: žodis turi būti bent 2 dokumentuose
# max_df=0.9: žodis negali būti daugiau nei 90% dokumentų (per bendras)
# max_features=1000: naudojam daugiausiai 1000 žodžių
vektorizeris = TfidfVectorizer(
    min_df=2,
    max_df=0.9,
    max_features=1000,
    ngram_range=(1, 2), # tik atskiri žodžiai, ne kombinacijos
)

X = vektorizeris.fit_transform(apdoroti)
print(f"Matrica: {X.shape[0]} dokumentai x {X.shape[1]} žodžių\n")


# ==============================================================
# K.3.2 + K.3.3 – K-VIDURKIŲ (K-MEANS) KLASTERIZAVIMAS
# ==============================================================

# K.3.2 – Taikomas k-vidurkių metodas dokumentams suskirstyti į grupes.
# K.3.3 – Bandome skirtingus k parametrus (3, 4, 5) ir lyginame rezultatus.
print("=" * 50)
print("K.3.2 + K.3.3 – K-MEANS KLASTERIZAVIMAS")
print("=" * 50)

for k in [3, 4, 5]:
    kmeans = KMeans(n_clusters=k, random_state=67, n_init=10)
    kmeans.fit(X)
    print(f"\nk={k} klasteriai:")
    for klasteri_id in range(k):
        indeksai = [i for i, l in enumerate(kmeans.labels_) if l == klasteri_id]
        temu_skaic = {}
        for i in indeksai:
            temu_skaic[temos[i]] = temu_skaic.get(temos[i], 0) + 1
        print(f"  Klasteris {klasteri_id} ({len(indeksai)} dok.): {temu_skaic}")

# K.3.3 – Pasirenkam k=4, nes geriausiai atitinka 4 turimas temas ir duoda aiškiausią atskyrimą.
print("\nK.3.3 – Renkamės k=4: geriausiai atitinka 4 temas, klasteriai aiškiausiai atsiskiria")
kmeans_galutinis = KMeans(n_clusters=4, random_state=67, n_init=10)
kmeans_galutinis.fit(X)


# ==============================================================
# K.3.1 + K.3.3 – HIERARCHINIS KLASTERIZAVIMAS
# ==============================================================

# K.3.1 – Taikomas hierarchinis klasterizavimas, kuris grupuoja dokumentus pagal panašumą žingsnis po žingsnio.
# K.3.3 – Bandome du sujungimo metodus (ward, complete) ir lyginame.
print("\n" + "=" * 50)
print("K.3.1 + K.3.3 – HIERARCHINIS KLASTERIZAVIMAS")
print("=" * 50)

X_dense = X.toarray()

for metodas in ["ward", "complete"]:
    # "ward" – sujungia klasterius, kurie mažiausiai padidina variaciją
    # "complete" – sujungia klasterius, kurių didžiausias atstumas tarp taškų yra mažiausias, nėra chainsų
    hier = AgglomerativeClustering(n_clusters=4, linkage=metodas)
    hier_etiketės = hier.fit_predict(X_dense)
    print(f"\nMetodas: {metodas}")
    for klasteri_id in range(4):
        indeksai = [i for i, l in enumerate(hier_etiketės) if l == klasteri_id]
        temu_skaic = {}
        for i in indeksai:
            temu_skaic[temos[i]] = temu_skaic.get(temos[i], 0) + 1
        print(f"  Klasteris {klasteri_id} ({len(indeksai)} dok.): {temu_skaic}")

# K.3.3 – Abu metodai duoda panašius rezultatus su šiais duomenimis.
# Renkamės 'ward' kaip standartinį pasirinkimą tekstų klasterizavimui,
# nes bent sporto tema atsiskiria į grynesnį klasterį.


# ==============================================================
# K.3.4 – PANAŠUMO MATAS TARP DVIEJŲ DOKUMENTŲ
# ==============================================================

# K.3.4 – Skaičiuojam kosinuso panašumą tarp dokumentų: 1.0 = identiški, 0.0 = visiškai skirtingi.
print("\n" + "=" * 50)
print("K.3.4 – PANAŠUMO MATAS (Cosine Similarity)")
print("=" * 50)

def geriausias_panasumo_pora(tema1, tema2, X, temos):
    #Randa panašiausią dokumentų porą tarp dviejų temų.
    idx1 = [i for i, t in enumerate(temos) if t == tema1]
    idx2 = [i for i, t in enumerate(temos) if t == tema2]
    geriausias = (idx1[0], idx2[0], -1)
    for i in idx1:
        for j in idx2:
            if i == j:
                continue
            p = cosine_similarity(X[i], X[j])[0][0]
            if p > geriausias[2]:
                geriausias = (i, j, p)
    return geriausias[0], geriausias[1]

# Skirtingos temos – ieškome panašiausios poros (tikimės mažo skaičiaus)
sporto_idx, verslo_idx = geriausias_panasumo_pora("sportas", "verslas", X, temos)

# Ta pati tema – ieškome panašiausios poros (tikimės didelio skaičiaus)
sporto_idx, sporto_idx2 = geriausias_panasumo_pora("sportas", "sportas", X, temos)

panasum_skirtingi = cosine_similarity(X[sporto_idx], X[verslo_idx])[0][0]
panasum_vienodi   = cosine_similarity(X[sporto_idx], X[sporto_idx2])[0][0]

print(f"\nDoc 1 (sportas): '{pavadinimai[sporto_idx]}'")
print(f"Doc 2 (verslas): '{pavadinimai[verslo_idx]}'")
print(f"Panašumas (skirtingos temos): {panasum_skirtingi:.4f}")

print(f"\nDoc 1 (sportas): '{pavadinimai[sporto_idx]}'")
print(f"Doc 2 (sportas): '{pavadinimai[sporto_idx2]}'")
print(f"Panašumas (ta pati tema):     {panasum_vienodi:.4f}")

print("\nKuo arčiau 1.0 – tuo panašesni. Kuo arčiau 0 – tuo labiau skiriasi.")


# ==============================================================
# K.3.5 – VIZUALIZACIJOS (padeda paaiškinti gautus rezultatus)
# ==============================================================

# K.3.5 – Du grafikai: k-means rezultatų stulpelinė diagrama ir hierarchinio klasterizavimo dendrograma.
print("\nKuriame grafikus...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- Grafika 1: K-means rezultatai (K.3.2 + K.3.5) ---
ax1 = axes[0]
spalvos = {"sportas": "#e74c3c", "verslas": "#3498db",
           "veidai": "#2ecc71", "laisvalaikis": "#f39c12"}

k_etiketės = kmeans_galutinis.labels_
klasteriu_temos = {}
for i, (tema, klasteris) in enumerate(zip(temos, k_etiketės)):
    if klasteris not in klasteriu_temos:
        klasteriu_temos[klasteris] = {}
    klasteriu_temos[klasteris][tema] = klasteriu_temos[klasteris].get(tema, 0) + 1

kategorijos = list(spalvos.keys())
x_pos = np.arange(4)
bar_width = 0.2

for j, tema in enumerate(kategorijos):
    skaičiai = [klasteriu_temos.get(k, {}).get(tema, 0) for k in range(4)]
    ax1.bar(x_pos + j * bar_width, skaičiai, bar_width,
            label=tema, color=spalvos[tema], alpha=0.85)

ax1.set_xlabel("Klasteris")
ax1.set_ylabel("Dokumentų skaičius")
ax1.set_title("K.3.2 – K-means (k=4): dokumentų pasiskirstymas")
ax1.set_xticks(x_pos + bar_width * 1.5)
ax1.set_xticklabels(["Klasteris 0", "Klasteris 1", "Klasteris 2", "Klasteris 3"])
ax1.legend()
ax1.grid(axis="y", alpha=0.3)

# --- Grafika 2: Dendrograma (K.3.1 + K.3.5) ---
ax2 = axes[1]

trumpi_pavad = []
for i, (pav, tema) in enumerate(zip(pavadinimai, temos)):
    trumpi_pavad.append(f"[{tema[:3].upper()}] {pav[:20]}")

Z = linkage(X_dense, method="ward")
dendrogram(Z, labels=trumpi_pavad, orientation="right", ax=ax2,
           leaf_font_size=7, color_threshold=Z[-3, 2])
ax2.set_title("K.3.1 – Hierarchinis klasterizavimas (Ward metodas)")
ax2.set_xlabel("Atstumas")
n = len(tekstai)
plt.tight_layout()
plt.savefig(f"../pictures/sample{n}/klasterizavimas_rezultatai{n}1.png", dpi=150, bbox_inches="tight")
print("Grafikai išsaugoti: klasterizavimas_rezultatai.png")

plt.show()


# ==============================================================
# K.3.5 – SCATTER PLOT SU VORONOI DIAGRAMOMIS (PCA 2D projekcija)
# ==============================================================

# K.3.5 – Trečias grafikas: sumažinam 1000 dimensijų į 2 naudodami PCA.
# Voronoi diagrama padalija erdvę į sritis pagal artimiausius k-means centroidus –
# kiekvienas taškas patenka į to centroido sritį, kuriam yra arčiausiai.
print("Kuriame scatter plot su Voronoi diagramomis (PCA 2D)...")

pca = PCA(n_components=2, random_state=67)
X2D = pca.fit_transform(X_dense)

klasteriai = kmeans_galutinis.labels_

TEMU_SPALVOS = {
    "sportas":       "#e74c3c",
    "verslas":       "#3498db",
    "veidai":        "#2ecc71",
    "laisvalaikis":  "#f39c12",
}


def dominuojanti_spalva(k):
    """Grąžina dominuojančios temos spalvą ir pavadinimą duotam klasteriui."""
    indeksai = np.where(klasteriai == k)[0]
    temu_skaic = {}
    for i in indeksai:
        temu_skaic[temos[i]] = temu_skaic.get(temos[i], 0) + 1
    dominuojanti = max(temu_skaic, key=temu_skaic.get)
    return TEMU_SPALVOS[dominuojanti], dominuojanti


def voronoi_sritys_ribos(vor, ax):
    """
    Grąžina Voronoi srities daugiakampius apkarpytus pagal grafiko ribas.
    Naudoja pint_regions ir vertices iš scipy Voronoi objekto.
    Begalinės kraštinės pratęsiamos už brėžinio ribų.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Plotis naudojamas begalinių kraštinių pratęsimui
    plotis = max(xlim[1] - xlim[0], ylim[1] - ylim[0]) * 2

    poligonai = []
    for point_idx, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if not region:
            poligonai.append(None)
            continue

        if -1 not in region:
            # Uždara sritis – tiesiog imame viršūnes
            poligonai.append(vor.vertices[region])
        else:
            # Atvira sritis – reikia pratęsti begalines kraštines
            virsunes = []
            ridge_points = vor.ridge_points
            ridge_vertices = vor.ridge_vertices

            for i, (p1, p2) in enumerate(ridge_points):
                if point_idx not in (p1, p2):
                    continue
                rv = ridge_vertices[i]
                if -1 not in rv:
                    virsunes.extend(vor.vertices[rv])
                else:
                    # Begalinė kraštinė – skaičiuojame kryptį
                    fin = rv[0] if rv[1] == -1 else rv[1]
                    kaimynas = p2 if p1 == point_idx else p1
                    tangent = vor.points[kaimynas] - vor.points[point_idx]
                    normal = np.array([-tangent[1], tangent[0]])
                    normal /= np.linalg.norm(normal)
                    vidurio = vor.vertices[fin]
                    # Pasirenkame kryptį toliau nuo centro
                    if np.dot(normal, vidurio - vor.points[point_idx]) < 0:
                        normal = -normal
                    virsunes.append(vidurio)
                    virsunes.append(vidurio + normal * plotis)

            if len(virsunes) < 3:
                poligonai.append(None)
                continue

            # Surikiuojame viršūnes pagal kampą aplink centroidą
            virsunes = np.array(virsunes)
            centras = virsunes.mean(axis=0)
            kampai = np.arctan2(virsunes[:, 1] - centras[1],
                                virsunes[:, 0] - centras[0])
            virsunes = virsunes[np.argsort(kampai)]
            poligonai.append(virsunes)

    return poligonai


fig2, ax3 = plt.subplots(figsize=(13, 9))
ax3.set_facecolor("#f8f9fa")
fig2.patch.set_facecolor("#f8f9fa")

# Pirma nustatome grafiko ribas pagal duomenų diapazoną su šiek tiek paraščių
margin = 0.15
x_min, x_max = X2D[:, 0].min(), X2D[:, 0].max()
y_min, y_max = X2D[:, 1].min(), X2D[:, 1].max()
x_range = x_max - x_min
y_range = y_max - y_min
ax3.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
ax3.set_ylim(y_min - margin * y_range, y_max + margin * y_range)

# Skaičiuojame k-means centroidų 2D projekcijas:
# transformuojame originalius centroidus per tą patį PCA modelį
centroidai_2d = pca.transform(kmeans_galutinis.cluster_centers_)

# Voronoi diagrama pagal centroidų pozicijas 2D erdvėje.
# Pridedame "tolimuosius taškus" keturiuose kampuose, kad visos sritys būtų uždaros –
# tai užtikrina, kad kraštiniai klasteriai taip pat turės apibrėžtą sritį.
kampo_taškai = np.array([
    [x_min - x_range, y_min - y_range],
    [x_min - x_range, y_max + y_range],
    [x_max + x_range, y_min - y_range],
    [x_max + x_range, y_max + y_range],
])
vor_taškai = np.vstack([centroidai_2d, kampo_taškai])
vor = Voronoi(vor_taškai)

# Piešiame Voronoi sritis – kiekviena sritis nuspalvinama pagal dominuojančią temą
poligonai = voronoi_sritys_ribos(vor, ax3)

for k in range(4):
    spalva, dom_tema = dominuojanti_spalva(k)
    pol = poligonai[k]
    if pol is None or len(pol) < 3:
        continue
    patch = Polygon(pol, closed=True,
                    facecolor=spalva, alpha=0.18,
                    edgecolor=spalva, linewidth=1.5, linestyle="--",
                    zorder=1)
    ax3.add_patch(patch)

    # Klasterio etiketė prie centroido
    cx, cy = centroidai_2d[k]
    ax3.text(cx, cy, f"Klasteris {k}\n({dom_tema})",
             ha="center", va="center", fontsize=9,
             color=spalva, style="italic", fontweight="bold", zorder=5)

# Centroidai pažymimi kryžiukais
ax3.scatter(centroidai_2d[:, 0], centroidai_2d[:, 1],
            marker="x", s=150, linewidths=2.5,
            color="black", zorder=6, label="Centroidai")

# Duomenų taškai – spalva pagal TIKRĄ temą (ne klasterį) – taip matosi klasifikavimo klaidos
for tema, spalva in TEMU_SPALVOS.items():
    indeksai = [i for i, t in enumerate(temos) if t == tema]
    ax3.scatter(
        X2D[indeksai, 0],
        X2D[indeksai, 1],
        c=spalva,
        label=tema,
        s=120,
        edgecolors="white",
        linewidths=1.2,
        zorder=4,
    )

ax3.set_title(
    "K.3.5 – Dokumentų klasteriai (PCA 2D + Voronoi)\n"
    "Taško spalva = tikra tema | Voronoi sritis = K-means klasteris",
    fontsize=13, pad=15
)
ax3.set_xlabel("PCA 1 komponentė", fontsize=10)
ax3.set_ylabel("PCA 2 komponentė", fontsize=10)
ax3.legend(title="Tema", fontsize=10, title_fontsize=10,
           loc="upper right", framealpha=0.9)
ax3.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(f"../pictures/sample{n}/klasteriai_voronoi{n}1.png", dpi=150, bbox_inches="tight")
print("Išsaugota: klasteriai_scatter.png")

plt.show()
print("\nViskas baigta!")