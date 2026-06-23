
# STATISTINIŲ METODŲ TERMINAI
## 2 paskaita | doc. dr. Pavel Stefanovič

---

## 1. STATISTIKOS TIPAI

| Lietuviškai | Angliškai | Apibūdinimas |
|:------------|:-----------|:--------------|
| **Aprašomoji statistika** | Descriptive statistics | Apibūdina duomenų savybes skaičiais ar grafikais |
| **Inferentinė statistika** | Inferential statistics | Daro išvadas apie populiaciją remiantis imtimi |

---

## 2. KINTAMIEJI

| Lietuviškai | Angliškai | Apibūdinimas |
|:------------|:-----------|:--------------|
| **Kintamasis** | Variable | Matuojama savybė (lytis, amžius, pajamos) |
| **Kokybinis** | Qualitative | *Ar* savybė egzistuoja (lytis, tautybė) |
| **Kiekybinis** | Quantitative | *Kiek* savybės yra (pajamos, amžius) |
| **Tolydusis** | Continuous | Gali būti skaidomas (svoris, ūgis) |
| **Diskretusis** | Discrete | Sveikieji skaičiai (vaikų skaičius) |

---

## 3. MATAVIMŲ SKALĖS

| Skalė | Angliškai | Apibūdinimas | Pavyzdžiai |
|:------|:----------|:--------------|:------------|
| **Nominali** | Nominal | Kategorijos, be operacijų | Vyras/Moteris |
| **Rangų** | Ordinal | Rikiavimas, ne skirtumai | I, II, III vieta |
| **Intervalų** | Interval | Rikiuotė + skirtumai, nėra absoliutaus nulio | Temperatūra °C |
| **Santykių** | Ratio | Turi absoliutų nulį | Svoris, ūgis |

---

## 4. APRAŠOMOSIOS STATISTIKOS TERMINAI

| Terminas | Angliškai | Apibūdinimas | Formulė |
|:---------|:-----------|:--------------|:---------|
| **Vidurkis** | Mean | Reikšmių suma / kiekis | μ = Σxᵢ/n |
| **Mediana** | Median | Vidurinė reikšmė | - |
| **Moda** | Mode | Dažniausia reikšmė | - |
| **Kvartilinis plotis** | IQR | Q3 - Q1 (viduriniai 50%) | IQR = Q₃ - Q₁ |
| **Imties plotis** | Range | Max - Min | R = xₘₐₓ - xₘᵢₙ |
| **Dispersija** | Variance | σ² = Σ(xᵢ - μ)² / N | σ² = Σ(xᵢ-μ)²/N |
| **Standartinis nuokrypis** | SD | √σ² | σ = √σ² |

### Asimetrija (Skewness)
```
As > 0 → dešininė (uodega dešinėn)
As < 0 → kairinė (uodega kairėn)
As = 0 → simetriškas
```

### Ekscesas (Kurtosis)
```
Ek > 0 → smailus (reikšmės centre)
Ek < 0 → plokščias (reikšmės išsibarsčiusios)
```

---

## 5. HIPOTEZIŲ TIKRINIMAS

| Terminas | Žymėjimas | Apibūdinimas |
|:---------|:----------|:--------------|
| **Nulinė hipotezė** | H₀ | Teiginys, kad skirtumo nėra |
| **Alternatyvioji** | H₁ | Priešinga H₀ |
| **Reikšmingumo lygmuo** | α = 0.05 | 5% klaidos tikimybė |
| **p-reikšmė** | p-value | Tikimybė, kai H₀ teisinga |

### Sprendimo taisyklė
```
p < 0.05 → H₀ atmetama (yra skirtumas)
p ≥ 0.05 → H₀ neatmetama (nėra skirtumo)
```

---

## 6. HIPOTEZIŲ TIPAI

| Tipas | Kada naudoti | Metodai |
|:------|:-------------|:---------|
| **Parametriniai** | Normalūs duomenys, intervalų skalė | t-test, ANOVA, Pearson |
| **Neparametriniai** | Nenormalūs duomenys, rangų skalė | Wilcoxon, Spearman |

---

## 7. T-TESTAS

| Variantas | Apibūdinimas | Formulė |
|:----------|:--------------|:---------|
| **Vienos imties** | Vidurkis vs konkreti reikšmė | t = (x̄ - μ) / (s/√n) |
| **Nepriklausomų imčių** | Dvi skirtingos grupės | t = (x̄₁ - x̄₂) / √(s₁²/n₁ + s₂²/n₂) |
| **Priklausomų imčių** | Tie patys matuoti 2 kartus | t = d̄ / (s_d/√n) |

---

## 8. KORELIACIJA

| Reikšmė | Interpretacija |
|:--------|:----------------|
| **r ≈ 0** | Silpna / nėra |
| **r → 1** | Stipri teigiama (abu didėja) |
| **r → -1** | Stipri neigiama (vienas didėja, kitas mažėja) |

| Tipas | Kada naudoti |
|:------|:--------------|
| **Pearson** | Tolygiesiems, normaliems |
| **Spearman** | Rangų duomenims |

---

*Šaltinis: doc. dr. Pavel Stefanovič — Statistiniai metodai ir jų taikymas (2 paskaita)*