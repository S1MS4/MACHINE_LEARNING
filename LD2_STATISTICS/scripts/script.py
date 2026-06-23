import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

plt.style.use('ggplot')
sns.set_palette("husl")

# Įkeliam csv failą iš projekto aplanko
df = pd.read_csv(r'../data/ai_worker_burnout_attrition_2026.csv')

print("=" * 60)
print("Duomenys sėkmingai įkelti!")
print(f"Eilučių skaičius: {len(df)}, Stulpelių skaičius: {len(df.columns)}")
print("=" * 60)


# #K.1.1 DUOMENŲ MANIPULIAVIMAS IR FILTRAVIMAS
print("\n" + "=" * 60)
print("#K.1.1 DUOMENŲ MANIPULIAVIMAS IR FILTRAVIMAS")
print("=" * 60)

# Filtruojam darbuotojus kurių perdegimo balas didesnis nei 70
high_burnout = df[df['burnout_score'] > 70]
print(f"\n[Filtras 1] Darbuotojai su aukštu perdegimu (burnout_score > 70):")
print(f"  Rasta darbuotojų: {len(high_burnout)}")
print(high_burnout[['employee_id', 'job_role', 'burnout_score', 'country']].to_string(index=False))

# Filtruojam darbuotojus iš USA ir Indijos
selected_countries = df[df['country'].isin(['USA', 'India'])]
print(f"\n[Filtras 2] Darbuotojai iš USA ir Indijos:")
print(f"  Rasta darbuotojų: {len(selected_countries)}")
print(selected_countries[['employee_id', 'job_role', 'country', 'burnout_score']].to_string(index=False))

# Filtruojam darbuotojus kurie naudoja ChatGPT
ai_tool_users = df[df['primary_ai_tool'] == 'ChatGPT']
print(f"\n[Filtras 3] Darbuotojai kurie naudoja ChatGPT:")
print(f"  Rasta darbuotojų: {len(ai_tool_users)}")
print(ai_tool_users[['employee_id', 'job_role', 'primary_ai_tool', 'burnout_score']].to_string(index=False))

# Grupuojam pagal profesiją
print(f"\n[Grupavimas] Vidutinis burnout_score pagal profesiją:")
burnout_by_role = df.groupby('job_role')['burnout_score'].mean().sort_values(ascending=False)
print(burnout_by_role.round(2).to_string())

# Išrenkam top 5 profesijas
print(f"\n[Top 5] Profesijos su didžiausiu vidutiniu perdegimu:")
top5_burnout = burnout_by_role.head(5)
for i, (role, score) in enumerate(top5_burnout.items(), 1):
    print(f"  {i}. {role}: {score:.2f}")


# #K.1.2 IŠSAMI APRAŠOMOJI STATISTIKA
print("\n" + "=" * 60)
print("#K.1.2 IŠSAMI APRAŠOMOJI STATISTIKA")
print("=" * 60)

# Pasirenkam 5 skaitinius kintamuosius
skaitiniai = ['burnout_score', 'productivity_score', 'salary_usd_k',
              'hours_with_ai_assistance_daily', 'ai_replaces_my_tasks_pct']

# Skaičiuojam išsamią statistiką
for col in skaitiniai:
    print(f"\n--- {col} ---")
    d = df[col].dropna()
    moda_val = d.mode()
    moda_str = ', '.join([f"{v:.2f}" for v in moda_val.values]) if len(moda_val) <= 3 else f"{moda_val.iloc[0]:.2f} (ir kiti)"
    print(f"  Vidurkis (mean):          {d.mean():.2f}")
    print(f"  Mediana (median):         {d.median():.2f}")
    print(f"  Moda (mode):              {moda_str}")
    print(f"  Std. nuokrypis (std):     {d.std():.2f}")
    print(f"  Dispersija (var):         {d.var():.2f}")
    print(f"  Minimumas:                {d.min():.2f}")
    print(f"  Maksimumas:               {d.max():.2f}")
    print(f"  Q1 (25%):                 {d.quantile(0.25):.2f}")
    print(f"  Q2 (50%):                 {d.quantile(0.50):.2f}")
    print(f"  Q3 (75%):                 {d.quantile(0.75):.2f}")
    print(f"  Asimetrija (skew):        {d.skew():.2f}")
    print(f"  Ekscesas (kurtosis):      {d.kurtosis():.2f}")

# Kategoriniai kintamieji
print(f"\n--- Kategoriniai kintamieji ---")
print(f"\n  Profesijos (job_role):")
print(df['job_role'].value_counts().to_string())
print(f"\n  Šalys (country):")
print(df['country'].value_counts().to_string())
print(f"\n  Darbo tipas (remote_work_type):")
print(df['remote_work_type'].value_counts().to_string())
print(f"\n  AI pakeitimo baimė (fear_of_ai_replacement):")
print(df['fear_of_ai_replacement'].value_counts().to_string())


# #K.1.3 BOX PLOTAI
print("\n" + "=" * 60)
print("#K.1.3 BOX PLOTAI")
print("=" * 60)

# Dinaminės reikšmės box plotams
burnout_data = df['burnout_score'].dropna()
salary_data = df['salary_usd_k'].dropna()

b_min = burnout_data.min()
b_max = burnout_data.max()
b_med = burnout_data.median()
b_q1 = burnout_data.quantile(0.25)
b_q3 = burnout_data.quantile(0.75)
b_iqr = b_q3 - b_q1
b_outliers = burnout_data[(burnout_data < b_q1 - 1.5*b_iqr) | (burnout_data > b_q3 + 1.5*b_iqr)]

s_min = salary_data.min()
s_max = salary_data.max()
s_med = salary_data.median()
s_q1 = salary_data.quantile(0.25)
s_q3 = salary_data.quantile(0.75)
s_iqr = s_q3 - s_q1
s_outliers = salary_data[(salary_data < s_q1 - 1.5*s_iqr) | (salary_data > s_q3 + 1.5*s_iqr)]

# Kuriame box plotus
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("Box Plot'ai - Perdegimo balas ir Atlyginimas", fontsize=14, fontweight='bold')

# Perdegimo balas
axes[0].boxplot(burnout_data, patch_artist=True,
                boxprops=dict(facecolor='lightcoral', color='darkred'),
                medianprops=dict(color='darkred', linewidth=2))
axes[0].set_title('Perdegimo balo pasiskirstymas')
axes[0].set_xlabel('Visi darbuotojai')
axes[0].set_ylabel('Perdegimo balas (burnout_score)')
axes[0].set_xticklabels([''])  # Vietoj "1"

# Atlyginimas
axes[1].boxplot(salary_data, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='darkblue'),
                medianprops=dict(color='darkblue', linewidth=2))
axes[1].set_title('Atlyginimo pasiskirstymas')
axes[1].set_xlabel('Visi darbuotojai')
axes[1].set_ylabel('Atlyginimas (salary_usd_k, tūkst. USD)')
axes[1].set_xticklabels([])  # Vietoj "1"

plt.tight_layout()
plt.savefig('../images/boxplots.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"  [Box plots išsaugoti kaip boxplots.png]")
print(f"  Perdegimo balas: mediana apie {b_med:.0f}, reikšmės nuo {b_min:.0f} iki {b_max:.0f}, rasta {len(b_outliers)} išskirčių.")
print(f"  Atlyginimas: platus pasiskirstymas, mediana apie {s_med:.0f}k, reikšmės nuo {s_min:.0f}k iki {s_max:.0f}k, rasta {len(s_outliers)} išskirčių.")


# #K.1.4 HISTOGRAMOS
print("\n" + "=" * 60)
print("#K.1.4 HISTOGRAMOS")
print("=" * 60)

# Dinaminės reikšmės histogramoms
b_skew = burnout_data.skew()
b_kurt = burnout_data.kurtosis()
s_skew = salary_data.skew()

# Kuriame histogramas
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Histogramos - Perdegimo balas ir Atlyginimas', fontsize=14, fontweight='bold')

# Perdegimo balas
sns.histplot(burnout_data, kde=True, ax=axes[0],
             color='coral', edgecolor='darkred', bins=15)
axes[0].set_title('Perdegimo balo pasiskirstymas')
axes[0].set_xlabel('Perdegimo balas (burnout_score)')
axes[0].set_ylabel('Dažnis')

# Atlyginimas
sns.histplot(salary_data, kde=True, ax=axes[1],
             color='steelblue', edgecolor='darkblue', bins=15)
axes[1].set_title('Atlyginimo pasiskirstymas')
axes[1].set_xlabel('Atlyginimas (salary_usd_k, tūkst. USD)')
axes[1].set_ylabel('Dažnis')

plt.tight_layout()
plt.savefig('../images/histogramos.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"  [Histogramos išsaugotos kaip histogramos.png]")
print(f"  Perdegimo balas: centras apie {b_med:.0f}, asimetrija = {b_skew:.2f} - {'artimas normaliam' if abs(b_skew) < 0.5 else 'asimetriškas'}.")
print(f"  Atlyginimas: {('pasvirimas į dešinę (right-skewed)' if s_skew > 0.5 else 'pasvirimas į kairę (left-skewed)' if s_skew < -0.5 else 'simetriškas')} - neprimena normalaus.")


# #K.1.5 REZULTATŲ PAAIŠKINIMAS
print("\n" + "=" * 60)
print("#K.1.5 REZULTATŲ PAAIŠKINIMAS")
print("=" * 60)

burnout = burnout_data
salary = salary_data

# Dinaminės išvados
print(f"\nPERDEGIMO BALAS (burnout_score):")
vid_skirtumas = abs(burnout.mean() - burnout.median())
print(f"  Vidurkis ({burnout.mean():.2f}) ir mediana ({burnout.median():.2f}) yra {'artimi' if vid_skirtumas < 2 else 'skiriasi'}.")
print(f"  Tai reiškia kad pasiskirstymas {'gana simetriškas' if vid_skirtumas < 2 else 'turi asimetriją'}.")
print(f"  Std. nuokrypis = {burnout.std():.2f} rodo vidutinį išsibarstymą.")
asim_b = burnout.skew()
if asim_b > 0.5:
    asim_text = "dešininės (ilgesnė dešinė uodega)"
elif asim_b < -0.5:
    asim_text = "kairinės (ilgesnė kairė uodega)"
else:
    asim_text = "pasiskirstymas gana simetriškas"
print(f"  Asimetrija = {asim_b:.2f}: {asim_text}.")
kurt_b = burnout.kurtosis()
if kurt_b > 0:
    kurt_text = "skirstinys smailus (daug reikšmių centre)"
else:
    kurt_text = "skirstinys plokščias (reikšmės labiau išsibarsčiusios)"
print(f"  Ekscesas = {kurt_b:.2f}: {kurt_text}.")
Q1_b = burnout.quantile(0.25)
Q3_b = burnout.quantile(0.75)
IQR_b = Q3_b - Q1_b
issk_b = burnout[(burnout < Q1_b - 1.5*IQR_b) | (burnout > Q3_b + 1.5*IQR_b)]
print(f"  Box plot išskirtys: {'yra ' + str(len(issk_b)) + ' išskirčių' if len(issk_b) > 0 else 'nėra ryškių išskirčių'}.")
print(f"  Histograma: {'primena normalųjį skirstinį' if abs(asim_b) < 0.5 and abs(kurt_b) < 1 else 'nenormalusis - pastebima asimetrija arba ekscesas'}.")

print(f"\nATLYGINIMAS (salary_usd_k):")
print(f"  Vidurkis ({salary.mean():.2f}) {'didesnis' if salary.mean() > salary.median() else 'mažesnis'} už medianą ({salary.median():.2f}).")
print(f"  Tai reiškia kad {'keli aukšti atlyginimai traukia vidurkį į viršų' if salary.mean() > salary.median() else 'keli žemi atlyginimai traukia vidurkį į apačią'}.")
print(f"  Std. nuokrypis = {salary.std():.2f} - atlyginimų išsibarstumas {'didelis' if salary.std() > 50 else 'vidutinis' if salary.std() > 20 else 'mažas'}.")
asim_s = salary.skew()
if asim_s > 0.5:
    asim_text = "dešininės - yra darbuotojų su labai aukštais atlyginimais"
elif asim_s < -0.5:
    asim_text = "kairinės asimetrija - yra darbuotojų su labai žemais atlyginimais"
else:
    asim_text = "simetriškas"
print(f"  Asimetrija = {asim_s:.2f}: {asim_text}.")
kurt_s = salary.kurtosis()
print(f"  Ekscesas = {kurt_s:.2f}: {'skirstinys smailus' if kurt_s > 0 else 'skirstinys plokščias - reikšmės tolygiau paskirstytos'}.")
print(f"  Histograma: pasiskirstymas {'pasviręs į dešinę' if asim_s > 0.5 else 'pasviręs į kairę' if asim_s < -0.5 else 'simetriškas'} - {'neprimena' if abs(asim_s) > 0.5 else 'gali priminti'} normalaus skirstinio.")


# #K.2.1 T-TESTAI
print("\n" + "=" * 60)
print("#K.2.1 T-TESTAI")
print("=" * 60)

# T-testas nr.1: nepriklausomų imčių
print("\n[T-testas Nr.1] Nepriklausomų imčių t-testas:")
print("  Lyginame burnout_score tarp grupių: High vs Low fear_of_ai_replacement")
print("  H0: abiejų grupių perdegimo vidurkiai yra vienodi")
print("  H1: vidurkiai skiriasi")

grupe_high = df[df['fear_of_ai_replacement'] == 'High']['burnout_score'].dropna()
grupe_low  = df[df['fear_of_ai_replacement'] == 'Low']['burnout_score'].dropna()

print(f"\n  Grupė 'High' (n={len(grupe_high)}): vidurkis = {grupe_high.mean():.2f}")
print(f"  Grupė 'Low'  (n={len(grupe_low)}): vidurkis = {grupe_low.mean():.2f}")

t1, p1 = stats.ttest_ind(grupe_high, grupe_low)
print(f"\n  t-statistika: {t1:.4f}")
print(f"  p-reikšmė:    {p1:.4f}")
if p1 < 0.05:
    print("  Išvada: Atmetame H0 - yra statistiškai reikšmingas skirtumas")
    print(f"  Praktiškai: Darbuotojai kurie labai bijo AI pakeitimo ({grupe_high.mean():.2f}) "
          f"{'turi didesnį' if grupe_high.mean() > grupe_low.mean() else 'turi mažesnį'} "
          f"perdegimą nei tie kurie nebijo ({grupe_low.mean():.2f}).")
else:
    print("  Išvada: Negalime atmesti H0 - nėra statistiškai reikšmingo skirtumo")

# T-testas nr.2: vienos imties
print("\n" + "-" * 40)
print("[T-testas Nr.2] Vienos imties t-testas:")
print("  Lyginame Data Scientist produktyvumą su visų darbuotojų vidurkiu")
print("  H0: Data Scientist vidurkis = visų darbuotojų vidurkis")
print("  H1: vidurkiai skiriasi")

visu_vidurkis = df['productivity_score'].mean()
data_scientists = df[df['job_role'] == 'Data Scientist']['productivity_score'].dropna()

print(f"\n  Visų darbuotojų produktyvumo vidurkis: {visu_vidurkis:.2f}")
print(f"  Data Scientist vidurkis (n={len(data_scientists)}): {data_scientists.mean():.2f}")

t2, p2 = stats.ttest_1samp(data_scientists, visu_vidurkis)
print(f"\n  t-statistika: {t2:.4f}")
print(f"  p-reikšmė:    {p2:.4f}")
if p2 < 0.05:
    print("  Išvada: Atmetame H0 - yra statistiškai reikšmingas skirtumas")
    print(f"  Praktiškai: Data Scientist ({data_scientists.mean():.2f}) yra statistiškai "
          f"{'produktyvesni' if data_scientists.mean() > visu_vidurkis else 'mažiau produktyvūs'} "
          f"nei tipinis darbuotojas ({visu_vidurkis:.2f}).")
else:
    print("  Išvada: Negalime atmesti H0 - nėra statistiškai reikšmingo skirtumo")


# #K.2.2 KORELIACIJA
print("\n" + "=" * 60)
print("#K.2.2 KORELIACIJA")
print("=" * 60)

print("  Analizuojame: hours_with_ai_assistance_daily vs burnout_score")

xy_df = df[['hours_with_ai_assistance_daily', 'burnout_score']].dropna()
x_clean = xy_df['hours_with_ai_assistance_daily']
y_clean = xy_df['burnout_score']

spearman_r, spearman_p = stats.spearmanr(x_clean, y_clean)
pearson_r, pearson_p = stats.pearsonr(x_clean, y_clean)

print(f"\n  Spirmeno koreliacija: r = {spearman_r:.4f}, p = {spearman_p:.4f}")
print(f"  Pirsono koreliacija:  r = {pearson_r:.4f},  p = {pearson_p:.4f}")

# Scatter plot
plt.figure(figsize=(8, 6))
sns.regplot(data=xy_df, x='hours_with_ai_assistance_daily', y='burnout_score',
            scatter_kws={'alpha': 0.6, 'color': 'steelblue'},
            line_kws={'color': 'red', 'linewidth': 2})
plt.title(f"Valandos su AI vs Perdegimo balas\n(Spearman r={spearman_r:.3f}, p={spearman_p:.3f})")
plt.xlabel('Valandos su AI pagalba per dieną')
plt.ylabel('Perdegimo balas')
plt.tight_layout()
plt.savefig('../images/koreliacija_scatter.png', dpi=150, bbox_inches='tight')
plt.show()
print("  [Scatter plot išsaugotas kaip koreliacija_scatter.png]")


# #K.2.3 T-TESTO REZULTATŲ INTERPRETACIJA
print("\n" + "=" * 60)
print("#K.2.3 T-TESTO REZULTATŲ INTERPRETACIJA")
print("=" * 60)

print(f"\nT-TESTAS Nr.1 (nepriklausomų imčių - High vs Low baimė):")
print(f"  t-statistika = {t1:.4f}")
print(f"  p-reikšmė    = {p1:.4f}")
if p1 < 0.05:
    print("  Išvada: Atmetame H0 - yra statistiškai reikšmingas skirtumas (p < 0.05)")
    print("  Praktiškai: Darbuotojai kurie labai bijo būti pakeisti AI turi statistiškai")
    print("  reikšmingai skirtingą perdegimo lygį.")
else:
    print("  Išvada: Negalime atmesti H0 - nėra statistiškai reikšmingo skirtumo (p >= 0.05)")

print(f"\nT-TESTAS Nr.2 (vienos imties - Data Scientists vs visi):")
print(f"  t-statistika = {t2:.4f}")
print(f"  p-reikšmė    = {p2:.4f}")
if p2 < 0.05:
    print("  Išvada: Atmetame H0 - yra statistiškai reikšmingas skirtumas (p < 0.05)")
    print("  Praktiškai: Data Scientist produktyvumas statistiškai reikšmingai skiriasi")
    print("  nuo visos įmonės vidurkio.")
else:
    print("  Išvada: Negalime atmesti H0 - nėra statistiškai reikšmingo skirtumo (p >= 0.05)")


# #K.2.4 KORELIACIJOS INTERPRETACIJA
print("\n" + "=" * 60)
print("#K.2.4 KORELIACIJOS INTERPRETACIJA")
print("=" * 60)

abs_r = abs(spearman_r)
if abs_r < 0.3:
    stiprumas = "silpnas"
elif abs_r < 0.6:
    stiprumas = "vidutinis"
else:
    stiprumas = "stiprus"

zenklas = "teigiamas (+)" if spearman_r > 0 else "neigiamas (-)"
reiksmingumas = "statistiškai reikšmingas (p < 0.05)" if spearman_p < 0.05 else "statistiškai nereikšmingas (p >= 0.05)"

print(f"\nSPIRMENO KORELIACIJOS REZULTATAI:")
print(f"  Koeficientas r = {spearman_r:.4f}")
print(f"  p-reikšmė      = {spearman_p:.4f}")
print(f"\n  1. Koeficiento ženklas: {zenklas}")
print(f"     Tai reiškia: {'daugiau valandų su AI -> aukštesnis perdegimas' if spearman_r > 0 else 'daugiau valandų su AI -> mažesnis perdegimas'}.")
print(f"\n  2. Ryšio stiprumas: {stiprumas} (|r| = {abs_r:.3f})")
if abs_r < 0.3:
    print("     Ryšys silpnas - valandos su AI menkai susijusios su perdegimu.")
elif abs_r < 0.6:
    print("     Ryšys vidutinis - pastebima tendencija, bet nėra labai stipri.")
else:
    print("     Ryšys stiprus - aiški tendencija.")
print(f"\n  3. Statistinis reikšmingumas: {reiksmingumas}")
if spearman_p < 0.05:
    print("     Ryšys yra tikras, o ne atsitiktinis (p < 0.05).")
else:
    print("     Negalime teigti kad ryšys yra tikras.")
print(f"\n  4. Praktinė išvada:")
if spearman_p < 0.05:
    print(f"     Yra statistiškai reikšmingas ryšys - darbuotojai kurie daugiau laiko")
    print(f"     praleidžia su AI pagalba turi {'aukštesnį' if spearman_r > 0 else 'žemesnį'} perdegimo balą.")
else:
    print("     Statistiškai reikšmingo ryšio nerasta - negalima teigti kad valandos")
    print("     su AI tiesiogiai veikia perdegimą šiame duomenų rinkinyje.")
print("=" * 60)

# #K.2.5 KORELIACIJA: Job Satisfaction vs Years of Experience
print("\n" + "=" * 60)
print("#K.2.5 KORELIACIJA: Job Satisfaction vs Years of Experience")
print("=" * 60)

print("  Analizuojame: years_experience vs job_satisfaction_1_5")

# Išvalome duomenis - pašaliname trūkstamas reikšmes
corr_df = df[['years_experience', 'job_satisfaction_1_5']].dropna()
x_clean = corr_df['years_experience']
y_clean = corr_df['job_satisfaction_1_5']

# Apskaičiuojame koreliacijas
spearman_r, spearman_p = stats.spearmanr(x_clean, y_clean)
pearson_r, pearson_p = stats.pearsonr(x_clean, y_clean)

print(f"\n  Spirmeno koreliacija: r = {spearman_r:.4f}, p = {spearman_p:.4f}")
print(f"  Pirsono koreliacija:  r = {pearson_r:.4f},  p = {pearson_p:.4f}")
print(f"  Koreliacijai naudota: n = {len(corr_df)}")

# Scatter plot su jitter
plt.figure(figsize=(10, 6))

# Pridedame mažą atsitiktinį jitter, kad geriau matytųsi tankumas
np.random.seed(42)  # kad rezultatai būtų atkuriami
x_jitter = x_clean + np.random.normal(0, 0.3, size=len(x_clean))
y_jitter = y_clean + np.random.normal(0, 0.05, size=len(y_clean))

plt.scatter(x_jitter, y_jitter, alpha=0.4, color='steelblue', s=30, label='Darbuotojai')

# Pridedame regresijos liniją
z = np.polyfit(x_clean, y_clean, 1)
p = np.poly1d(z)
plt.plot(sorted(x_clean), p(sorted(x_clean)), 
         color='red', linewidth=2, label=f'Tendencijos linija')

plt.title(f'Darbo pasitenkinimas vs Patirtis\n(Spearman r={spearman_r:.3f}, p={spearman_p:.4f})', 
          fontsize=14, fontweight='bold')
plt.xlabel('Patirtis metais (years_experience)')
plt.ylabel('Darbo pasitenkinimas (job_satisfaction_1_5)')
plt.yticks([1, 2, 3, 4, 5])
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('../images/satisfaction_vs_experience.png', dpi=150, bbox_inches='tight')
plt.show()

print("  [Scatter plot išsaugotas kaip satisfaction_vs_experience.png]")

# #K.2.6 KORELIACIJA: Atlyginimas vs Darbo pasitenkinimas
print("\n" + "=" * 60)
print("#K.2.6 KORELIACIJA: Atlyginimas vs Darbo pasitenkinimas")
print("=" * 60)

print("  Analizuojame: salary_usd_k vs job_satisfaction_1_5")

# Išvalome duomenis - pašaliname trūkstamas reikšmes
corr_df = df[['salary_usd_k', 'job_satisfaction_1_5']].dropna()
x_clean = corr_df['salary_usd_k']
y_clean = corr_df['job_satisfaction_1_5']

# Apskaičiuojame koreliacijas
spearman_r, spearman_p = stats.spearmanr(x_clean, y_clean)
pearson_r, pearson_p = stats.pearsonr(x_clean, y_clean)

print(f"\n  Spirmeno koreliacija: r = {spearman_r:.4f}, p = {spearman_p:.4f}")
print(f"  Pirsono koreliacija:  r = {pearson_r:.4f},  p = {pearson_p:.4f}")
print(f"  Koreliacijai naudota: n = {len(corr_df)}")

# Scatter plot - be jitter, nes pasitenkinimas nėra diskretus
plt.figure(figsize=(10, 6))

plt.scatter(x_clean, y_clean, alpha=0.4, color='steelblue', s=30, label='Darbuotojai')

# Pridedame regresijos liniją
z = np.polyfit(x_clean, y_clean, 1)
p = np.poly1d(z)
plt.plot(sorted(x_clean), p(sorted(x_clean)), 
         color='red', linewidth=2, label=f'Tendencijos linija')

plt.title(f'Atlyginimas vs Darbo pasitenkinimas\n(Spearman r={spearman_r:.3f}, p={spearman_p:.4f})', 
          fontsize=14, fontweight='bold')
plt.xlabel('Atlyginimas (salary_usd_k, tūkst. USD)')
plt.ylabel('Darbo pasitenkinimas (job_satisfaction_1_5)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('../images/salary_vs_satisfaction.png', dpi=150, bbox_inches='tight')
plt.show()

print("  [Scatter plot išsaugotas kaip salary_vs_satisfaction.png]")

# Papildoma analizė - koreliacijos stiprumo interpretacija
abs_r = abs(spearman_r)
if abs_r < 0.3:
    strength = "silpnas"
elif abs_r < 0.6:
    strength = "vidutinis"
else:
    strength = "stiprus"

print(f"\n  Koreliacijos interpretacija:")
print(f"  Ryšio stiprumas: {strength} (|r| = {abs_r:.3f})")
print(f"  Kryptis: {'teigiamas' if spearman_r > 0 else 'neigiamas'} - {'daugiau pinigų → didesnis pasitenkinimas' if spearman_r > 0 else 'daugiau pinigų → mažesnis pasitenkinimas'}")

# =============================================================
# #K.2.7 T-TESTŲ VIZUALIZACIJA SU HISTOGRAMOMIS
# =============================================================
print("\n" + "=" * 60)
print("#K.2.7 T-TESTŲ VIZUALIZACIJA SU HISTOGRAMOMIS")
print("=" * 60)

# Sukuriame figūrą su dviem subplotais histogramoms
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('T-testų grupių pasiskirstymo histogramos', fontsize=14, fontweight='bold')

# =============================================================
# 1 TESTAS: Fear of AI Replacement grupių histograma
# =============================================================

# Parenkame duomenis
high_fear = df[df['fear_of_ai_replacement'] == 'High']['burnout_score'].dropna()
low_fear = df[df['fear_of_ai_replacement'] == 'Low']['burnout_score'].dropna()

# Braižome histogramas
axes[0].hist(high_fear, bins=15, alpha=0.6, density=False, 
             label=f'High Fear (n={len(high_fear)})', 
             color='coral', edgecolor='darkred', linewidth=1)
axes[0].hist(low_fear, bins=15, alpha=0.6, density=False, 
             label=f'Low Fear (n={len(low_fear)})', 
             color='lightblue', edgecolor='darkblue', linewidth=1)

# Pridedame vertikalias linijas vidurkiams
axes[0].axvline(high_fear.mean(), color='darkred', linestyle='--', 
                linewidth=2, label=f'High vidurkis = {high_fear.mean():.1f}')
axes[0].axvline(low_fear.mean(), color='darkblue', linestyle='--', 
                linewidth=2, label=f'Low vidurkis = {low_fear.mean():.1f}')

# Pridedame legendą ir pavadinimus
axes[0].legend(loc='upper right', fontsize=9)
axes[0].set_xlabel('Perdegimo balas (burnout_score)', fontsize=11)
axes[0].set_ylabel('Dažnis', fontsize=11)
axes[0].set_title(f'AI pakeitimo baimė\n(t = {t1:.3f}, p = {p1:.4f})', fontsize=12)

# Pridedame tekstą su reikšmingumu
if p1 < 0.05:
    axes[0].text(0.5, 0.95, f'REIKŠMINGA (p < 0.05)', 
                 transform=axes[0].transAxes, ha='center', 
                 color='green', fontweight='bold', fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
else:
    axes[0].text(0.5, 0.95, f'NEREIKŠMINGA (p = {p1:.3f})', 
                 transform=axes[0].transAxes, ha='center', 
                 color='red', fontweight='bold', fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# =============================================================
# 2 TESTAS: Data Scientist vs Kiti darbuotojai histograma
# =============================================================

# Parenkame duomenis
ds_prod = df[df['job_role'] == 'Data Scientist']['productivity_score'].dropna()
others_prod = df[df['job_role'] != 'Data Scientist']['productivity_score'].dropna()

# Braižome histogramas
axes[1].hist(ds_prod, bins=15, alpha=0.6, density=False, 
             label=f'Data Scientist (n={len(ds_prod)})', 
             color='lightgreen', edgecolor='darkgreen', linewidth=1)
axes[1].hist(others_prod, bins=15, alpha=0.6, density=False, 
             label=f'Kiti (n={len(others_prod)})', 
             color='lightgray', edgecolor='black', linewidth=1)

# Pridedame vertikalias linijas vidurkiams
axes[1].axvline(ds_prod.mean(), color='darkgreen', linestyle='--', 
                linewidth=2, label=f'DS vidurkis = {ds_prod.mean():.1f}')
axes[1].axvline(others_prod.mean(), color='black', linestyle='--', 
                linewidth=2, label=f'Kitų vidurkis = {others_prod.mean():.1f}')
axes[1].axvline(visu_vidurkis, color='purple', linestyle='-', 
                linewidth=2, label=f'Visų vidurkis = {visu_vidurkis:.1f}')

# Pridedame legendą ir pavadinimus
axes[1].legend(loc='upper right', fontsize=9)
axes[1].set_xlabel('Produktyvumas (productivity_score)', fontsize=11)
axes[1].set_ylabel('Dažnis', fontsize=11)
axes[1].set_title(f'Data Scientist vs Kiti darbuotojai\n(t = {t2:.3f}, p = {p2:.4f})', fontsize=12)

# Pridedame tekstą su reikšmingumu
if p2 < 0.05:
    axes[1].text(0.5, 0.95, f'REIKŠMINGA (p < 0.05)', 
                 transform=axes[1].transAxes, ha='center', 
                 color='green', fontweight='bold', fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
else:
    axes[1].text(0.5, 0.95, f'NEREIKŠMINGA (p = {p2:.3f})', 
                 transform=axes[1].transAxes, ha='center', 
                 color='red', fontweight='bold', fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('../images/ttest_histograms.png', dpi=150, bbox_inches='tight')
plt.show()

print("  [T-testų histogramos išsaugotos kaip ttest_histograms.png]")

# Pridedame papildomą informaciją apie histogramas
print(f"\n  Histogramų interpretacija:")
print(f"  1 testas (Fear): Grupių pasiskirstymai labai persidengia -")
print(f"     tai paaiškina, kodėl skirtumas nėra statistiškai reikšmingas.")
print(f"  2 testas (Data Scientist): Data Scientist grupės pasiskirstymas")
print(f"     yra šiek tiek pasislinkęs į dešinę, palyginti su kitais -")
print(f"     tai atitinka statistiškai reikšmingą skirtumą.")

print("\n" + "=" * 60)
print("ANALIZĖ BAIGTA - visi grafikai išsaugoti kaip .png failai")
print("=" * 60)