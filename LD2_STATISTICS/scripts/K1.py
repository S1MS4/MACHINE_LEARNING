import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('ggplot')
sns.set_palette("husl")


# #K.1.1 DUOMENŲ MANIPULIAVIMAS IR FILTRAVIMAS

def run(df):

    high_burnout = df[df['burnout_score'] > 70]
    print(f"\n[Filtras 1] Darbuotojai su aukštu perdegimu (burnout_score > 70): {len(high_burnout)}")
    print(high_burnout[['employee_id', 'job_role', 'burnout_score', 'country']].to_string(index=False))

    selected_countries = df[df['country'].isin(['USA', 'India'])]
    print(f"\n[Filtras 2] Darbuotojai iš USA ir Indijos: {len(selected_countries)}")
    print(selected_countries[['employee_id', 'job_role', 'country', 'burnout_score']].to_string(index=False))

    ai_tool_users = df[df['primary_ai_tool'] == 'ChatGPT']
    print(f"\n[Filtras 3] Darbuotojai kurie naudoja ChatGPT: {len(ai_tool_users)}")
    print(ai_tool_users[['employee_id', 'job_role', 'primary_ai_tool', 'burnout_score']].to_string(index=False))

    burnout_by_role = df.groupby('job_role')['burnout_score'].mean().sort_values(ascending=False)
    print(f"\n[Grupavimas] Vidutinis burnout_score pagal profesiją:")
    print(burnout_by_role.round(2).to_string())

    print(f"\n[Top 5] Profesijos su didžiausiu vidutiniu perdegimu:")
    for i, (role, score) in enumerate(burnout_by_role.head(5).items(), 1):
        print(f"  {i}. {role}: {score:.2f}")


    # #K.1.2 IŠSAMI APRAŠOMOJI STATISTIKA

    skaitiniai = ['burnout_score', 'productivity_score', 'salary_usd_k',
                  'hours_with_ai_assistance_daily', 'ai_replaces_my_tasks_pct']

    for col in skaitiniai:
        print(f"\n--- {col} ---")
        d = df[col].dropna()
        moda_val = d.mode()
        moda_str = ', '.join([f"{v:.2f}" for v in moda_val.values]) if len(moda_val) <= 3 else f"{moda_val.iloc[0]:.2f} (ir kiti)"
        print(f"  Vidurkis:       {d.mean():.2f}")
        print(f"  Mediana:        {d.median():.2f}")
        print(f"  Moda:           {moda_str}")
        print(f"  Std. nuokrypis: {d.std():.2f}")
        print(f"  Dispersija:     {d.var():.2f}")
        print(f"  Min / Max:      {d.min():.2f} / {d.max():.2f}")
        print(f"  Q1 / Q2 / Q3:   {d.quantile(0.25):.2f} / {d.quantile(0.50):.2f} / {d.quantile(0.75):.2f}")
        print(f"  Asimetrija:     {d.skew():.2f}")
        print(f"  Ekscesas:       {d.kurtosis():.2f}")

    print(f"\n--- Kategoriniai kintamieji ---")
    for col, label in [('job_role', 'Profesijos'), ('country', 'Šalys'),
                       ('remote_work_type', 'Darbo tipas'), ('fear_of_ai_replacement', 'AI pakeitimo baimė')]:
        print(f"\n  {label} ({col}):")
        print(df[col].value_counts().to_string())


    # #K.1.3 BOX PLOTAI

    burnout_data = df['burnout_score'].dropna()
    salary_data = df['salary_usd_k'].dropna()

    def outlier_count(d):
        q1, q3 = d.quantile(0.25), d.quantile(0.75)
        iqr = q3 - q1
        return len(d[(d < q1 - 1.5*iqr) | (d > q3 + 1.5*iqr)])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Box Plot'ai - Perdegimo balas ir Atlyginimas", fontsize=14, fontweight='bold')

    axes[0].boxplot(burnout_data, patch_artist=True,
                    boxprops=dict(facecolor='lightcoral', color='darkred'),
                    medianprops=dict(color='darkred', linewidth=2))
    axes[0].set_title('Perdegimo balo pasiskirstymas')
    axes[0].set_ylabel('Perdegimo balas (burnout_score)')
    axes[0].set_xticklabels([''])

    axes[1].boxplot(salary_data, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', color='darkblue'),
                    medianprops=dict(color='darkblue', linewidth=2))
    axes[1].set_title('Atlyginimo pasiskirstymas')
    axes[1].set_ylabel('Atlyginimas (salary_usd_k, tūkst. USD)')
    axes[1].set_xticklabels([])

    plt.tight_layout()
    plt.savefig('../images/boxplots.png', dpi=150, bbox_inches='tight')
    

    print(f"\n  Perdegimo balas: mediana = {burnout_data.median():.0f}, "
          f"rasta {outlier_count(burnout_data)} išskirčių.")
    print(f"  Atlyginimas: mediana = {salary_data.median():.0f}k, "
          f"rasta {outlier_count(salary_data)} išskirčių.")


    # #K.1.4 HISTOGRAMOS

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Histogramos - Perdegimo balas ir Atlyginimas', fontsize=14, fontweight='bold')

    sns.histplot(burnout_data, kde=True, ax=axes[0], color='coral', edgecolor='darkred', bins=15)
    axes[0].set_title('Perdegimo balo pasiskirstymas')
    axes[0].set_xlabel('Perdegimo balas (burnout_score)')
    axes[0].set_ylabel('Dažnis')

    sns.histplot(salary_data, kde=True, ax=axes[1], color='steelblue', edgecolor='darkblue', bins=15)
    axes[1].set_title('Atlyginimo pasiskirstymas')
    axes[1].set_xlabel('Atlyginimas (salary_usd_k, tūkst. USD)')
    axes[1].set_ylabel('Dažnis')

    plt.tight_layout()
    plt.savefig('../images/histogramos.png', dpi=150, bbox_inches='tight')
    


    # #K.1.5 REZULTATŲ PAAIŠKINIMAS
    print(f"\nPERDEGIMO BALAS:")
    b_mean, b_med = burnout_data.mean(), burnout_data.median()
    b_skew, b_kurt = burnout_data.skew(), burnout_data.kurtosis()
    print(f"  Vidurkis ({b_mean:.2f}) ir mediana ({b_med:.2f}) yra "
          f"{'artimi' if abs(b_mean - b_med) < 2 else 'skiriasi'}.")
    print(f"  Asimetrija = {b_skew:.2f}, ekscesas = {b_kurt:.2f}.")
    print(f"  Histograma: {'primena normalųjį skirstinį' if abs(b_skew) < 0.5 and abs(b_kurt) < 1 else 'nenormalusis - pastebima asimetrija arba ekscesas'}.")

    print(f"\nATLYGINIMAS:")
    s_mean, s_med = salary_data.mean(), salary_data.median()
    s_skew = salary_data.skew()
    print(f"  Vidurkis ({s_mean:.2f}) {'didesnis' if s_mean > s_med else 'mažesnis'} už medianą ({s_med:.2f}).")
    print(f"  Asimetrija = {s_skew:.2f}: "
          f"{'dešininė - yra darbuotojų su labai aukštais atlyginimais' if s_skew > 0.5 else 'kairinė' if s_skew < -0.5 else 'simetriška'}.")
