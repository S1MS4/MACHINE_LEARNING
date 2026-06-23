import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

plt.style.use('ggplot')
sns.set_palette("husl")


# #K.2.1 T-TESTAI

def run(df):

    print("\n[T-testas Nr.1] Nepriklausomų imčių t-testas:")
    print("  Lyginame burnout_score: High vs Low fear_of_ai_replacement")
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
        print("  Išvada: Atmetame H0 - yra statistiškai reikšmingas skirtumas (p < 0.05)")
    else:
        print("  Išvada: Negalime atmesti H0 - nėra statistiškai reikšmingo skirtumo")


    print("\n[T-testas Nr.2] Vienos imties t-testas:")
    print("  Lyginame Data Scientist produktyvumą su visų darbuotojų vidurkiu")
    print("  H0: Data Scientist vidurkis = visų darbuotojų vidurkis")
    print("  H1: vidurkiai skiriasi")

    visu_vidurkis = df['productivity_score'].mean()
    data_scientists = df[df['job_role'] == 'Data Scientist']['productivity_score'].dropna()

    print(f"\n  Visų darbuotojų vidurkis: {visu_vidurkis:.2f}")
    print(f"  Data Scientist vidurkis (n={len(data_scientists)}): {data_scientists.mean():.2f}")

    t2, p2 = stats.ttest_1samp(data_scientists, visu_vidurkis)
    print(f"\n  t-statistika: {t2:.4f}")
    print(f"  p-reikšmė:    {p2:.4f}")
    if p2 < 0.05:
        print("  Išvada: Atmetame H0 - yra statistiškai reikšmingas skirtumas (p < 0.05)")
        print(f"  Data Scientist ({data_scientists.mean():.2f}) yra statistiškai "
              f"{'produktyvesni' if data_scientists.mean() > visu_vidurkis else 'mažiau produktyvūs'} "
              f"nei tipinis darbuotojas ({visu_vidurkis:.2f}).")
    else:
        print("  Išvada: Negalime atmesti H0 - nėra statistiškai reikšmingo skirtumo")


    # #K.2.2 KORELIACIJA: Valandos su AI vs Perdegimo balas

    print("\n[Koreliacija Nr.1] hours_with_ai_assistance_daily vs burnout_score")

    xy_df = df[['hours_with_ai_assistance_daily', 'burnout_score']].dropna()
    x1, y1 = xy_df['hours_with_ai_assistance_daily'], xy_df['burnout_score']

    sp_r1, sp_p1 = stats.spearmanr(x1, y1)
    pe_r1, pe_p1 = stats.pearsonr(x1, y1)

    print(f"  Spirmeno koreliacija: r = {sp_r1:.4f}, p = {sp_p1:.4f}")
    print(f"  Pirsono koreliacija:  r = {pe_r1:.4f}, p = {pe_p1:.4f}")

    plt.figure(figsize=(8, 6))
    sns.regplot(data=xy_df, x='hours_with_ai_assistance_daily', y='burnout_score',
                scatter_kws={'alpha': 0.6, 'color': 'steelblue'},
                line_kws={'color': 'red', 'linewidth': 2})
    plt.title(f"Valandos su AI vs Perdegimo balas\n(Spearman r={sp_r1:.3f}, p={sp_p1:.3f})")
    plt.xlabel('Valandos su AI pagalba per dieną')
    plt.ylabel('Perdegimo balas')
    plt.tight_layout()
    plt.savefig('../images/koreliacijos/1_hours_with_ai_assistance_daily.png', dpi=150, bbox_inches='tight')
    


    # #K.2.3 KORELIACIJA: AI užduočių pakeitimas vs Perdegimo balas

    print("\n[Koreliacija Nr.2] ai_replaces_my_tasks_pct vs burnout_score")

    xy_df2 = df[['ai_replaces_my_tasks_pct', 'burnout_score', 'fear_of_ai_replacement']].dropna()
    x2, y2 = xy_df2['ai_replaces_my_tasks_pct'], xy_df2['burnout_score']

    sp_r2, sp_p2 = stats.spearmanr(x2, y2)
    pe_r2, pe_p2 = stats.pearsonr(x2, y2)

    print(f"  Spirmeno koreliacija: r = {sp_r2:.4f}, p = {sp_p2:.4f}")
    print(f"  Pirsono koreliacija:  r = {pe_r2:.4f}, p = {pe_p2:.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'Low': 'steelblue', 'Medium': 'orange', 'High': 'salmon'}
    for fear_level, color in colors.items():
        subset = xy_df2[xy_df2['fear_of_ai_replacement'] == fear_level]
        ax.scatter(subset['ai_replaces_my_tasks_pct'], subset['burnout_score'],
                   alpha=0.5, color=color, s=30,
                   label=f'{fear_level} Fear (n={len(subset)})')

    z = np.polyfit(x2, y2, 1)
    p_line = np.poly1d(z)
    x_sorted = np.sort(x2)
    ax.plot(x_sorted, p_line(x_sorted), 'k--', linewidth=2, label='Tendencijos linija')

    significance = "REIKŠMINGA (p < 0.05)" if sp_p2 < 0.05 else f"NEREIKŠMINGA (p = {sp_p2:.3f})"
    color_sig = 'green' if sp_p2 < 0.05 else 'red'
    ax.text(0.65, 0.95, significance, transform=ax.transAxes,
            color=color_sig, fontweight='bold', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.text(0.02, 0.05,
            f"Spearman r = {sp_r2:.3f}\nPearson r = {pe_r2:.3f}",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_title(f'AI Užduočių Pakeitimas vs Perdegimo Balas\n(Spearman r={sp_r2:.3f}, p={sp_p2:.4f})',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('AI pakeičia mano užduotis (%)')
    ax.set_ylabel('Perdegimo balas (burnout_score)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../images/koreliacijos/2_ai_replaces_vs_burnout.png', dpi=150, bbox_inches='tight')
    


    # #K.2.4 KORELIACIJA: Patirtis vs Darbo pasitenkinimas

    print("\n[Koreliacija Nr.3] years_experience vs job_satisfaction_1_5")

    corr_df = df[['years_experience', 'job_satisfaction_1_5']].dropna()
    x3, y3 = corr_df['years_experience'], corr_df['job_satisfaction_1_5']

    sp_r3, sp_p3 = stats.spearmanr(x3, y3)
    pe_r3, pe_p3 = stats.pearsonr(x3, y3)

    print(f"  Spirmeno koreliacija: r = {sp_r3:.4f}, p = {sp_p3:.4f}")
    print(f"  Pirsono koreliacija:  r = {pe_r3:.4f}, p = {pe_p3:.4f}")

    np.random.seed(42)
    x_jitter = x3 + np.random.normal(0, 0.3, size=len(x3))
    y_jitter = y3 + np.random.normal(0, 0.05, size=len(y3))

    plt.figure(figsize=(10, 6))
    plt.scatter(x_jitter, y_jitter, alpha=0.4, color='steelblue', s=30, label='Darbuotojai')
    z = np.polyfit(x3, y3, 1)
    plt.plot(sorted(x3), np.poly1d(z)(sorted(x3)), color='red', linewidth=2, label='Tendencijos linija')
    plt.title(f'Darbo pasitenkinimas vs Patirtis\n(Spearman r={sp_r3:.3f}, p={sp_p3:.4f})',
              fontsize=14, fontweight='bold')
    plt.xlabel('Patirtis metais (years_experience)')
    plt.ylabel('Darbo pasitenkinimas (job_satisfaction_1_5)')
    plt.yticks([1, 2, 3, 4, 5])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('../images/koreliacijos/3_satisfaction_vs_experience.png', dpi=150, bbox_inches='tight')
    


    # #K.2.5 KORELIACIJA: Atlyginimas vs Darbo pasitenkinimas

    print("\n[Koreliacija Nr.4] salary_usd_k vs job_satisfaction_1_5")

    corr_df2 = df[['salary_usd_k', 'job_satisfaction_1_5']].dropna()
    x4, y4 = corr_df2['salary_usd_k'], corr_df2['job_satisfaction_1_5']

    sp_r4, sp_p4 = stats.spearmanr(x4, y4)
    pe_r4, pe_p4 = stats.pearsonr(x4, y4)

    print(f"  Spirmeno koreliacija: r = {sp_r4:.4f}, p = {sp_p4:.4f}")
    print(f"  Pirsono koreliacija:  r = {pe_r4:.4f}, p = {pe_p4:.4f}")

    strength = "silpnas" if abs(sp_r4) < 0.3 else "vidutinis" if abs(sp_r4) < 0.6 else "stiprus"
    direction = "teigiamas - daugiau pinigų → didesnis pasitenkinimas" if sp_r4 > 0 else "neigiamas"
    print(f"  Ryšio stiprumas: {strength} (|r| = {abs(sp_r4):.3f}), kryptis: {direction}")

    plt.figure(figsize=(10, 6))
    plt.scatter(x4, y4, alpha=0.4, color='steelblue', s=30, label='Darbuotojai')
    z = np.polyfit(x4, y4, 1)
    plt.plot(sorted(x4), np.poly1d(z)(sorted(x4)), color='red', linewidth=2, label='Tendencijos linija')
    plt.title(f'Atlyginimas vs Darbo pasitenkinimas\n(Spearman r={sp_r4:.3f}, p={sp_p4:.4f})',
              fontsize=14, fontweight='bold')
    plt.xlabel('Atlyginimas (salary_usd_k, tūkst. USD)')
    plt.ylabel('Darbo pasitenkinimas (job_satisfaction_1_5)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('../images/koreliacijos/4_salary_vs_satisfaction.png', dpi=150, bbox_inches='tight')
    


    # #K.2.6 T-TESTŲ VIZUALIZACIJA SU HISTOGRAMOMIS

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('T-testų grupių pasiskirstymo histogramos', fontsize=14, fontweight='bold')

    high_fear = df[df['fear_of_ai_replacement'] == 'High']['burnout_score'].dropna()
    low_fear  = df[df['fear_of_ai_replacement'] == 'Low']['burnout_score'].dropna()

    axes[0].hist(high_fear, bins=15, alpha=0.6, label=f'High Fear (n={len(high_fear)})',
                 color='coral', edgecolor='darkred', linewidth=1)
    axes[0].hist(low_fear, bins=15, alpha=0.6, label=f'Low Fear (n={len(low_fear)})',
                 color='lightblue', edgecolor='darkblue', linewidth=1)
    axes[0].axvline(high_fear.mean(), color='darkred', linestyle='--', linewidth=2,
                    label=f'High vidurkis = {high_fear.mean():.1f}')
    axes[0].axvline(low_fear.mean(), color='darkblue', linestyle='--', linewidth=2,
                    label=f'Low vidurkis = {low_fear.mean():.1f}')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].set_xlabel('Perdegimo balas (burnout_score)')
    axes[0].set_ylabel('Dažnis')
    axes[0].set_title(f'AI pakeitimo baimė\n(t = {t1:.3f}, p = {p1:.4f})')
    sig_text = 'REIKŠMINGA (p < 0.05)' if p1 < 0.05 else f'NEREIKŠMINGA (p = {p1:.3f})'
    axes[0].text(0.5, 0.95, sig_text, transform=axes[0].transAxes, ha='center',
                 color='green' if p1 < 0.05 else 'red', fontweight='bold', fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ds_prod     = df[df['job_role'] == 'Data Scientist']['productivity_score'].dropna()
    others_prod = df[df['job_role'] != 'Data Scientist']['productivity_score'].dropna()

    axes[1].hist(ds_prod, bins=15, alpha=0.6, label=f'Data Scientist (n={len(ds_prod)})',
                 color='lightgreen', edgecolor='darkgreen', linewidth=1)
    axes[1].hist(others_prod, bins=15, alpha=0.6, label=f'Kiti (n={len(others_prod)})',
                 color='lightgray', edgecolor='black', linewidth=1)
    axes[1].axvline(ds_prod.mean(), color='darkgreen', linestyle='--', linewidth=2,
                    label=f'DS vidurkis = {ds_prod.mean():.1f}')
    axes[1].axvline(others_prod.mean(), color='black', linestyle='--', linewidth=2,
                    label=f'Kitų vidurkis = {others_prod.mean():.1f}')
    axes[1].axvline(visu_vidurkis, color='purple', linestyle='-', linewidth=2,
                    label=f'Visų vidurkis = {visu_vidurkis:.1f}')
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].set_xlabel('Produktyvumas (productivity_score)')
    axes[1].set_ylabel('Dažnis')
    axes[1].set_title(f'Data Scientist vs Kiti darbuotojai\n(t = {t2:.3f}, p = {p2:.4f})')
    sig_text2 = 'REIKŠMINGA (p < 0.05)' if p2 < 0.05 else f'NEREIKŠMINGA (p = {p2:.3f})'
    axes[1].text(0.5, 0.95, sig_text2, transform=axes[1].transAxes, ha='center',
                 color='green' if p2 < 0.05 else 'red', fontweight='bold', fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('../images/ttest_histograms.png', dpi=150, bbox_inches='tight')

    # #K.2.7 T-TESTŲ VIZUALIZACIJA SU BOX PLOTAIS

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('T-testų grupių box plotai', fontsize=14, fontweight='bold')

    # T-testas Nr.1: High vs Low baimė
    axes[0].boxplot([high_fear, low_fear], patch_artist=True,
                    labels=[f'High Fear (n={len(high_fear)})', f'Low Fear (n={len(low_fear)})'],
                    boxprops=dict(facecolor='lightcoral', color='darkred'),
                    showmeans=True,
                    medianprops=dict(color='darkred', linewidth=2))
    axes[0].set_title(f'AI pakeitimo baimė\n(t = {t1:.3f}, p = {p1:.4f})')
    axes[0].set_ylabel('Perdegimo balas (burnout_score)')
    sig_text = 'REIKŠMINGA (p < 0.05)' if p1 < 0.05 else f'NEREIKŠMINGA (p = {p1:.3f})'
    axes[0].text(0.5, 0.95, sig_text, transform=axes[0].transAxes, ha='center',
                color='green' if p1 < 0.05 else 'red', fontweight='bold', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # T-testas Nr.2: Data Scientist vs Kiti
    axes[1].boxplot([ds_prod, others_prod], patch_artist=True,
                labels=[f'Data Scientist (n={len(ds_prod)})', f'Kiti (n={len(others_prod)})'],
                boxprops=dict(facecolor='lightgreen', color='darkgreen'),
                medianprops=dict(color='darkgreen', linewidth=2),
                showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='red', markersize=8))
    axes[1].set_title(f'Data Scientist vs Kiti\n(t = {t2:.3f}, p = {p2:.4f})')
    axes[1].set_ylabel('Produktyvumas (productivity_score)')
    sig_text2 = 'REIKŠMINGA (p < 0.05)' if p2 < 0.05 else f'NEREIKŠMINGA (p = {p2:.3f})'
    axes[1].text(0.5, 0.95, sig_text2, transform=axes[1].transAxes, ha='center',
                color='green' if p2 < 0.05 else 'red', fontweight='bold', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('../images/ttest_boxplots.png', dpi=150, bbox_inches='tight')