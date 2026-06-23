"""
Statistinė duomenų analizė - ND2
Studentas: [Arijus]
Data: 2026-03-02
"""

import pandas as pd
import K1
import K2

df = pd.read_csv(r'../data/ai_worker_burnout_attrition_2026.csv')

print("=" * 60)
print("Duomenys sėkmingai įkelti!")
print(f"Eilučių skaičius: {len(df)}, Stulpelių skaičius: {len(df.columns)}")
print("=" * 60)


print("\n" + "=" * 60)
print("K1 - APRAŠOMOJI STATISTIKA")
print("=" * 60)
K1.run(df)


print("\n" + "=" * 60)
print("K2 - INFERENTINĖ STATISTIKA")
print("=" * 60)
K2.run(df)


print("\n" + "-" * 60)
print("ANALIZĖ BAIGTA - visi grafikai išsaugoti kaip .png failai")
