import pandas as pd

df = pd.read_csv("../duomenys/duomenys40.csv", encoding="utf-8")

# groupby().sample() išlaiko visus stulpelius įskaitant 'tema'
df20 = df.groupby("tema").sample(n=5, random_state=42).reset_index(drop=True)

df20.to_csv("../duomenys/duomenys20.csv", index=False, encoding="utf-8")

print(f"Sukurta: duomenys20.csv ({len(df20)} eilučių)")
print("Stulpeliai:", df20.columns.tolist())
print(df20["tema"].value_counts())