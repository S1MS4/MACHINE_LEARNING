import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Read existing data
df = pd.read_excel('../data/duomenys.xlsx')

# Define possible values for categorical columns
lytis_options = ['vyras', 'moteris']
metu_laikai = ['Pavasaris', 'Vasara', 'Ruduo', 'Žiema']
savaites_dienos = ['Pirmadienis', 'Antradienis', 'Trečiadienis', 'Ketvirtadienis', 
                   'Penktadienis', 'Šeštadienis', 'Sekmadienis']
filmu_zanrai = ['Veiksmo', 'Komedija', 'Drama', 'Siaubo', 'Sci-Fi', 'Trileris', 
                'Dokumentika', 'Biografija']
naršyklės = ['Chrome', 'Mozilla', 'Internet Explorer', 'Safari', 'Edge']
darbingumo_laikas = ['Ryte', 'Dieną', 'Vakare', 'Naktį']

# Create new dataframe for additional rows
new_rows = 500 - len(df)

# Generate age distribution (more realistic: 18-70 years, with peak in 20-40 range)
ages = np.random.normal(35, 12, new_rows)
ages = np.clip(ages, 18, 70).astype(int)

# Generate height based on gender and age (realistic distributions)
heights = []
genders = []

for i in range(new_rows):
    gender = np.random.choice(lytis_options, p=[0.55, 0.45])  # Slightly more males
    genders.append(gender)
    
    if gender == 'vyras':
        height = np.random.normal(180, 7)
        height = np.clip(height, 165, 205)
    else:
        height = np.random.normal(167, 6)
        height = np.clip(height, 152, 185)
    heights.append(round(height))

# Generate weight based on height, gender, and age (with realistic BMI range)
weights = []
for i in range(new_rows):
    bmi = np.random.normal(24, 3)  # BMI between 18-30 typically
    bmi = np.clip(bmi, 18, 33)
    height_m = heights[i] / 100
    weight = bmi * (height_m ** 2)
    weights.append(round(weight))

# Generate shoe size (correlated with height)
shoe_sizes = []
for i in range(new_rows):
    if genders[i] == 'vyras':
        shoe_size = 39 + (heights[i] - 165) / 5 + np.random.normal(0, 1)
        shoe_size = np.clip(shoe_size, 39, 47)
    else:
        shoe_size = 36 + (heights[i] - 152) / 5 + np.random.normal(0, 1)
        shoe_size = np.clip(shoe_size, 35, 42)
    shoe_sizes.append(round(shoe_size))

# Generate categorical data with realistic correlations
favorite_seasons = []
favorite_days = []
favorite_genres = []
browsers = []
work_times = []
bike_liking = []
kayak_liking = []
university_satisfaction = []

for i in range(new_rows):
    # Season preferences (slight correlation with age)
    if ages[i] < 30:
        probs = [0.35, 0.40, 0.15, 0.10]  # Younger prefer summer/spring
    elif ages[i] > 50:
        probs = [0.25, 0.30, 0.30, 0.15]  # Older more balanced
    else:
        probs = [0.30, 0.35, 0.25, 0.10]
    favorite_seasons.append(np.random.choice(metu_laikai, p=probs))
    
    # Favorite day (weekend preference)
    weekend_prob = 0.6 if ages[i] < 40 else 0.5
    if random.random() < weekend_prob:
        favorite_days.append(np.random.choice(['Penktadienis', 'Šeštadienis', 'Sekmadienis']))
    else:
        favorite_days.append(np.random.choice(['Pirmadienis', 'Antradienis', 'Trečiadienis', 'Ketvirtadienis']))
    
    # Movie genres (age correlation)
    if ages[i] < 30:
        genre_probs = [0.25, 0.20, 0.10, 0.15, 0.15, 0.10, 0.03, 0.02]
    elif ages[i] > 50:
        genre_probs = [0.10, 0.15, 0.25, 0.05, 0.05, 0.10, 0.15, 0.15]
    else:
        genre_probs = [0.15, 0.20, 0.20, 0.10, 0.10, 0.10, 0.08, 0.07]
    favorite_genres.append(np.random.choice(filmu_zanrai, p=genre_probs))
    
    # Browser (Chrome dominates, some variation)
    browser_probs = [0.75, 0.15, 0.05, 0.03, 0.02]
    browsers.append(np.random.choice(naršyklės, p=browser_probs))
    
    # Most productive time
    if ages[i] < 30:
        time_probs = [0.30, 0.30, 0.25, 0.15]  # More night owls
    else:
        time_probs = [0.50, 0.30, 0.15, 0.05]  # More morning people
    work_times.append(np.random.choice(darbingumo_laikas, p=time_probs))
    
    # Activity preferences (correlated with age and gender)
    # Bike liking (1-5 scale)
    if genders[i] == 'vyras':
        bike_mean = 3.5 + (35 - ages[i]) / 20  # Younger men like biking more
    else:
        bike_mean = 3.2 + (35 - ages[i]) / 25
    
    bike_liking.append(np.clip(round(np.random.normal(bike_mean, 1.2)), 1, 5))
    
    # Kayak liking (correlated with age, slightly with gender)
    if genders[i] == 'vyras':
        kayak_mean = 3.8 - abs(ages[i] - 30) / 25  # Peak at age 30
    else:
        kayak_mean = 3.5 - abs(ages[i] - 28) / 30
    
    kayak_liking.append(np.clip(round(np.random.normal(kayak_mean, 1.3)), 1, 5))
    
    # University satisfaction (mostly positive, slight correlation with age)
    sat_mean = 4.0 + (ages[i] - 35) / 50  # Older slightly more satisfied
    sat_mean = np.clip(sat_mean, 3.2, 4.5)
    university_satisfaction.append(np.clip(round(np.random.normal(sat_mean, 1.0)), 1, 5))

# Create new data dictionary
new_data = {
    'Lytis': genders,
    'Amžius (metai)': ages,
    'Ūgis (cm)': heights,
    'Svoris (kg)': weights,
    'Batų dydis (EU)': shoe_sizes,
    'Mėgstamiausia metų laikas': favorite_seasons,
    'Mėgstamiausia savaitės diena': favorite_days,
    'Mėgstamiausias filmų žanras': favorite_genres,
    'Dažniausiai naudojama interneto naršyklė': browsers,
    'Darbingiausiai jaučiatės': work_times,
    'Ar Jums patinka važinėtis dviračiu?': bike_liking,
    'Ar Jums patinka plaukti baidarėmis?': kayak_liking,
    'Ar Jūs esate patenkinti VILNIUS TECH universitetu?': university_satisfaction
}

# Create new dataframe
new_df = pd.DataFrame(new_data)

# Combine with original data
final_df = pd.concat([df.iloc[:,1:], new_df], ignore_index=True)

# Save to Excel
final_df.to_excel('../data/duomenys_500.xlsx', index=False)