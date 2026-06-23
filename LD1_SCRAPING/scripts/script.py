from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import csv
import random
import re

# chrome nustatymai kad nesprogtų
options = webdriver.ChromeOptions()
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

# paleidžiam naršyklę
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# čia saugosim visus duomenis
all_data = []

# pagrindinis URL su filtrais
base_url = "https://autoplius.lt/skelbimai/naudoti-automobiliai?make_id=48"

page = 1


def find_by_pattern(texts, pattern):
    for t in texts:
        if re.search(pattern, t):
            return t
    return ""


while True:
    driver.get(base_url + "&page_nr=" + str(page))
    time.sleep(random.uniform(1.5, 2.3))

    listings = driver.find_elements(By.CSS_SELECTOR, "a.announcement-item")
    print(f"Page {page}: found {len(listings)} listings")

    for listing in listings:
        try:
            driver.execute_script("arguments[0].scrollIntoView();", listing)

            # pavadinimas
            title = listing.find_element(By.CSS_SELECTOR, "div.announcement-title").text.strip()

            # nuoroda į skelbimą
            link = listing.get_attribute("href")

            # metai ir kėbulo tipas
            title_params = listing.find_elements(By.CSS_SELECTOR, "div.announcement-title-parameters span")
            year = title_params[0].text.strip() if len(title_params) > 0 else ""
            body = title_params[1].text.strip() if len(title_params) > 1 else ""

            # kaina
            try:
                price = listing.find_element(By.CSS_SELECTOR, "div.announcement-pricing-info strong").text.strip()
            except:
                price = ""

            # likę parametrai - pagal turinį, ne poziciją
            detail_texts = [d.text.strip() for d in listing.find_elements(By.CSS_SELECTOR, "div.announcement-parameters-block span")]
            fuel         = find_by_pattern(detail_texts, r'Benzinas|Dyzelinas|Elektra|Dujos|Hibridas')
            transmission = find_by_pattern(detail_texts, r'Automatinė|Mechaninė|Pusiau')
            engine       = find_by_pattern(detail_texts, r'\d+[.,]\d+\s*l\.|\d+\s*kWh|\d+\s*kW')
            mileage      = find_by_pattern(detail_texts, r'\d[\d\s]+km')

            # miestas - paskutinis span be skaičių
            used = {fuel, transmission, engine, mileage}
            location_candidates = [t for t in detail_texts if t not in used and t and not re.search(r'\d', t)]
            location = location_candidates[-1] if location_candidates else ""

            all_data.append([title, year, body, price, fuel, transmission, engine, mileage, location, link])

        except Exception as e:
            print("Error:", e)
            continue

    # tikrinam ar yra kitas puslapis
    next_buttons = driver.find_elements(By.CSS_SELECTOR, "a.next")
    if not next_buttons:
        print("Paskutinis puslapis!")
        break

    page += 1
    time.sleep(random.uniform(1.9, 2.2))

print("Total listings:", len(all_data))
raw_title = all_data[-1][0] if all_data else "cars"
safe_title = re.sub(r'[\\/*?:"<>|]', "_", raw_title)

if "model_id" not in base_url:
    safe_title = safe_title.split()[0]

# išsaugom į CSV
with open(rf"C:\Users\ariju\Documents\code_proj\MACHINE_LEARNING\LD1_SCRAPING\data\{safe_title}.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Pavadinimas", "Metai", "Kėbulo tipas", "Kaina", "Kuras", "Pavarų dėžė", "Variklis", "Rida", "Miestas", "Nuoroda"])
    writer.writerows(all_data)

print(f"Saved to {safe_title}.csv!")
driver.quit()