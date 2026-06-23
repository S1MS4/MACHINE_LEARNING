import requests
from bs4 import BeautifulSoup
import csv
import time
import re

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

MIN_TEKSTO_ILGIS = 300

def gauti_teksta(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [!] {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    pavadinimas = ""
    el = soup.find("h1")
    if el:
        # separator=" " – tarpas tarp kiekvieno inline elemento
        pavadinimas = el.get_text(separator=" ", strip=True)

    tekstas_dalys = []
    for sel in ["div.article-body", "div[class*='article-body']",
                "div[class*='articleBody']", "div[itemprop='articleBody']",
                "div[class*='text']", "article"]:
        blokas = soup.select_one(sel)
        if blokas:
            for tag in blokas.select("aside, .related, script, style, figure, .ad, .tags"):
                tag.decompose()
            # separator=" " yra esminis pataisymas
            paragrafai = [p.get_text(separator=" ", strip=True) for p in blokas.find_all("p")]
            paragrafai = [p for p in paragrafai if len(p) > 30]
            if len(paragrafai) >= 2:
                tekstas_dalys = paragrafai
                break

    if not tekstas_dalys:
        tekstas_dalys = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 40]

    tekstas = " ".join(tekstas_dalys)
    # Valymas
    tekstas = re.sub(r"[„""\u201e\u201c]", "", tekstas)
    tekstas = tekstas.replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-")
    tekstas = re.sub(r"\s+", " ", tekstas).strip()

    if len(tekstas) < MIN_TEKSTO_ILGIS:
        return None

    return {"pavadinimas": pavadinimas, "tekstas": tekstas}


# Nuskaitome esamus URL ir temas iš CSV
with open("../duomenys/duomenys160.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"Iš viso {len(rows)} dokumentų – re-scrape pradedamas...\n")

pataisyta = 0
for i, row in enumerate(rows):
    print(f"[{i+1}/{len(rows)}] {row['url'][:80]}")
    doc = gauti_teksta(row["url"])
    if doc:
        row["pavadinimas"] = doc["pavadinimas"]
        row["tekstas"] = doc["tekstas"]
        pataisyta += 1
        print(f"  OK: {row['tekstas'][:80]}...")
    else:
        print(f"  [!] Nepavyko – paliekame seną tekstą")
    time.sleep(1)

# Išsaugome
with open("../duomenys/duomenys160.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["tema", "pavadinimas", "tekstas", "url"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\nPataisyta {pataisyta}/{len(rows)} dokumentų.")
print("Pavyzdys:")
print(rows[0]["tekstas"][:200])