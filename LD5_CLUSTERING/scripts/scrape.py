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

TEMOS = {
    "sportas": [
        "https://www.delfi.lt/sportas/",
    ],
    "verslas": [
        "https://www.delfi.lt/verslas/",
    ],
    "veidai": [
        "https://www.delfi.lt/veidai/",
    ],
    "laisvalaikis": [
        "https://www.delfi.lt/gyvenimas/",
        "https://www.delfi.lt/gyvenimas/brandiems/",
        "https://www.delfi.lt/gyvenimas/sveikata/",
    ],
}

MIN_DOKUMENTU_TEMAI = 40
MIN_TEKSTO_ILGIS = 300

# Delfi straipsnio URL baigiasi ilgu skaičiumi pvz. -120242691
STRAIPSNIO_RE = re.compile(r"-\d{8,}$")


def gauti_nuorodas(url, max_nuorodu=50):
    """Iš kategorijos puslapio surenka straipsnių nuorodas."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"  [!] {url}: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    nuorodos = []
    matyta = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].rstrip("/")

        if not STRAIPSNIO_RE.search(href):
            continue

        if href.startswith("http"):
            full_url = href
        elif href.startswith("/"):
            full_url = "https://www.delfi.lt" + href
        else:
            continue

        if any(x in full_url for x in ["galerija", "video", "testas", "reklama", "prenumerata"]):
            continue

        if full_url not in matyta:
            matyta.add(full_url)
            nuorodos.append(full_url)

        if len(nuorodos) >= max_nuorodu:
            break

    return nuorodos


def gauti_teksta(url):
    """Iš straipsnio puslapio ištraukia pavadinimą ir tekstą."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"    [!] {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    pavadinimas = ""
    el = soup.find("h1")
    if el:
        pavadinimas = el.get_text(strip=True)

    tekstas_dalys = []
    for sel in ["div.article-body", "div[class*='article-body']",
                "div[class*='articleBody']", "div[itemprop='articleBody']",
                "div[class*='text']", "article"]:
        blokas = soup.select_one(sel)
        if blokas:
            for tag in blokas.select("aside, .related, script, style, figure, .ad, .tags"):
                tag.decompose()
            paragrafai = [p.get_text(strip=True) for p in blokas.find_all("p") if len(p.get_text(strip=True)) > 30]
            if len(paragrafai) >= 2:
                tekstas_dalys = paragrafai
                break

    if not tekstas_dalys:
        tekstas_dalys = [p.get_text(strip=True) for p in soup.find_all("p") if len(p.get_text(strip=True)) > 40]

    tekstas = " ".join(tekstas_dalys)
    tekstas = re.sub(r'[„""\u201e\u201c]', '', tekstas)
    tekstas = tekstas.replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-")
    tekstas = re.sub(r"\s+", " ", tekstas).strip()

    if len(tekstas) < MIN_TEKSTO_ILGIS:
        print(f"    Per trumpas ({len(tekstas)} sim.), praleidžiame.")
        return None

    return {"url": url, "pavadinimas": pavadinimas, "tekstas": tekstas}


if __name__ == "__main__":
    CSV_FAILAS = "../duomenys/duomenys120.csv"

    # Kuriame failą iš naujo – ištrinama visa buvusi informacija
    with open(CSV_FAILAS, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["tema", "pavadinimas", "tekstas", "url"])
        writer.writeheader()
    print(f"CSV failas sukurtas iš naujo: {CSV_FAILAS}\n")

    matyta_url = set()

    for tema, url_sarasas in TEMOS.items():
        print(f"\n=== Tema: {tema.upper()} ===")
        surinkta = []

        for kategorijos_url in url_sarasas:
            if len(surinkta) >= MIN_DOKUMENTU_TEMAI:
                break

            print(f"  Kategorija: {kategorijos_url}")
            nuorodos = gauti_nuorodas(kategorijos_url)
            nuorodos = [n for n in nuorodos if n not in matyta_url]
            print(f"  Naujų nuorodų: {len(nuorodos)}")

            for nuoroda in nuorodos:
                if len(surinkta) >= MIN_DOKUMENTU_TEMAI:
                    break

                print(f"    -> {nuoroda[:90]}")
                doc = gauti_teksta(nuoroda)

                if doc:
                    doc["tema"] = tema
                    surinkta.append(doc)
                    matyta_url.add(nuoroda)
                    print(f"       OK ({len(doc['tekstas'])} sim.) \"{doc['pavadinimas'][:50]}\"")

                time.sleep(1)

        # Išsaugome iš karto po kiekvienos temos
        with open(CSV_FAILAS, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["tema", "pavadinimas", "tekstas", "url"])
            writer.writerows(surinkta)

        print(f"  Surinkta: {len(surinkta)}/{MIN_DOKUMENTU_TEMAI} dok. temai '{tema}'")

    # Galutinė statistika
    print(f"\n{'='*50}")
    with open(CSV_FAILAS, encoding="utf-8") as f:
        visi = list(csv.DictReader(f))
    temu_skaic = {}
    for d in visi:
        temu_skaic[d["tema"]] = temu_skaic.get(d["tema"], 0) + 1
    for t, sk in sorted(temu_skaic.items()):
        print(f"  {t:<15} {sk} dok.")
    print(f"  {'IŠ VISO':<15} {len(visi)} dok.")