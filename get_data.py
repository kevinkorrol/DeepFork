import os
import re
import shutil
from pathlib import Path
from zipfile import ZipFile
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from src.data_preprocessing import filter_games
import requests
import time

BASE_URL = "https://www.pgnmentor.com/files.html"
TARGET_DIR = Path("data/temp")
NUM_FILES = 5

def resolve_url(link: str) -> str:
    if link.startswith("http"):
        return link
    return f"https://www.pgnmentor.com/{link.lstrip('/')}"

def progress_bar(count, total, fname):
    percent = int((count / total) * 100)
    bar_len = percent // 2
    bar = "#" * bar_len + "-" * (50 - bar_len)
    print(f"\r[{bar}] {percent}% ({count}/{total}) Downloading {fname}", end="")

def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

    print(f"Fetching file list from {BASE_URL}...")
    driver.get(BASE_URL)
    time.sleep(3)

    html = driver.page_source
    driver.quit()

    zip_links = re.findall(r'href="([^"]*players/[^"]+\.zip)"', html)
    zip_links = sorted(set(zip_links))

    if not zip_links:
        print("No player ZIP files found — site format may have changed or blocked.")
        return

    print(f"Found {len(zip_links)} player ZIP files. Downloading first {NUM_FILES}...")

    for count, link in enumerate(zip_links[:NUM_FILES], start=1):
        url = resolve_url(link)
        fname = os.path.basename(link)
        dl_path = TARGET_DIR / fname

        progress_bar(count, NUM_FILES, fname)

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, stream=True)
        with open(dl_path, "wb") as f:
            for chunk in resp.iter_content(1024 * 64):
                if chunk:
                    f.write(chunk)

        with ZipFile(dl_path, "r") as zip_ref:
            zip_ref.extractall(TARGET_DIR)

        dl_path.unlink()

    print("\nRunning Python preprocessing...")
    filter_games()

    print("Cleaning temporary directory...")
    shutil.rmtree(TARGET_DIR)

    print("All player files downloaded, extracted, and processed.")

if __name__ == "__main__":
    main()
