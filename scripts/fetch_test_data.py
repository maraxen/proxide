import urllib.request
import shutil
from pathlib import Path

DATA_DIR = Path("tests/data/trajectories")
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://raw.githubusercontent.com/mdtraj/mdtraj/master/tests/data"

FILES = ["frame0.pdb", "frame0.dcd", "frame0.trr", "frame0.xtc"]


def fetch_data():
  for filename in FILES:
    target = DATA_DIR / filename
    if not target.exists():
      print(f"Downloading {filename}...")
      url = f"{BASE_URL}/{filename}"
      # Use a proper user agent to avoid 403s
      req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
      try:
        with urllib.request.urlopen(req) as response, open(target, "wb") as out_file:
          shutil.copyfileobj(response, out_file)
        print(f"Downloaded {target}")
      except urllib.error.HTTPError as e:
        print(f"Failed to download {filename}: {e}")
    else:
      print(f"Skipping {filename}, already exists")


if __name__ == "__main__":
  fetch_data()
