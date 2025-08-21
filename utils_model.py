import pathlib
import requests

def ensure_file(local_path:str, url:str):
    p = pathlib.Path(local_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        p.write_bytes(r.content)