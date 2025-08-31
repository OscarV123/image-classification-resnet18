from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import webbrowser
import threading
import uvicorn
import urllib.request
import time
import os

PORT_STATIC = 5500
DOCS_DIR = "docs"
API_HOST = "127.0.0.1"
API_PORT = int(os.environ.get("PORT", 8000))

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DOCS_DIR, **kwargs)

def start_static():
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT_STATIC), Handler)
    httpd.serve_forever()

def start_api():
    uvicorn.run("app:app", host=API_HOST, port=API_PORT, reload=False, log_level="info")

def wait_until_up(url: str, timeout=10.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if 200 <= r.status < 400:
                    return True
        except Exception:
            time.sleep(0.3)
    return False

if __name__ == "__main__":
    
    try:
        # levanta la API y la página web en paralelo
        threading.Thread(target=start_api, daemon=True).start()
        threading.Thread(target=start_static, daemon=True).start()

        # abre el navegador cuando el estático esté listo
        url = f"http://127.0.0.1:{PORT_STATIC}/index.html"
        if wait_until_up(url, timeout=10):
            webbrowser.open(url)
        else:
            print(f"No se pudo verificar {url} en el tiempo esperado.")
        
        # mantener el proceso principal vivo
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nSaliendo…")