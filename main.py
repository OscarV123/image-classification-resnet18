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

def start_api(api_ready_evt: threading.Event):
    import app as app_module # importa la app FastAPI desde app.py
    app_module.app.state.ready_evt = api_ready_evt
    uvicorn.run(app_module.app,
                host=API_HOST, 
                port=API_PORT,
                reload=False, 
                log_level="info",
                limit_concurrency=50,
                timeout_keep_alive=5,
                backlog=64)

def wait_until_up(url: str, timeout=10.0, interval=0.3) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=1) as r:
                if 200 <= r.status < 400:
                    return True
        except Exception:
            time.sleep(interval)
    return False

if __name__ == "__main__":
    
    try:
        api_ready = threading.Event()
        threading.Thread(target=start_api, args=(api_ready,), daemon=True).start()
        
        if not api_ready.wait(timeout=30):
            print("La API no indicó 'lista' a tiempo.")
            raise SystemExit(1)
        
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