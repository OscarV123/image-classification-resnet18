import uvicorn
import time
import webbrowser
import threading
import os

def open_browser():
    time.sleep(7)
    webbrowser.open("http://127.0.0.1:8000/docs")

if __name__ == "__main__":
    
    threading.Thread(target=open_browser).start()
    uvicorn.run("app:app", 
                host="127.0.0.1", 
                port=int(os.environ.get("PORT", 8000)), 
                reload=False)