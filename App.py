"""
BaatSeBharat — single-command launcher.

`python app.py` (this file — Windows filesystems resolve `app.py` and
`App.py` to the same file) builds the React frontend if needed, starts the
FastAPI backend that serves both the /api/* endpoints and the built React
app, and opens it in the default browser. No separate frontend/backend
processes to manage.

All the calculations, SQLite queries, prediction models, and Plotly figures
this used to render directly (as a Streamlit script) now live in backend/
and src/ — this file is only responsible for getting the whole app running
with one command.
"""
import os
import sys
import subprocess
import threading
import time
import webbrowser

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_FRONTEND_DIR = os.path.join(_APP_DIR, "frontend")
_FRONTEND_DIST = os.path.join(_FRONTEND_DIR, "dist")
_FRONTEND_INDEX = os.path.join(_FRONTEND_DIST, "index.html")

HOST = os.environ.get("BAATSEBHARAT_HOST", "127.0.0.1")
PORT = int(os.environ.get("BAATSEBHARAT_PORT", "8000"))


def _npm_cmd():
    # npm ships as npm.cmd on Windows; shutil.which resolves the right one.
    import shutil
    return shutil.which("npm") or "npm"


def _ensure_frontend_built():
    if os.path.exists(_FRONTEND_INDEX):
        return
    if not os.path.isdir(_FRONTEND_DIR):
        print(f"ERROR: frontend/ directory not found at {_FRONTEND_DIR}")
        sys.exit(1)

    npm = _npm_cmd()
    node_modules = os.path.join(_FRONTEND_DIR, "node_modules")
    if not os.path.isdir(node_modules):
        print("Installing frontend dependencies (npm install)…")
        subprocess.run([npm, "install"], cwd=_FRONTEND_DIR, check=True)

    print("Building frontend (npm run build)…")
    subprocess.run([npm, "run", "build"], cwd=_FRONTEND_DIR, check=True)


def _open_browser_later(url, delay=1.5):
    def _open():
        time.sleep(delay)
        try:
            webbrowser.open(url)
        except Exception:
            pass
    threading.Thread(target=_open, daemon=True).start()


def main():
    _ensure_frontend_built()

    import uvicorn

    url = f"http://{HOST}:{PORT}"
    print(f"Starting BaatSeBharat at {url}")
    _open_browser_later(url)

    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
