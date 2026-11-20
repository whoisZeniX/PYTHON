import argparse
import errno
import importlib.util
import json
import logging
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VENV_PYTHON = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

if importlib.util.find_spec("fastapi") is None:
    if VENV_PYTHON.exists() and Path(sys.prefix).resolve() != VENV.resolve():
        os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])
    sys.exit("Packages are missing. Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt")

import uvicorn

sys.path.insert(0, str(ROOT))

from voxora.app import create_app
from voxora.config import get_settings

app = create_app()


def missing_files(settings) -> list[Path]:
    required = [
        settings.corpus_path,
        settings.faiss_path,
        settings.faiss_en_path,
        settings.bm25_path,
        settings.onnx_model_path,
        settings.tokenizer_path,
        settings.css_path,
    ]
    return [path for path in required if not path.exists()]


def voxora_running(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
            return json.load(response).get("status") == "ok"
    except (OSError, ValueError):
        return False


def port_free(port: int) -> bool:
    for family, host in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError as err:
                if err.errno == errno.EADDRINUSE:
                    return False
    return True


def open_when_ready(port: int, open_browser: bool) -> None:
    def wait():
        for _ in range(240):
            if voxora_running(port):
                print(f"\n  Voxora is running at http://localhost:{port}  (CTRL+C to stop)\n", flush=True)
                if open_browser:
                    webbrowser.open(f"http://localhost:{port}/app")
                return
            time.sleep(0.5)

    threading.Thread(target=wait, daemon=True).start()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Voxora server.")
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    parser.add_argument("--reload", action="store_true", help="restart when the code changes")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")
    os.chdir(ROOT)
    settings = get_settings()

    missing = missing_files(settings)
    if missing:
        print("Voxora can't start, these files are missing:")
        for path in missing:
            print(f"  {path.relative_to(ROOT)}")
        print("See the Setup section in README.md.")
        return 1

    port = int(os.environ.get("PORT", settings.port))
    if voxora_running(port):
        print(f"Voxora is already running at http://localhost:{port}")
        if not args.no_browser:
            webbrowser.open(f"http://localhost:{port}/app")
        return 0

    first = port
    while not port_free(port):
        port += 1
    if port != first:
        print(f"Port {first} is busy, using {port} instead.")

    open_when_ready(port, not args.no_browser)
    uvicorn.run("main:app" if args.reload else app, host=settings.host, port=port, reload=args.reload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
