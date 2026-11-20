import argparse
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
PYTHON = ".venv\\Scripts\\python" if os.name == "nt" else ".venv/bin/python"


def ensure_dependencies() -> None:
    try:
        import fastapi
        import uvicorn
    except ImportError:
        if VENV_PYTHON.exists() and Path(sys.prefix).resolve() != (ROOT / ".venv").resolve():
            os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])
        print(
            "Voxora's packages are not installed for this Python.\n\n"
            "Set up the project once:\n"
            f"  cd {ROOT}\n"
            "  python3 -m venv .venv\n"
            f"  {PYTHON} -m pip install -r requirements.txt\n\n"
            "Then start it with:  python3 main.py",
            file=sys.stderr,
        )
        sys.exit(1)


ensure_dependencies()
sys.path.insert(0, str(ROOT))

from voxora.app import create_app
from voxora.config import get_settings

DOWNLOAD_MODEL = (
    f'{PYTHON} -c "from huggingface_hub import hf_hub_download; import shutil; '
    "shutil.copy(hf_hub_download('Xenova/multilingual-e5-small', 'onnx/model_int8.onnx'), "
    "'onnx/model_int8.onnx')\""
)

app = create_app()


def missing_files() -> list[tuple[Path, str]]:
    settings = get_settings()
    copy_data = "cp -R ../voxora/data ./data   (or rebuild: scripts/build_corpus.py, build_embeddings.py, build_faiss_index.py, build_bm25_index.py)"
    required = [
        (settings.corpus_path, copy_data),
        (settings.faiss_path, f"{PYTHON} scripts/build_faiss_index.py   (after build_embeddings.py)"),
        (settings.bm25_path, f"{PYTHON} scripts/build_bm25_index.py"),
        (settings.tokenizer_path, "mkdir -p onnx && cp ../voxora/onnx/tokenizer.json onnx/"),
        (settings.onnx_model_path, DOWNLOAD_MODEL),
        (settings.css_path, f"{PYTHON} scripts/build_css.py"),
    ]
    return [(path, fix) for path, fix in required if not path.exists()]


def port_is_free(host: str, port: int) -> bool:
    for family, address in ((socket.AF_INET, host), (socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((address, port))
            except OSError as err:
                if family == socket.AF_INET6 and err.errno in (47, 49, 99):
                    continue
                return False
    return True


def running_voxora(port: int) -> bool:
    import json
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as response:
            return json.load(response).get("status") == "ok"
    except Exception:
        return False


def pick_port(host: str, wanted: int) -> int:
    for port in range(wanted, wanted + 20):
        if port_is_free(host, port):
            if port != wanted:
                print(f"Port {wanted} is already in use; using port {port} instead.")
            return port
    print(f"Ports {wanted}-{wanted + 19} are all in use. Stop the other server or set PORT=<free port>.", file=sys.stderr)
    sys.exit(1)


def announce_when_ready(port: int, open_browser: bool) -> None:
    url = f"http://localhost:{port}"

    def wait_and_open():
        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    break
            except OSError:
                time.sleep(0.3)
        else:
            return
        print(f"\n  Voxora is running:  {url}   (also http://127.0.0.1:{port})\n  Press CTRL+C to stop.\n", flush=True)
        if open_browser:
            webbrowser.open(url)

    threading.Thread(target=wait_and_open, daemon=True).start()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Voxora server.")
    parser.add_argument("--reload", action="store_true", help="reload on code changes and watch Tailwind")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser tab on start")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")
    os.chdir(ROOT)

    problems = missing_files()
    if problems:
        print("Voxora cannot start; these files are missing:\n", file=sys.stderr)
        for path, fix in problems:
            print(f"  {path.relative_to(ROOT)}\n    fix: {fix}\n", file=sys.stderr)
        return 1

    import uvicorn

    settings = get_settings()
    wanted = int(os.environ.get("PORT", settings.port))
    if not port_is_free(settings.host, wanted) and running_voxora(wanted):
        url = f"http://localhost:{wanted}"
        print(f"Voxora is already running at {url} - opening it. (Stop the other copy with CTRL+C in its Terminal.)")
        if not args.no_browser:
            webbrowser.open(url + "/app")
        return 0
    port = pick_port(settings.host, wanted)
    announce_when_ready(port, open_browser=not args.no_browser)

    if not args.reload:
        uvicorn.run(app, host=settings.host, port=port)
        return 0

    css_watch = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "build_css.py"), "--watch"])
    try:
        uvicorn.run(
            "main:app",
            host=settings.host,
            port=port,
            reload=True,
            reload_dirs=[str(ROOT / "voxora"), str(ROOT / "templates")],
            reload_includes=["*.py", "*.html"],
        )
    finally:
        css_watch.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
