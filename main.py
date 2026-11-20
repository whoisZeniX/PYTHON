import os
import sys
from pathlib import Path

project = Path(__file__).resolve().parent / "voxora_py"
os.chdir(project)
os.execv(sys.executable, [sys.executable, str(project / "main.py"), *sys.argv[1:]])
