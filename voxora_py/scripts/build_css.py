import argparse
import os
import sys
from pathlib import Path

import pytailwindcss

ROOT = Path(__file__).resolve().parent.parent
TAILWIND_VERSION = "v4.3.3"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--watch", action="store_true", help="watch templates/ and static/app.js")
    args = parser.parse_args()

    env = dict(os.environ)
    env.setdefault("TAILWINDCSS_VERSION", TAILWIND_VERSION)
    try:
        import certifi

        env.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass

    cli_args = ["-i", "static/input.css", "-o", "static/app.css"]
    cli_args.append("--watch" if args.watch else "--minify")

    result = pytailwindcss.run(
        cli_args, cwd=ROOT, env=env, live_output=True, auto_install=True,
        version=env["TAILWINDCSS_VERSION"],
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
