"""Build and serve MkDocs together with generated reports in Docker Compose."""

from __future__ import annotations

import subprocess
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE_DIR = ROOT / "site"
REPORTS_DIR = ROOT / "reports"


def main() -> None:
    """Build the site and expose reports without duplicating generated files."""
    subprocess.run(
        [str(ROOT / ".venv/bin/mkdocs"), "build", "--no-strict"],
        cwd=ROOT,
        check=True,
    )
    reports_link = SITE_DIR / "reports"
    if reports_link.is_symlink():
        reports_link.unlink()
    elif reports_link.exists():
        raise RuntimeError(
            f"Expected {reports_link} to be a symlink, but it is a directory or file"
        )
    reports_link.symlink_to(REPORTS_DIR, target_is_directory=True)

    handler = partial(SimpleHTTPRequestHandler, directory=str(SITE_DIR))
    with ThreadingHTTPServer(("0.0.0.0", 8000), handler) as server:
        print("Documentación disponible en http://0.0.0.0:8000", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
