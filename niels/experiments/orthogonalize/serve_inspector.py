from __future__ import annotations

import argparse
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve an orthogonalize inspector directory.")
    parser.add_argument("--output-dir", required=True, help="Orthogonalize run output dir")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    inspector_root = Path(args.output_dir).resolve() / "eval-orthogonalized" / "inspector"
    if not (inspector_root / "index.html").exists():
        raise SystemExit(f"Inspector assets not found in {inspector_root}")

    os.chdir(inspector_root)
    server = ThreadingHTTPServer((args.host, args.port), SimpleHTTPRequestHandler)
    print(f"http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
