"""Lanzador CLI para ejecutar la interfaz Streamlit del proyecto."""

import os
from pathlib import Path
import sys


def main() -> None:
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

    from streamlit.web import cli as streamlit_cli

    app_path = Path(__file__).with_name("streamlit_app.py")
    sys.argv = ["streamlit", "run", str(app_path), *sys.argv[1:]]
    raise SystemExit(streamlit_cli.main())


if __name__ == "__main__":
    main()
