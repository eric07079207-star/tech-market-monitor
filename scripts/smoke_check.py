from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    modules = [
        "app",
        "src.ai_summary",
        "src.data",
        "src.health",
        "src.news",
        "src.sentiment",
    ]
    for module_name in modules:
        importlib.import_module(module_name)
        print(f"import_ok {module_name}")

    from streamlit.testing.v1 import AppTest

    app_path = ROOT / "app.py"
    app_test = AppTest.from_file(str(app_path))
    app_test.run(timeout=180)
    if len(app_test.exception):
        raise RuntimeError(f"streamlit_app_exception: {list(app_test.exception)}")

    print(f"streamlit_ok title_count={len(app_test.title)} markdown_count={len(app_test.markdown)}")


if __name__ == "__main__":
    main()
