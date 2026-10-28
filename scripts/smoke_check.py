from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _check_lstm_evaluation_consistency() -> None:
    backtest_path = ROOT / "data/cache/lstm/lstm_backtest.parquet"
    evaluation_path = ROOT / "data/cache/lstm/lstm_evaluation.json"
    if not backtest_path.exists() or not evaluation_path.exists():
        return

    backtest = pd.read_parquet(backtest_path)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8")).get("lstm", {})
    if backtest.empty or evaluation.get("status") != "可用":
        return

    expected_rows = len(backtest)
    expected_accuracy = float((backtest["label"] == backtest["predicted_label"]).mean())
    if int(evaluation.get("rows", -1)) != expected_rows:
        raise RuntimeError(
            f"lstm_evaluation_row_mismatch: evaluation={evaluation.get('rows')} backtest={expected_rows}"
        )
    if abs(float(evaluation.get("accuracy", -1.0)) - expected_accuracy) > 1e-12:
        raise RuntimeError(
            "lstm_evaluation_accuracy_mismatch: "
            f"evaluation={evaluation.get('accuracy')} backtest={expected_accuracy}"
        )
    print(f"lstm_evaluation_ok rows={expected_rows} accuracy={expected_accuracy:.4f}")


def main() -> None:
    modules = [
        "app",
        "src.api_balance",
        "src.ai_summary",
        "src.data",
        "src.health",
        "src.news",
        "src.sentiment",
    ]
    for module_name in modules:
        importlib.import_module(module_name)
        print(f"import_ok {module_name}")

    _check_lstm_evaluation_consistency()

    from streamlit.testing.v1 import AppTest

    app_path = ROOT / "app.py"
    app_test = AppTest.from_file(str(app_path))
    app_test.run(timeout=180)
    if len(app_test.exception):
        raise RuntimeError(f"streamlit_app_exception: {list(app_test.exception)}")

    print(f"streamlit_ok title_count={len(app_test.title)} markdown_count={len(app_test.markdown)}")


if __name__ == "__main__":
    main()
