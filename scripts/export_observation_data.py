from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache"
OBS_DIR = ROOT / "data" / "觀察版資料"


@dataclass(frozen=True)
class ExportItem:
    source: str
    folder: str
    name: str
    description: str
    streamlit_used: str = "是"
    auto_updated: str = "是"


EXPORTS = [
    ExportItem("prices.parquet", "01_市場價格", "市場價格", "股票、ETF、年度十大與市場壓力標的的日線價格與量能。"),
    ExportItem("macro.parquet", "02_總經與市場壓力", "總經市場壓力", "FRED 總經、利率、信用利差與市場壓力資料。"),
    ExportItem("news.parquet", "03_標的新聞", "標的新聞", "科技股、ETF 與 watchlist 相關新聞。"),
    ExportItem("international_news.parquet", "04_國際新聞", "國際新聞", "國際重大新聞、戰爭、貿易、央行與能源消息。"),
    ExportItem("tsla_keyword_news.parquet", "04_國際新聞", "TSLA專題新聞", "TSLA 專用關鍵字追蹤新聞與分類結果。"),
    ExportItem("discovery_news.parquet", "05_潛力股探索", "探索新聞", "隨機市場主題新聞，用來尋找潛在個股。"),
    ExportItem("discovery_mentions.parquet", "05_潛力股探索", "探索股票提及", "從探索新聞中擷取出的股票代號與新聞來源。"),
    ExportItem("discovery_candidates.parquet", "06_候選股追蹤", "每日候選股", "每日量化後的候選觀察股名單。"),
    ExportItem("discovery_history.parquet", "06_候選股追蹤", "候選股歷史", "每日 Top 15 候選觀察股歷史紀錄。"),
    ExportItem("discovery_performance.parquet", "06_候選股追蹤", "候選股表現驗證", "候選股入榜後的後續表現驗證資料。"),
    ExportItem("prediction_log.csv", "07_預測驗證", "市場預測紀錄", "市場判斷與後續 5D/20D/60D 驗證紀錄。"),
    ExportItem("ai_summary.json", "08_AI摘要", "AI市場摘要", "每日 AI 或規則摘要結果。"),
    ExportItem("ai_summary_history.parquet", "08_AI摘要", "AI摘要歷史", "每日 AI 或規則摘要歷史，用於回看與後續準確率驗證。"),
    ExportItem("kg/fact_events.parquet", "09_金融知識圖譜", "事實層", "客觀事件與來源資料。"),
    ExportItem("kg/narrative_features.parquet", "09_金融知識圖譜", "敘事層", "量化敘事與情緒特徵。"),
    ExportItem("kg/market_reactions.parquet", "09_金融知識圖譜", "反應層", "事件後市場反應與驗證結果。"),
    ExportItem("kg/event_links.parquet", "09_金融知識圖譜", "事件連結", "事件與標的關聯關係。"),
    ExportItem("kg/kg_metadata.json", "09_金融知識圖譜", "知識圖譜資訊", "金融知識圖譜更新時間與筆數。"),
    ExportItem("governance_summary.parquet", "11_資料治理", "資料治理摘要", "新聞資料流 official、pending 與 rejected 分層統計。"),
    ExportItem("lstm/lstm_features.parquet", "10_LSTM模型", "LSTM特徵表", "LSTM 訓練與回測使用的序列特徵表。"),
    ExportItem("lstm/lstm_split.parquet", "10_LSTM模型", "LSTM切分表", "LSTM 訓練/驗證/測試切分。"),
    ExportItem("lstm/lstm_predictions.parquet", "10_LSTM模型", "LSTM最新預測", "LSTM 每次更新產生的最新預測結果。"),
    ExportItem("lstm/lstm_backtest.parquet", "10_LSTM模型", "LSTM回測", "LSTM 歷史回測與驗證紀錄。"),
    ExportItem("lstm/lstm_model.pt", "10_LSTM模型", "LSTM模型權重", "LSTM 訓練後輸出的模型權重。"),
    ExportItem("lstm/lstm_model.json", "10_LSTM模型", "LSTM模型資訊", "LSTM 模型版本、訓練指標與 metadata。"),
    ExportItem("lstm/lstm_scaler.json", "10_LSTM模型", "LSTM標準化器", "LSTM 特徵標準化參數。"),
    ExportItem("lstm/lstm_status.json", "10_LSTM模型", "LSTM流程狀態", "LSTM 更新狀態、最後訓練時間與工件統計。"),
    ExportItem("metadata.json", "00_資料說明", "快取更新資訊", "目前快取更新時間、資料起始日與資料筆數。"),
]


def main() -> None:
    OBS_DIR.mkdir(parents=True, exist_ok=True)
    for folder in sorted({item.folder for item in EXPORTS}):
        (OBS_DIR / folder).mkdir(parents=True, exist_ok=True)

    cache_updated_at = _cache_updated_at()
    manifest_rows = []
    for item in EXPORTS:
        source_path = CACHE_DIR / item.source
        target_dir = OBS_DIR / item.folder
        exists = source_path.exists()
        row_count = ""
        latest_date = ""
        fetched_at_utc = ""
        file_size = ""
        exported_files: list[str] = []
        if exists:
            file_size = _file_size(source_path)
            if source_path.suffix == ".parquet":
                df = pd.read_parquet(source_path)
                row_count = len(df)
                latest_date = _latest_date(df)
                fetched_at_utc = _latest_datetime(df, "fetched_at_utc")
                parquet_target = target_dir / item.source
                parquet_target.parent.mkdir(parents=True, exist_ok=True)
                csv_target = target_dir / f"{source_path.stem}.csv"
                csv_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, parquet_target)
                df.to_csv(csv_target, index=False)
                exported_files = [str(parquet_target.relative_to(ROOT)), str(csv_target.relative_to(ROOT))]
            elif source_path.suffix == ".csv":
                df = pd.read_csv(source_path)
                row_count = len(df)
                latest_date = _latest_date(df)
                fetched_at_utc = _latest_datetime(df, "fetched_at_utc")
                csv_target = target_dir / item.source
                csv_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, csv_target)
                exported_files = [str(csv_target.relative_to(ROOT))]
            elif source_path.suffix == ".json":
                json_target = target_dir / item.source
                json_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_path, json_target)
                exported_files = [str(json_target.relative_to(ROOT))]
                try:
                    payload = json.loads(source_path.read_text(encoding="utf-8"))
                    row_count = 1
                    latest_date = payload.get("updated_at_utc") or payload.get("generated_at_utc") or ""
                    fetched_at_utc = payload.get("updated_at_utc") or payload.get("generated_at_utc") or ""
                except json.JSONDecodeError:
                    row_count = ""
        if exists and not fetched_at_utc:
            fetched_at_utc = cache_updated_at

        manifest_rows.append(
            {
                "分類": item.folder,
                "資料名稱": item.name,
                "來源快取": f"data/cache/{item.source}",
                "說明": item.description,
                "已匯出": "是" if exists else "否",
                "自動更新": item.auto_updated,
                "Streamlit使用": item.streamlit_used,
                "筆數": row_count,
                "最新日期或時間": latest_date,
                "最近抓取 UTC": fetched_at_utc,
                "來源檔案大小": file_size,
                "觀察版檔案": "；".join(exported_files),
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OBS_DIR / "00_資料說明" / "資料清單.csv", index=False)
    _write_readme(manifest)
    print(f"exported observation data to {OBS_DIR}")


def _latest_date(df: pd.DataFrame) -> str:
    for column in ["date", "published", "prediction_date", "generated_at_utc", "updated_at_utc"]:
        if column in df.columns:
            value = pd.to_datetime(df[column], errors="coerce", utc=True).max()
            if pd.notna(value):
                return value.isoformat()
    return ""


def _latest_datetime(df: pd.DataFrame, column: str) -> str:
    if column not in df.columns:
        return ""
    value = pd.to_datetime(df[column], errors="coerce", utc=True).max()
    if pd.isna(value):
        return ""
    return value.isoformat()


def _cache_updated_at() -> str:
    path = CACHE_DIR / "metadata.json"
    if not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""
    return payload.get("updated_at_utc", "")


def _file_size(path: Path) -> str:
    size = path.stat().st_size
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / 1024 / 1024:.1f} MB"


def _write_readme(manifest: pd.DataFrame) -> None:
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        "# 觀察版資料",
        "",
        f"產生時間 UTC：{generated}",
        "",
        "這個資料夾是給人工檢查、研究與未來資料倉儲使用的觀察版匯出。",
        "",
        "- `data/cache/` 是系統與 Streamlit 使用的正式快取，請保留不動。",
        "- `data/觀察版資料/` 是由快取複製與轉出的檢查版資料。",
        "- `.csv` 方便用 Excel、Numbers 或一般文字工具查看。",
        "- `.parquet` 保留原本高效率格式，方便未來做回測與資料倉儲。",
        "- `資料清單.csv` 會列出來源檔案、筆數、最新資料日期、最近抓取時間與檔案大小。",
        "",
        "## 資料分類",
        "",
    ]
    for folder in manifest["分類"].drop_duplicates():
        subset = manifest[manifest["分類"] == folder]
        lines.append(f"### {folder}")
        for row in subset.itertuples(index=False):
            lines.append(f"- {row.資料名稱}：{row.說明}")
        lines.append("")
    lines.extend(
        [
            "## 使用方式",
            "",
            "每次要重新整理觀察版資料時，執行：",
            "",
            "```bash",
            "python scripts/export_observation_data.py",
            "```",
            "",
            "資料清單請看：`00_資料說明/資料清單.csv`",
            "",
        ]
    )
    (OBS_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
