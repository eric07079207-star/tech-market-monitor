# 市場資料結構

這個資料夾分成兩種用途：

## cache

`data/cache/` 是系統正式快取，Streamlit 儀表板與 GitHub Actions 都讀寫這裡。請不要手動搬移或改名。

主要檔案：

- `prices.parquet`：股票、ETF、年度十大與市場壓力標的價格資料。
- `macro.parquet`：總經、利率、信用利差與市場壓力資料。
- `news.parquet`：科技股與 ETF watchlist 新聞。
- `international_news.parquet`：國際重大新聞、戰爭、貿易、央行與能源消息。
- `discovery_news.parquet`：潛力股探索用的主題新聞。
- `discovery_mentions.parquet`：從探索新聞抽出的股票代號。
- `discovery_candidates.parquet`：每日候選觀察股。
- `discovery_history.parquet`：每日 Top 15 候選觀察股歷史。
- `discovery_performance.parquet`：候選股後續表現驗證。
- `prediction_log.csv`：市場預測與後續驗證紀錄。
- `ai_summary.json`：每日 AI 或規則摘要。
- `metadata.json`：快取更新時間與資料筆數。

## 觀察版資料

`data/觀察版資料/` 是人工檢查與未來資料倉儲使用的整理版。它由 `scripts/export_observation_data.py` 從 `data/cache/` 匯出。

另外：

- `data/觀察版資料/13_專案記憶/` 是專案長期記憶、討論摘要與決策紀錄區，供聊天壓縮後持續保留重要上下文。

這裡會同時保留：

- `.csv`：方便用 Excel、Numbers 或文字工具查看。
- `.parquet`：保留高效率格式，方便未來回測和資料倉儲。
- `00_資料說明/資料清單.csv`：記錄每個資料集的來源、筆數、最新日期、抓取時間與檔案大小。

GitHub Actions 會在每次資料更新後自動重新產生觀察版資料。
