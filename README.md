# Tech Market Monitor

科技股與科技 ETF 的本機 Streamlit 監控儀表板。

## 監控範圍

- ETF: `QQQ`, `XLK`, `SMH`, `SOXX`, `IGV`, `IYW`, `VGT`
- 個股: `AAPL`, `MSFT`, `NVDA`, `AMD`, `META`, `GOOGL`, `AMZN`, `TSLA`
- 對照: `SPY`, `IWM`, `^VIX`, `TLT`, `HYG`, `DX-Y.NYB`
- Macro: FRED 的 10Y/2Y 利率、殖利率曲線、信用利差、金融條件

## 啟動

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

或直接：

```bash
bash scripts/run_dashboard.sh
```

本機區網模式只能在 Mac 開機、同一個 Wi-Fi、Streamlit 正在跑時使用。若要手機在 Mac 關機或休眠時仍能看，請看 [DEPLOY.md](DEPLOY.md) 部署到雲端。

## 每日收盤後更新

```bash
.venv/bin/python scripts/update_data.py
```

或直接：

```bash
bash scripts/run_update.sh
```

更新結果會寫到 `data/cache/`。儀表板按鈕也可以手動更新市場資料與新聞。

## AI 摘要

如果設定 `OPENAI_API_KEY`，在側邊欄打開「產生 AI 摘要」即可用新聞與量化異常產生繁中摘要。

可選：

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-4.1-mini"
```

沒有 API key 時，系統會使用規則式摘要。

## 注意

這是研究與監控工具，不是投資建議。歷史相似情境呈現的是條件式統計，不代表未來必然重演。
