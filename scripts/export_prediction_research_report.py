"""Render the measured comparison, never hard-code an alleged improvement."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
report = json.loads((ROOT / "data/cache/prediction_research/report.json").read_text())
lines = ["# 樣本外模型比較實測報告", "", f"實驗版本：`{report['run_id']}`；來源最後完整交易日：{report['source_last_date']}。",
         "", "## 結論", "", "本輪沒有足夠證據替換原模型。模型有效性與流程成功是兩件事。",
         "歷史區間已被舊系統研究過，因此此處為回溯樣本外比較，不能冒稱全新未知樣本。",
         "每期限2,008列、251個日期，八檔同日股票及重疊標籤並非獨立樣本。",
         "", "## 原版本（不同目標，不能直接相減）", ""]
for row in report["legacy_baselines"]:
    lines.append(f"- {row['name']}：{row['rows']}筆，準確率 {row['accuracy']:.2%}，持續偏多 {row['always_long_accuracy']:.2%}；{row['start']} 至 {row['end']}。")
lines += ["", "舊版為20日收盤至收盤且不同取樣；新實驗為次日開盤至未來開盤。不能聲稱兩者差就是升級幅度。"]
for h, experiment in report["horizons"].items():
    lines += ["", f"## {h} 交易日", "", f"開發期選定候選：`{experiment['development_selected_model']}`（非看測試表選擇）。",
              "", "|模型|Accuracy|Balanced Accuracy|Precision|Recall|F1|AUC|Brier|淨報酬|最大回撤|Sharpe|",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in experiment["results"].items():
        portfolio = row["portfolio"]
        sharpe = f"{portfolio['sharpe']:.3f}" if portfolio["sharpe"] is not None else "未定義（持現）"
        auc = f"{row['roc_auc']:.3f}" if row["roc_auc"] is not None else "N/A"
        lines.append(f"|{name}|{row['accuracy']:.2%}|{row['balanced_accuracy']:.2%}|{row['precision']:.2%}|{row['recall']:.2%}|{row['f1']:.2%}|{auc}|{row['brier']:.4f}|{portfolio['net_return']:.2%}|{portfolio['max_drawdown']:.2%}|{sharpe}|")
    a = experiment["ablation"]
    lines += ["", f"市場特徵消融：基本特徵 CV Log Loss {a['base_only_cv_log_loss']:.5f}；含市場特徵 {a['with_market_cv_log_loss']:.5f}；越低越好。",
              "", "|候選|對持續偏多準確率差|95%下界|95%上界|訊號覆蓋率|Brier校準前→後|20bps淨報酬|",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name in experiment["cv"]:
        row = experiment["results"][name]
        compare = row["paired_comparisons"]["buy_hold"]
        ci = compare["accuracy_95ci"]
        lines.append(f"|{name}|{compare['accuracy_difference']:.2%}|{ci['low']:.2%}|{ci['high']:.2%}|{row['coverage']:.2%}|{row['raw_brier']:.4f} → {row['brier']:.4f}|{row['cost_stress_20bps']['net_return']:.2%}|")
lines += ["", "## 解讀與保存", "",
          "- 多數區間涵蓋0，沒有足夠穩定超額優勢；零報酬代表持現而非模型成功避開所有風險。",
          "- 20日LSTM分類幾乎一直看多，Balanced Accuracy為50%；其結果不能證明有擇時能力。",
          "- 市場特徵在本輪四個期限的開發集消融均未改善Log Loss，不應盲目擴增因子。",
          "- 機率校準不保證變好，報告保留校準前後分數；未用測試結果反過來選校準方法。",
          "- 成本、滑價各單邊5bps；長倉或現金；現金利息0；無槓桿、無做空，淨值逐日計算。",
          "- 1/5/10/20日期限的最終平倉日期不同，所以各期限Buy & Hold報酬也不同。",
          "- 完整逐股、逐市場環境、可靠度分桶、分割日期、Optuna參數、漂移與雜湊在JSON報告。",
          "- 原模型備份和逐列測試預測保存在 `data/research_runs/`，不進Git；輕量部署包與報告進Git。",
          "", "[技術診斷與修改說明](PREDICTION_AUDIT_20261114.md)",
          "", "[完整JSON報告](../data/cache/prediction_research/report.json)", ""]
target = ROOT / "docs/PREDICTION_RESULTS_20261114.md"
target.write_text("\n".join(lines), encoding="utf-8")
print(target)
