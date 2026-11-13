"""Read-only Streamlit view. No optional ML dependencies or training on page load."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st

DIRECTORY = Path(__file__).resolve().parents[1] / "data/cache/prediction_research"
LABELS = {"logistic": "邏輯迴歸", "xgboost": "XGBoost", "lightgbm": "LightGBM", "lstm": "LSTM",
          "buy_hold": "買入持有", "momentum20": "20日動能", "qqq_momentum20": "QQQ動能",
          "random_walk": "隨機漫步／持現", "train_majority": "訓練期多數方向"}


def read_report(name="report.json"):
    try:
        return json.loads((DIRECTORY / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def summary():
    report = read_report()
    st.markdown("#### 嚴格樣本外研究")
    if not report:
        st.info("研究報告尚未完成；正式模型維持原版本。")
        return
    st.warning("本輪沒有證明新模型穩定優於基準，原模型未被替換；研究結果不作為交易指令。")
    c = st.columns(3)
    c[0].metric("比較期限", "1／5／10／20 日")
    c[1].metric("每期限測試", f"{next(iter(report['horizons'].values()))['partition']['test']:,} 筆")
    c[2].metric("正式模型", "保留原版")
    st.caption(f"測試訊號區間 {report['test_start']} 至 {report['test_end']}｜研究執行 {report['generated_at_utc']}。歷史測試期曾被舊實驗使用，不冒稱全新未知樣本。")
    state = read_report("monitor.json")
    if state:
        st.caption(f"雲端影子觀察最後掃描：{state.get('scanned_at_utc', '尚未掃描')}｜事前記錄 {state.get('rows', 0)} 筆｜已到期 {state.get('settled', 0)} 筆。")
        if state.get("waiting"):
            st.info("目前快取已錯過下一次開盤的事前預測時間，等待新的完整收盤資料；不補造已發生的預測。")
    age = pd.Timestamp.now(tz="UTC") - pd.Timestamp(report["generated_at_utc"])
    if age.days > 30:
        st.warning("研究回測超過 30 天未重跑；這是固定歷史實驗，不代表今日模型表現。")


def detail():
    report = read_report()
    st.markdown("#### 樣本外模型比較")
    if not report:
        st.info("尚無研究報告。")
        return
    h = st.selectbox("預測期限（交易日）", list(report["horizons"]), key="research_horizon")
    experiment = report["horizons"][h]
    rows = []
    for name, result in experiment["results"].items():
        strategy = result["portfolio"]
        rows.append({"模型": LABELS.get(name, name), "準確率": result["accuracy"],
                     "平衡準確率": result["balanced_accuracy"], "Precision": result["precision"],
                     "Recall": result["recall"], "F1": result["f1"], "ROC-AUC": result["roc_auc"],
                     "Brier（低較佳）": result["brier"], "ECE（低較佳）": result["ece"],
                     "訊號覆蓋率": result["coverage"], "扣成本報酬": strategy["net_return"],
                     "最大回撤": strategy["max_drawdown"], "Sharpe": strategy["sharpe"]})
    percentages = ["準確率", "平衡準確率", "Precision", "Recall", "F1", "訊號覆蓋率", "扣成本報酬", "最大回撤"]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={k: st.column_config.NumberColumn(format="percent") for k in percentages})
    st.caption("收盤後訊號、次日開盤成交，長倉／現金，不做空。單邊費用 5 bps＋滑價 5 bps；另測試 20 bps。Sharpe 現金利率假設 0。覆蓋率為方向訊號比例，不等於持股比例；持現無波動時 Sharpe 不定義。隨機漫步機率 50% 的分類平手按多方統計，策略仍持現。")
    name = st.selectbox("研究模型明細", list(experiment["cv"]), format_func=lambda n: LABELS.get(n, n), key="research_model")
    r = experiment["results"][name]
    symbols = []
    for row in r["by_symbol"]:
        symbols.append({"股票": row["symbol"], "樣本": row["rows"], "準確率": row["accuracy"],
                        "平衡準確率": row["balanced_accuracy"], "平均後續報酬": row["mean_future_return"],
                        "扣成本報酬": row["portfolio"]["net_return"], "最大回撤": row["portfolio"]["max_drawdown"]})
    st.dataframe(pd.DataFrame(symbols), hide_index=True, width="stretch",
                 column_config={k: st.column_config.NumberColumn(format="percent")
                                for k in ["準確率", "平衡準確率", "平均後續報酬", "扣成本報酬", "最大回撤"]})
    st.markdown("**可靠度與資料隔離**")
    intervals = []
    for benchmark, value in r["paired_comparisons"].items():
        intervals.append({"比較基準": LABELS[benchmark], "準確率差": value["accuracy_difference"],
                          "95%下界": value["accuracy_95ci"]["low"], "95%上界": value["accuracy_95ci"]["high"]})
    st.dataframe(pd.DataFrame(intervals), hide_index=True, width="stretch",
                 column_config={k: st.column_config.NumberColumn(format="percent")
                                for k in ["準確率差", "95%下界", "95%上界"]})
    st.caption("區間涵蓋 0 代表尚無明確改善證據；同日股票一起抽樣，並保留 20 日區塊。這不是已校正多重試驗的上線證明。")
    st.dataframe(pd.DataFrame(r["calibration"]).rename(columns={"lower": "機率區間下界", "count": "樣本", "predicted": "預測上漲機率", "observed": "實際上漲比例"}), hide_index=True, width="stretch")
    a = experiment["ablation"]
    st.caption(f"市場特徵消融（只用開發集）：基本技術特徵 Log Loss {a['base_only_cv_log_loss']:.4f}，加入市場特徵 {a['with_market_cv_log_loss']:.4f}；越低越好。")
    st.dataframe(pd.DataFrame(experiment["cv"][name]["folds"]), hide_index=True, width="stretch")
    st.dataframe(pd.DataFrame(experiment["drift"]).rename(columns={"feature": "特徵", "psi": "PSI漂移", "warning": "需檢視"}), hide_index=True, width="stretch")
    st.caption("PSI 是分布變化提示，不是模型錯誤率；目前仍存在倖存者偏差與歷史還原限制，新聞／基本面未通過時點驗證前不加入訓練。")
    st.download_button("下載完整研究報告", json.dumps(report, ensure_ascii=False, indent=2), "prediction_research.json", "application/json")
