# Cloud Deployment

如果希望手機在 Mac 關機、休眠、或 Streamlit 沒有在本機執行時仍能看儀表板，需要把 app 部署到雲端。

## 建議方案：Streamlit Community Cloud

適合第一版，因為它直接支援 Streamlit，會提供一個 `https://...streamlit.app` 網址，手機可直接開。

官方文件：

- https://docs.streamlit.io/deploy/streamlit-community-cloud
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management

## 步驟

1. 把這個專案推到 GitHub repository。
2. 到 https://share.streamlit.io 登入並連接 GitHub。
3. 選擇 repository、branch，以及 entrypoint file：

```text
app.py
```

4. Python 版本可選預設值，或選擇仍受支援的 Python 版本。
5. 如果要用 AI 摘要，在 Advanced settings 的 Secrets 放入：

```toml
OPENAI_API_KEY = "你的 key"
OPENAI_MODEL = "gpt-4.1-mini"
```

6. 部署完成後會得到一個公開或受控分享的網址，例如：

```text
https://your-app-name.streamlit.app
```

## 本機與雲端差異

- 本機：`http://192.168.x.x:8501` 只在同 Wi-Fi、Mac 開機、Streamlit 正在跑時有效。
- 雲端：`https://...streamlit.app` 不依賴 Mac，手機行動網路也可看。

## 隱私提醒

如果 repository 或 Streamlit app 設成公開，任何知道網址的人都可能看到儀表板。這個 app 目前沒有交易帳戶或個資，但如果之後加入 API key、持倉、交易紀錄，要改成私人部署或加登入保護。
