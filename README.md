# AFS 模擬交易所

以 Streamlit 執行的虛構金融市場實驗沙盒。行情、代理人與交易都是模擬資料，沒有連接券商，不是投資建議。

## 雲端部署

在 Streamlit Community Cloud 選擇本儲存庫，分支使用 `main`，入口檔案使用 `app.py`，Python 使用 3.13。

這份上線副本可獨立執行模擬行情、下單、新聞與研究模型；不包含課程 MariaDB 或付費 AI 摘要介面，不需要資料庫密碼或 API 金鑰。

每個瀏覽器連線有自己的模擬狀態。重新建立連線或雲端重新啟動時，尚未匯出的資料可能消失；請使用畫面上的匯出功能保存紀錄。

GitHub 儲存庫可保持私人，網站的公開範圍另外在 Streamlit 的 App settings → Sharing 設定。

## 本機啟動

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```
