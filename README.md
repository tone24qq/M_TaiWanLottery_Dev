# 賓果賓果研究系統

FastAPI／Python 3.12+，僅支援三星、四星、五星。每 45 秒檢查台彩官方資料，先封存下一期預測，開獎後結算。網頁使用繁體中文並支援手機。

目前的真實歷史樣本外比較**尚未證明任何模型優於隨機**；投票分數、相對信心與平均命中率都不是下一期中獎機率。詳見 [模型比較](docs/model-comparison.md) 與 [驗收紀錄](docs/acceptance.md)。

## 啟動

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
# CPU 主機先裝官方 CPU wheel，避免下載 GPU 執行環境
python -m pip install 'torch>=2.2,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

小型單機可以使用 SQLite，不需要另外部署資料庫：

```bash
# Bash；PowerShell 可先設定相同名稱的 $env: 變數
DATABASE_URL=sqlite+aiosqlite:///./bingo.sqlite3 \
DATABASE_URL_SYNC=sqlite:///./bingo.sqlite3 \
python -m uvicorn taiwan_lottery.main:app --host 127.0.0.1 --port 8000
```

也可以把這兩個 URL 寫入本機 `.env`。啟動時自動建立賓果資料表，取得最近三個日曆日（含台北當日）已開獎資料。若不足 100 期，會安全向前擴充，預設最多 31 天；仍不足時顯示原因。資料未就緒時不產生假預測。開啟 http://127.0.0.1:8000；API 文件位於 `/docs`。

PostgreSQL 使用既有環境變數，預設 URL 沿用原專案。新資料庫先建立空資料庫，設定 `.env` 後執行：

```bash
python -m alembic upgrade head
python -m uvicorn taiwan_lottery.main:app --host 127.0.0.1 --port 8000
```

既有 `ac41ae7f9661` 資料庫執行同一遷移即可新增封存表與索引，沒有刪除既有表或資料。SQLite 單機使用啟動初始化，不執行原本 PostgreSQL 專用的初始遷移。正式啟動前請依既有程序備份資料庫與模型目錄。

使用**單一 Uvicorn worker**，同一資料庫的第二個主程序會被鎖阻止。CPU 工作另有訓練／回測程序與一個推論程序；不是在 HTTP 處理程序訓練。此專案提供單機應用，不含登入服務；如需對外開放，請沿用部署端的身分驗證與 HTTPS。

## 已實作

- 官方 JSON API、正常 TLS 驗證、有限重試、完整分頁、20 個不重複號碼檢查、時間與期號驗證、補期及結果衝突拒收。
- 每 45 秒同步；已確認完整的歷史日期不重複下載。最新結果優先結算及封存，不等待較舊日期的重試。
- 事前不可覆寫封存、完整 80 號分數、模型與資料快照版本、雜湊稽核、原子寫入、冪等對獎、重啟恢復。
- Frequency、CDM、River、LightGBM、CatBoost、Transformer、Double DQN、Ensemble 皆實際計算；單一模型故障不阻塞其他模型。
- Equal／Shrink／Borda／Majority 獨立計算；三星、四星、五星各自使用已結算成績調權並封存。
- 少於 30 期維持等權，30～99 期保守收縮；100 期以上才標示可比較。策略變更另需三種星數各至少 100 期、保守配對樣本外比較通過。
- 輕量模型隨新資料更新，River 每期增量學習並保存線上檢查點；耗時模型預設每 36 個新期數或 3 小時排程重訓。
- 真正模型迭代／樣本進度、停止與失敗重試。按「停止訓練」會暫停本次主程序的自動訓練，直到手動開始或重新啟動；模型仍可預測。
- 每個完成訓練的模型立即發布，其他仍有效的舊模型繼續使用；損毀模型檔會標示錯誤並使用真正的統計模型。
- 三種星數累計與最後連續實戰榜；漏一期只影響連續榜。累計榜用 SQL 直方圖，權重使用最近最多 1000 個已結算期數，避免全部預測分數載入記憶體。
- 嚴格 Walk-forward：最近 100／300／1000 期與自訂期號區間，需要時補抓前置資料；歷史模擬獨立儲存，絕不補造正式實戰。
- 全中／部分命中／零命中、平均命中數、全中率、同星數超幾何隨機基準、保守信賴區間與樣本可信程度。
- 首頁四區塊、5 秒不重疊狀態更新；獨立分析頁提供完整三種星數 80 號排名、冷熱／間隔／共現／大小單雙、歷史與盲測報告下載。大型回測報告每次工作只下載一次。

保留原有可重用模型、統計、特徵與離線研究程式；舊 `LSTMModel` 實際已是 Transformer 架構，新執行路徑使用縮小後的真正 Transformer。舊模型研究程式不會寫入新的正式實戰帳本。其他彩種的 API、爬蟲、CRUD、ORM 與介面已移除；歷史遷移保留供既有資料庫升級。

## 主要設定

| 設定 | 預設 | 用途 |
|---|---:|---|
| `SCRAPER_ENABLED` | `true` | 啟動自動同步 |
| `SYNC_INTERVAL_SECONDS` | `45` | 每次檢查後的輪詢間隔 |
| `CPU_THREADS` | `2` | 每個 CPU 工作程序的執行緒上限，另依可用 CPU 限制 |
| `TRAIN_WINDOW` | `1000` | 近期自適應訓練視窗 |
| `HISTORY_MIN_DRAWS` | `100` | 初始訓練最少資料量 |
| `MAX_BACKFILL_DAYS` | `31` | 自動歷史補抓上限 |
| `RETRAIN_MIN_DRAWS` | `36` | 觸發重訓的新期數 |
| `RETRAIN_INTERVAL_SECONDS` | `10800` | 重訓間隔上限 |
| `AUTO_TRAIN` | `true` | 自動訓練 |
| `MODEL_ARTIFACTS_DIR` | `./model_artifacts` | 模型、版本資訊、線上檢查點 |

重訓和推論有各自的受限程序；短暫重疊時總 CPU 使用量可能高於一個程序的上限。較小主機可設定 `CPU_THREADS=1`。實測八模型訓練約需 400～500 MiB 程序記憶體；總程序記憶體及實際速度依主機與同時工作而異。模型檔僅可使用本機程式產生的檔案，勿匯入不可信 pickle。

## 測試與重現

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
# 選用：Linux 上安裝內含 PostgreSQL binaries 的測試工具
python -m pip install pgserver==0.1.4
python -m pytest tests/test_postgres_integration.py -q
# 使用附帶的真實官方資料快照重現完整模型比較
python -m scripts.validate_real_history --count 1000 --threads 2
```

測試中的受控資料僅供測試使用，正式資料入口不會生成資料。`docs/validation/` 提供真實官方快照與 100／300／1000 期逐折報告（gzip JSON），可核對每期訓練截止期、特徵截止期、凍結候選與結算結果。

目前測試紀錄、實際功能界線、官方發布延遲、長時間運行尚未涵蓋的部分，見 [驗收紀錄](docs/acceptance.md)。未變更或重新發佈 Codex 雲端環境設定。
