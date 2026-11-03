# 賓果賓果研究系統

FastAPI／Python 3.12+，僅支援三星、四星、五星。每 45 秒檢查台彩官方資料，先封存下一期預測，開獎後結算。網頁使用繁體中文並支援手機。

目前的真實歷史樣本外比較**尚未證明任何模型優於隨機**；投票分數、相對信心與平均命中率都不是下一期中獎機率。詳見 [模型比較](docs/model-comparison.md)、[第一階段驗收](docs/acceptance.md) 與 [第二階段驗收及效能](docs/stage2-acceptance.md)。

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

## 三套並行實戰與智慧策略

每一期都先以同一截止時間，原子封存三套策略的三星、四星、五星及完整 80 號分數，再等待官方結果結算：

| 策略 | 資料及工件範圍 |
|---|---|
| Today Only | 目標台北日曆日已開期數，獨立當日工件；不借用前一天已訓練權重。未滿 10 期使用明示 Beta(2,6) 先驗，零期為均勻分數。 |
| Historical | 跨日已知歷史資料；保留原固定視窗及 Shrink 基準。 |
| Adaptive Hybrid | 融合當日與歷史分數；只根據先前已結算的共同期數收縮調權，少於 30 期維持等權；當日零期時保留歷史訊息。 |

三套的共同實戰排行榜與歷史模擬分開。各星數至少 **300 個共同事前封存期數、涵蓋至少 3 日**，才在每 100 期檢查點評估切換；仍須通過保守配對平均命中／全中率門檻與 100 期冷卻。樣本門檻不是效果保證。未通過時繼續 Historical，不因連敗或相同選號加隨機擾動。

首頁增加可收合「智慧策略監控」：三策略預測／實績、模型視窗與理由、最近 10／30／100／300 期變化、重複率、排名變化、診斷、調整／回退證據、程序 CPU／記憶體及耗時。原四區塊操作保留。

自適應視窗使用 5、10、17、20、30、50、60、100、200、300、500、1000 期及受限鄰近探索；每個模型、星數各自選擇。每輪最多固定基準加兩個候選，30 期選擇後以另外 100 期事前影子紀錄驗證，使用多重比較及重複搜尋校正、最小改善、成本與冷卻限制。訓練視窗與推論特徵視窗分開稽核；縮短特徵視窗不代表重型模型已重新訓練。

回測 API `POST /api/v1/bingo/backtest` 支援 `mode: fixed | adaptive | nested | associations`，預設 API 仍是 `fixed`。`adaptive` 重播真正的逐期視窗控制器；`nested` 使用訓練、30 期選擇、100 期驗證、後續外層及獨立保留區間。候選實際比較視窗、特徵配比、CDM 先驗、訓練頻率及四種投票；沒有通過獨立門檻不會取代正式方案。每期凍結參數和分數後才揭露答案；缺期會拒絕回測，避免把隔期當下一期。

大型完整報告以 gzip 檔案儲存於模型目錄 `reports/`；狀態 API 只傳摘要，介面提供 `GET /api/v1/bingo/backtest/artifact/{run_id}` 下載。`GET /api/v1/bingo/audit/{term}` 可查不可覆寫的截止期、視窗、工件版本與影子紀錄。研究狀態、診斷、待處理結算佇列及策略事件皆持久化，歷史回測無法寫入實戰成績。

## A／B／C 可解釋關聯候選

每期另封存最多九組 A／B／C 三星、四星、五星研究候選，與上述 Today／Historical／Hybrid 是不同維度的研究，不會相互覆寫或自動取代正式策略。

- **A 間隔關聯**：1～20 期的時間延遲條件矩陣；搜尋範圍與視窗受限，沒有可靠關聯時使用可重現的真實平滑探索統計。
- **B 跨期組合**：雙號條件到後續雙／三號群組，間隔 2～5 期；最多 16 組前因／後果、32 條保留規則，不做無限制序列枚舉。
- **C 模型交叉參考**：真正融合當折可用八模型分數與其投票權重、目標當日頻率／動能／間隔／共現，以及已獨立確認且當前觸發的 A／B 預測規則。未通過規則貢獻為零。

規則選擇使用所有候選視窗**共用的時間界線**：前半段發現、中間四分之一選擇驗證、最後四分之一獨立確認。短視窗不能借用其他視窗的確認答案。觸發至少 20 次、有限搜尋 Bonferroni 校正、平滑、區間與實際分段統計均封存。跨期單號基準 25%；同一期指定另一號的條件基準 19/79；同時指定群組採無放回組合機率。

解釋完全由真實計數生成，包含歷史範圍、觸發／成功數、條件頻率、平滑 Lift、區間、規則及模型版本。沒有證據即顯示「目前沒有可靠關聯，僅供探索」；相對分數不標為中獎機率。各星數 ABC 獨立實戰榜提供累計、最近 100／300 期結果，歷史比較不灌入實戰。

首頁／分析頁提供可展開證據與完整排名，`GET /api/v1/bingo/associations` 僅讀取既有封存，不在 HTTP 執行規則搜尋。分析頁可選「A／B／C 關聯比較」，使用背景 `mode=associations` 的嚴格 Walk-forward，與相同期數八模型／四投票比較並下載完整 gzip 紀錄。詳見 [關聯引擎驗收](docs/associations-acceptance.md) 與 [真實歷史比較](docs/associations-comparison.md)。

## 鄰號／跨期轉移研究

分析頁新增獨立鄰號區塊：1～80 全部來源與後果、原號及 ±1／±2／±3／±5、後 1／2／3／5／10／20 期、九種歷史視窗，以及最新／前 5／10／20 期截點。邊界不循環，缺期配對排除。畫面分別列出全部合法配對的歷史頻率，以及不共享開獎資料之推論配對的計數、25% 基準、區間、多重校正和獨立確認。

`GET /api/v1/bingo/neighbors` 使用惰性、單程序單執行緒研究快取；`POST /api/v1/bingo/backtest` 增加 `mode=neighbors`，與八模型、四投票、ABC 及可重現隨機選號比較。鄰號不在正式 `MODEL_NAMES`，研究回測不消耗正式策略的驗證次數／冷卻，也不修改融合權重或回填實戰。

真實 100／300／1000 期回測皆**尚未證明優於隨機**。完整結果、方法、效能及重現指令見 [鄰號引擎驗收](docs/neighbors-acceptance.md)；逐折決策與資料雜湊見 [研究報告](docs/validation/neighbors/manifest.json)。

```bash
python -m scripts.benchmark_neighbors --counts 100 300 1000 --threads 1 --output work/neighbors
python -m scripts.validate_neighbor_ui --report work/neighbors/neighbor-walk-forward-100.json.gz --output work/neighbors-ui
```

重現腳本預設審計並重用專案已封存的歷史八模型／ABC 比較，本次重新逐折計算鄰號與隨機；加上 `--fresh-comparators` 可重新訓練全部比較模型。網頁啟動的鄰號回測一律重新計算比較模型，均不讀取現有正式模型回推歷史。

## 主要設定

| 設定 | 預設 | 用途 |
|---|---:|---|
| `SCRAPER_ENABLED` | `true` | 啟動自動同步 |
| `SYNC_INTERVAL_SECONDS` | `45` | 每次檢查後的輪詢間隔 |
| `CPU_THREADS` | `2` | 每個 CPU 工作程序的執行緒上限，另依可用 CPU 限制 |
| `TRAIN_WINDOW` | `1000` | 正式固定基準／背景批次訓練最多期數 |
| `HISTORY_MIN_DRAWS` | `100` | 初始訓練最少資料量 |
| `MAX_BACKFILL_DAYS` | `31` | 自動歷史補抓上限 |
| `RETRAIN_MIN_DRAWS` | `36` | 觸發重訓的新期數 |
| `RETRAIN_INTERVAL_SECONDS` | `10800` | 重訓間隔上限 |
| `AUTO_TRAIN` | `true` | 自動訓練，無新資料不重複重訓 |
| `ADAPTIVE_ENABLED` | `true` | 啟用獨立影子視窗比較；三套資料範圍策略仍持續封存 |
| `ADAPTIVE_SHADOW_BUDGET` | `3` | 固定視窗加候選視窗總數，允許 2～3 |
| `AUTO_VALIDATE` | `true` | 初次模型就緒及診斷觸發的背景巢狀驗證 |
| `VALIDATION_COUNT` | `100` | 自動驗證外層期數，100～1000 |
| `VALIDATION_COOLDOWN_SECONDS` | `86400` | 自動驗證最短間隔 |
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
# 第二階段：真實資料的巢狀外層及三策略比較（獨立研究檔案，非實戰）
python -m scripts.benchmark_adaptation --counts 100 300 1000 --threads 2
python -m scripts.benchmark_research_strategies --count 1000 --threads 2
# 純效能比較需先將 PR #1 的 eb7ec1c 建立為獨立 worktree
python -m scripts.benchmark_bingo --baseline work/stage1-baseline --output work/performance
```

測試中的受控資料僅供測試使用，正式資料入口不會生成資料。`docs/validation/` 提供真實官方快照與 100／300／1000 期逐折報告（gzip JSON），可核對每期訓練截止期、特徵截止期、凍結候選與結算結果。

目前測試紀錄、實際功能界線、官方發布延遲、長時間運行尚未涵蓋的部分，見 [驗收紀錄](docs/acceptance.md)。未變更或重新發佈 Codex 雲端環境設定。
