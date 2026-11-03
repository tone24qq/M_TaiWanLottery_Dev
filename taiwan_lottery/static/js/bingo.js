/* Bingo UI: API values are rendered as text; refreshes never overlap. */
(() => {
  "use strict";
  const root = document.querySelector("[data-page]");
  if (!root) return;
  const page = root.dataset.page;
  const $ = (id) => document.getElementById(id);
  const names = {
    frequency: "Frequency 頻率",
    cdm: "CDM 貝氏統計",
    river: "River 線上學習",
    lightgbm: "LightGBM",
    catboost: "CatBoost",
    transformer: "Transformer",
    dqn: "DQN",
    lstm: "LSTM",
    ensemble: "Ensemble 集成",
    equal: "Equal 等權",
    shrink: "Shrink 收縮加權",
    borda: "Borda 排名積分",
    majority: "Majority 多數決",
  };
  const states = {
    ready: "可用",
    trained: "已訓練",
    completed: "已完成",
    complete: "已完成",
    done: "已完成",
    running: "執行中",
    training: "訓練中",
    fitting: "訓練中",
    waiting: "等待中",
    pending: "等待中",
    queued: "已排入佇列",
    idle: "待命",
    stopped: "已停止",
    stopping: "停止中",
    cancelled: "已停止",
    canceled: "已停止",
    failed: "失敗",
    error: "異常",
    disabled: "未啟用",
    unavailable: "無法使用",
    untrained: "尚未訓練",
    insufficient: "樣本不足",
    missing_dependency: "缺少套件",
    connected: "已連線",
    online: "已連線",
    healthy: "正常",
    synced: "同步完成",
    ok: "正常",
    syncing: "同步中",
    degraded: "資料延遲",
    stale: "資料延遲",
    offline: "連線中斷",
    disconnected: "連線中斷",
    delayed: "資料延遲",
    retrying: "重試中",
    recovering: "補資料中",
  };
  const labels = { 3: "三星", 4: "四星", 5: "五星" };
  let status = null,
    analysis = null,
    selectedStar = "3",
    selectedBoard = "cumulative";
  let refreshing = false,
    timer = null,
    reportURL = null,
    reportKey = null;
  let actionPending = false,
    transportError = false,
    suspended = false;
  let lastAnalysisAt = 0,
    lastReportRunId = null,
    backtestState = "waiting";

  function el(tag, text, cls) {
    const node = document.createElement(tag);
    if (text !== undefined && text !== null) node.textContent = String(text);
    if (cls) node.className = cls;
    return node;
  }
  function put(id, text) {
    const node = $(id);
    if (node) node.textContent = text ?? "—";
  }
  function num(value, digits = 2) {
    return value !== null &&
      value !== undefined &&
      Number.isFinite(Number(value))
      ? Number(value).toLocaleString("zh-TW", {
          maximumFractionDigits: digits,
          minimumFractionDigits: digits,
        })
      : "—";
  }
  function percent(value, digits = 1) {
    return value !== null &&
      value !== undefined &&
      Number.isFinite(Number(value))
      ? `${num(Number(value) * 100, digits)}%`
      : "—";
  }
  function integer(value) {
    return value !== null &&
      value !== undefined &&
      Number.isFinite(Number(value))
      ? Number(value).toLocaleString("zh-TW", { maximumFractionDigits: 0 })
      : "—";
  }
  function displayName(value) {
    const key = String(value || "");
    if (key === "walk_forward") return "逐期驗證";
    if (key.startsWith("walk_forward/")) return `逐期驗證／${displayName(key.slice(13))}`;
    const work = key.match(/^(\w+) (\d+\/\d+)$/);
    if (work) return `${displayName(work[1])} ${work[2]}`;
    return (
      names[key.toLowerCase().replace(/^fusion[_:]/, "")] || key || "未命名模型"
    );
  }
  function stateLabel(value) {
    return (
      states[String(value || "").toLowerCase()] ||
      (value ? "狀態待確認" : "尚無狀態")
    );
  }
  function dateText(value) {
    if (!value) return "—";
    let iso = String(value).replace(" ", "T");
    if (!/(Z|[+-]\d{2}:?\d{2})$/i.test(iso)) iso += "+08:00";
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return "時間格式無法辨識";
    return new Intl.DateTimeFormat("zh-TW", {
      timeZone: "Asia/Taipei",
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    }).format(date);
  }
  function badge(node, text, state) {
    if (!node) return;
    node.textContent = text;
    const value = String(state || "").toLowerCase();
    const good = [
      "ready",
      "trained",
      "completed",
      "complete",
      "done",
      "connected",
      "online",
      "healthy",
      "ok",
      "synced",
      "sealed",
    ];
    const bad = ["failed", "error", "offline", "disconnected"];
    const warn = [
      "degraded",
      "stale",
      "delayed",
      "disabled",
      "unavailable",
      "untrained",
      "insufficient",
      "missing_dependency",
      "retrying",
    ];
    node.className = `badge ${good.includes(value) ? "good" : bad.includes(value) ? "bad" : warn.includes(value) ? "warning" : "neutral"}`;
  }
  function message(text, error = false) {
    const node = $("message");
    node.hidden = !text;
    node.textContent = text || "";
    node.className = `notice${error ? " error" : ""}`;
  }
  async function request(path, options = {}) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 60000);
    try {
      const response = await fetch(`/api/v1/bingo/${path}`, {
        ...options,
        signal: controller.signal,
        headers: {
          Accept: "application/json",
          ...(options.body ? { "Content-Type": "application/json" } : {}),
          ...options.headers,
        },
      });
      let data;
      try {
        data = await response.json();
      } catch {
        throw new Error(`伺服器未回傳有效資料（${response.status}）`);
      }
      if (!response.ok) {
        const detail = data.detail;
        throw new Error(
          typeof detail === "string"
            ? detail
            : Array.isArray(detail)
              ? detail.map((row) => row.msg).join("；")
              : `請求失敗（${response.status}）`,
        );
      }
      return data;
    } catch (error) {
      if (error.name === "AbortError")
        throw new Error("請求逾時，系統會繼續自動重試。");
      throw error;
    } finally {
      clearTimeout(timeout);
    }
  }
  function balls(numbers) {
    const fragment = document.createDocumentFragment();
    for (const n of numbers || [])
      fragment.append(el("span", String(n).padStart(2, "0"), "ball"));
    return fragment;
  }
  function emptyRow(body, colspan, text) {
    body.replaceChildren();
    const row = el("tr");
    const cell = el("td", text, "empty");
    cell.colSpan = colspan;
    row.append(cell);
    body.append(row);
  }
  function progress(id, wrapId, completed, total, value) {
    const node = $(id),
      wrap = $(wrapId);
    if (!node) return;
    let actual = null;
    if (
      Number.isFinite(Number(total)) &&
      Number(total) > 0 &&
      Number.isFinite(Number(completed))
    )
      actual = (100 * Number(completed)) / Number(total);
    else if (
      value !== null &&
      value !== undefined &&
      Number.isFinite(Number(value))
    )
      actual = Number(value);
    if (wrap) wrap.hidden = false;
    if (actual === null) node.removeAttribute("value");
    else node.value = Math.max(0, Math.min(100, actual));
    return actual;
  }
  function renderPredictions(prediction) {
    const holder = $("predictions");
    holder.replaceChildren();
    put(
      "prediction-target",
      prediction ? `目標第 ${prediction.target_term} 期` : "尚無封存預測",
    );
    badge(
      $("archive-status"),
      prediction
        ? prediction.settled
          ? "已封存・已結算"
          : "已封存・待開獎"
        : "等待封存",
      prediction ? "sealed" : "waiting",
    );
    put(
      "voting-method",
      prediction
        ? `融合方式：${displayName(prediction.method)}`
        : "等待下一期預測",
    );
    put(
      "archive-time",
      prediction
        ? `封存時間：${dateText(prediction.created_at)}（台北時間）`
        : "封存後同一期號碼保持固定。",
    );
    for (const stars of ["3", "4", "5"]) {
      const data = prediction?.by_star?.[stars],
        card = el("article", null, "prediction-card");
      const heading = el("h3", labels[stars]);
      heading.append(el("small", `Top ${stars}`));
      card.append(heading);
      if (!data?.numbers?.length) {
        card.append(el("p", "等待有效資料與可用模型", "empty"));
        holder.append(card);
        continue;
      }
      const group = el("div", null, "balls");
      group.append(balls(data.numbers));
      card.append(group);
      const details = el("div", null, "candidate-detail");
      data.numbers.forEach((n, index) => {
        const row = el("div", null, "candidate-row");
        const score = Array.isArray(data.scores)
          ? data.scores.length === 80
            ? data.scores[n - 1]
            : data.scores[index]
          : data.scores?.[n];
        const support = Array.isArray(data.support)
          ? data.support.length === 80
            ? data.support[n - 1]
            : data.support[index]
          : typeof data.support === "number"
            ? data.support
            : data.support?.[n];
        row.append(
          el(
            "span",
            `${String(n).padStart(2, "0")} 號 · 分數 ${num(score, 4)}`,
          ),
          el("span", `支持 ${integer(support)} 個模型`),
        );
        details.append(row);
      });
      if (data.confidence !== null && data.confidence !== undefined) {
        const confidence = Array.isArray(data.confidence)
          ? data.confidence.map((v) => num(v, 3)).join("／")
          : typeof data.confidence === "number"
            ? num(data.confidence, 3)
            : String(data.confidence);
        details.append(el("div", `相對信心：${confidence}`));
      }
      card.append(details);
      holder.append(card);
    }
  }
  function renderModels(models, training) {
    const holder = $("models");
    holder.replaceChildren();
    for (const model of models || []) {
      const item = el("div", null, "model-item"),
        top = el("div", null, "model-top"),
        state = el("span");
      badge(state, stateLabel(model.status), model.status);
      top.append(el("span", displayName(model.name), "model-name"), state);
      item.append(top);
      const info = [`有效樣本 ${integer(model.samples)} 期`];
      if (model.version) info.push(`版本 ${model.version}`);
      if (model.updated_at) info.push(`更新 ${dateText(model.updated_at)}`);
      item.append(el("p", info.join(" · "), "model-description"));
      if (
        model.total > 0 ||
        (model.progress !== undefined && model.progress !== null)
      ) {
        const row = el("div", null, "model-progress"),
          bar = el("progress");
        bar.max = 100;
        const value =
          model.total > 0
            ? (100 * Number(model.completed || 0)) / Number(model.total)
            : Number(model.progress);
        if (Number.isFinite(value))
          bar.value = Math.max(0, Math.min(100, value));
        bar.setAttribute("aria-label", `${displayName(model.name)} 訓練進度`);
        row.append(
          bar,
          el(
            "span",
            `${num(value, 0)}%${model.total > 0 ? ` · ${integer(model.completed)}／${integer(model.total)}` : ""}`,
          ),
        );
        item.append(row);
      }
      if (model.error) item.append(el("p", model.error, "error-detail"));
      holder.append(item);
    }
    if (!models?.length) holder.append(el("p", "尚無模型狀態", "empty"));
    const job = training || {};
    badge($("training-state"), stateLabel(job.state), job.state);
    put(
      "training-detail",
      job.error
        ? `訓練異常：${job.error}`
        : job.current_model
          ? `目前模型：${displayName(job.current_model)}`
          : job.state === "idle"
            ? "背景訓練待命；新資料會依排程更新模型。"
            : "背景工作狀態持續更新。",
    );
    if (
      job.total > 0 ||
      (job.progress !== undefined && job.progress !== null)
    ) {
      const value = progress(
        "training-progress",
        "training-progress-wrap",
        job.completed,
        job.total,
        job.progress,
      );
      put(
        "training-work",
        `${value === null ? "工作進行中" : `${num(value, 0)}%`}${job.total > 0 ? ` · 已完成 ${integer(job.completed)}／${integer(job.total)} 工作單位` : ""}`,
      );
    } else $("training-progress-wrap").hidden = true;
    const running = ["running", "training", "queued", "stopping"].includes(
      job.state,
    );
    document.querySelector('[data-action="train"]').disabled =
      running || actionPending;
    document.querySelector('[data-action="train/stop"]').disabled =
      !running || actionPending;
  }
  function metricCells(row, index, expanded = false) {
    const tr = el("tr"),
      title = el("td"),
      titleText = el("div", `${index + 1}. ${displayName(row.name)}`);
    title.append(
      titleText,
      el("div", row.credibility || "樣本待確認", "subtle"),
      el("div", row.evidence || "尚未證明優於隨機", "subtle"),
    );
    if (row.updated_at) title.title = `最近更新：${dateText(row.updated_at)}`;
    const mean = el("td", num(row.mean_hits, 3));
    if (
      row.ci_low !== null &&
      row.ci_low !== undefined &&
      row.ci_high !== null &&
      row.ci_high !== undefined
    )
      mean.append(
        el(
          "div",
          `95% 區間 ${num(row.ci_low, 3)}～${num(row.ci_high, 3)}`,
          "subtle",
        ),
      );
    if (!expanded)
      mean.append(el("div", `號碼命中率 ${percent(row.hit_rate)}`, "subtle"));
    const full = el("td", percent(row.full_hit_rate));
    full.append(el("div", `${integer(row.full_hits)} 次全中`, "subtle"));
    if (row.random_full_hit_rate !== undefined)
      full.append(
        el("div", `隨機 ${percent(row.random_full_hit_rate, 2)}`, "subtle"),
      );
    if (row.full_hit_ci_low !== undefined && row.full_hit_ci_high !== undefined)
      full.title = `全中率 95% 區間：${percent(row.full_hit_ci_low, 2)}～${percent(row.full_hit_ci_high, 2)}`;
    const delta = el(
      "td",
      row.delta_vs_random === null || row.delta_vs_random === undefined
        ? "—"
        : `${Number(row.delta_vs_random) > 0 ? "+" : ""}${num(row.delta_vs_random, 3)}`,
      Number(row.delta_vs_random) > 0
        ? "positive"
        : Number(row.delta_vs_random) < 0
          ? "negative"
          : "",
    );
    delta.append(el("div", `隨機平均 ${num(row.random_mean, 3)}`, "subtle"));
    tr.append(title, el("td", integer(row.samples)), mean, full);
    if (expanded) {
      const partial = row.partial_hits;
      tr.append(
        el(
          "td",
          typeof partial === "object" && partial !== null
            ? Object.entries(partial)
                .map(([hits, count]) => `${hits} 碼：${count}`)
                .join("；")
            : integer(partial),
        ),
      );
      tr.append(el("td", percent(row.hit_rate)));
    } else tr.append(el("td", percent(row.weight)));
    tr.append(delta);
    return tr;
  }
  function renderLeaderboard() {
    const body = $("leaderboard");
    if (!body) return;
    const boards = status?.leaderboards || {};
    const rows =
      boards[selectedStar]?.[selectedBoard] ||
      boards[selectedBoard]?.[selectedStar] ||
      [];
    body.replaceChildren();
    rows.forEach((row, index) => body.append(metricCells(row, index)));
    if (!rows.length)
      emptyRow(body, 6, "尚無已結算實戰樣本；歷史盲測不列入此榜。");
    const strategy = status?.strategy;
    const methods = ["equal", "shrink", "borda", "majority"];
    const bestModel = rows.find(
        (row) => !methods.includes(String(row.name).replace(/^fusion[_:]/, "")),
      ),
      bestFusion = rows.find((row) =>
        methods.includes(String(row.name).replace(/^fusion[_:]/, "")),
      );
    const leaders = [
      bestModel ? `模型首位：${displayName(bestModel.name)}` : null,
      bestFusion ? `融合首位：${displayName(bestFusion.name)}` : null,
    ]
      .filter(Boolean)
      .join(" · ");
    put(
      "strategy",
      (strategy
        ? `目前融合：${displayName(strategy.method)} · ${strategy.status || "尚未完成足夠的樣本外比較"}`
        : "尚未完成足夠的樣本外比較；不宣稱優於隨機。") +
        (leaders ? `。${leaders}` : ""),
    );
  }
  function renderDashboard(data) {
    status = data;
    const latest = data.latest,
      sync = data.sync || {},
      target = data.target;
    put("latest-title", latest ? `第 ${latest.draw_term} 期` : "等待開獎資料");
    put(
      "latest-date",
      latest
        ? `${dateText(latest.draw_datetime)}（台北時間）`
        : "尚未取得可靠開獎資料",
    );
    const holder = $("latest-numbers");
    holder.replaceChildren();
    if (latest?.numbers?.length) holder.append(balls(latest.numbers));
    else holder.append(el("p", "自動同步後顯示二十個開獎號碼", "empty"));
    badge($("source-status"), stateLabel(sync.state), sync.state);
    badge(
      $("page-status"),
      sync.error ? "同步待恢復" : "狀態持續更新",
      sync.error ? "error" : "connected",
    );
    put("last-sync", dateText(sync.last_success));
    put(
      "data-delay",
      sync.delay_seconds !== null && sync.delay_seconds !== undefined
        ? `${num(sync.delay_seconds, 0)} 秒`
        : "—",
    );
    put(
      "target-term",
      target?.draw_term ? `第 ${target.draw_term} 期` : "等待可靠目標",
    );
    put("target-time", dateText(target?.draw_datetime));
    const errors = [];
    if (sync.error) errors.push(sync.error);
    if (Array.isArray(sync.missing_terms) && sync.missing_terms.length)
      errors.push(
        `待補期數：${sync.missing_terms.map((gap) => (typeof gap === "object" ? `${gap.after}～${gap.before} 之間` : String(gap))).join("、")}`,
      );
    else if (Number(sync.missing_terms) > 0)
      errors.push(`待補期數：${integer(sync.missing_terms)}`);
    put("sync-error", errors.join("；"));
    $("sync-error").hidden = !errors.length;
    renderPredictions(data.prediction);
    renderModels(data.models, data.training);
    renderLeaderboard();
  }
  function renderFullRanking() {
    const star = $("ranking-star")?.value || "3",
      body = $("full-ranking");
    if (!body) return;
    let ranking = analysis?.ranking;
    if (ranking && !Array.isArray(ranking))
      ranking = ranking[star] || ranking.ranking || [];
    if (!ranking?.length) {
      emptyRow(body, 4, "尚無有效封存排名");
      return;
    }
    body.replaceChildren();
    ranking.forEach((entry, index) => {
      const row = typeof entry === "number" ? { number: entry } : entry,
        tr = el("tr");
      tr.append(
        el("td", index + 1),
        el("td", String(row.number).padStart(2, "0")),
        el("td", num(row.score, 4)),
        el("td", integer(row.support)),
      );
      body.append(tr);
    });
  }
  function renderAnalysis(data) {
    analysis = data;
    $("ranking-star").parentElement.hidden = Array.isArray(data.ranking);
    put(
      "analysis-samples",
      data.samples !== undefined ? `${integer(data.samples)} 期資料` : "",
    );
    const body = $("frequency");
    body.replaceChildren();
    for (const row of data.frequency || []) {
      const tr = el("tr");
      tr.append(
        el("td", String(row.number).padStart(2, "0")),
        el("td", integer(row.count)),
        el("td", percent(row.rate)),
        el("td", integer(row.gap)),
      );
      body.append(tr);
    }
    if (!data.frequency?.length) emptyRow(body, 4, "尚無有效歷史資料");
    renderFullRanking();
    const history = $("history");
    history.replaceChildren();
    for (const draw of data.history || []) {
      const tr = el("tr"),
        term = el("td", draw.draw_term);
      term.append(el("div", dateText(draw.draw_datetime), "subtle"));
      tr.append(
        term,
        el(
          "td",
          (draw.numbers || []).map((n) => String(n).padStart(2, "0")).join(" "),
          "history-numbers",
        ),
      );
      history.append(tr);
    }
    if (!data.history?.length) emptyRow(history, 2, "尚無開獎紀錄");
    const distribution = $("distribution");
    distribution.replaceChildren();
    const stats = data.size_odd_even;
    if (stats && Object.keys(stats).length) {
      const grid = el("div", null, "distribution-grid");
      const keys = {
        small: "小號（1～40）",
        big: "大號（41～80）",
        large: "大號（41～80）",
        odd: "單數",
        even: "雙數",
        small_count: "小號（1～40）",
        big_count: "大號（41～80）",
        odd_count: "單數",
        even_count: "雙數",
      };
      for (const [key, value] of Object.entries(stats)) {
        if (!(key in keys)) continue;
        const item = el("div", null, "distribution-item");
        item.append(el("strong", integer(value)), el("span", keys[key]));
        grid.append(item);
      }
      distribution.append(grid);
    } else distribution.append(el("p", "尚無分布統計", "empty"));
    const pairs = $("cooccurrence");
    pairs.replaceChildren();
    let entries = data.cooccurrence || [];
    if (!Array.isArray(entries))
      entries = Object.entries(entries).map(([pair, count]) => ({
        numbers: pair.split(/[,\- ]+/),
        count,
      }));
    for (const item of entries.slice(0, 20)) {
      const pair = item.numbers ||
        item.pair || [item.number_a ?? item.a, item.number_b ?? item.b];
      pairs.append(
        el(
          "span",
          `${pair.map((n) => String(n).padStart(2, "0")).join(" ＋ ")} · ${integer(item.count)} 次`,
          "pair-chip",
        ),
      );
    }
    if (!entries.length) pairs.append(el("p", "尚無號碼共現統計", "empty"));
  }
  function renderBacktest(job) {
    backtestState = job.state;
    badge($("backtest-state"), stateLabel(job.state), job.state);
    const active = ["running", "queued", "training"].includes(job.state);
    $("backtest-form").querySelector("button").disabled =
      active || actionPending;
    if (job.error) message(`盲測失敗：${job.error}`, true);
    if (active || job.total > 0 || Number(job.progress) > 0) {
      const value = progress(
        "backtest-progress",
        "backtest-progress-wrap",
        job.completed,
        job.total,
        job.progress,
      );
      put(
        "backtest-detail",
        `${value === null ? "逐期工作進行中" : `${num(value, 0)}%`}${job.total > 0 ? ` · 已完成 ${integer(job.completed)}／${integer(job.total)}` : ""}${job.current_model ? ` · ${displayName(job.current_model)}` : ""}`,
      );
    } else $("backtest-progress-wrap").hidden = true;
    const report = job.report;
    if (!report) return;
    const key = JSON.stringify([
      job.run_id,
      report.run_id,
      job.finished_at,
      report.count,
      report.results?.length,
      report.method_comparison,
      report.selected_method,
    ]);
    if (key === reportKey) return;
    reportKey = key;
    const holder = $("backtest-results");
    holder.replaceChildren();
    holder.append(
      el(
        "p",
        `歷史樣本外模擬 · ${integer(report.count)} 期 · ${report.selection_reason || "比較結果不等於未來命中保證。"}`,
        "report-note",
      ),
    );
    if (Array.isArray(report.warnings))
      for (const warning of report.warnings)
        holder.append(el("p", warning, "small subtle"));
    for (const star of ["3", "4", "5"]) {
      const rows =
        report.leaderboards?.cumulative?.[star] ||
        report.leaderboards?.[star]?.cumulative ||
        report.method_comparison?.[star] ||
        [];
      const section = el("section", null, "report-section");
      section.append(el("h3", `${labels[star]}模型與投票比較`));
      const scroll = el("div", null, "table-scroll"),
        table = el("table"),
        head = el("thead"),
        headRow = el("tr");
      for (const title of [
        "模型／融合",
        "期數",
        "平均命中與區間",
        "全中率",
        "部分命中",
        "號碼命中率",
        "與隨機差距",
      ])
        headRow.append(el("th", title));
      head.append(headRow);
      const body = el("tbody");
      rows.forEach((row, index) => body.append(metricCells(row, index, true)));
      if (!rows.length) emptyRow(body, 7, "此星數尚無可用比較結果");
      table.append(head, body);
      scroll.append(table);
      section.append(scroll);
      holder.append(section);
    }
    if (reportURL) URL.revokeObjectURL(reportURL);
    reportURL = URL.createObjectURL(
      new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }),
    );
    const download = el(
      "a",
      "下載完整盲測報告（含逐期預測）",
      "report-download",
    );
    download.href = reportURL;
    download.download = "bingo-walk-forward.json";
    holder.append(download);
  }
  async function refresh() {
    if (refreshing) return;
    refreshing = true;
    try {
      if (page === "dashboard") {
        const data = await request("status");
        if (transportError) {
          message("連線已恢復，資料持續更新。");
          transportError = false;
        }
        renderDashboard(data);
      } else {
        // Historical statistics change slowly; completed reports may be large.
        const loadAnalysis = !analysis || Date.now() - lastAnalysisAt >= 30000;
        const results = await Promise.allSettled([
          loadAnalysis ? request("analysis") : Promise.resolve(null),
          request("backtest?summary=true"),
        ]);
        const failure = results.find((result) => result.status === "rejected");
        if (failure) throw failure.reason;
        if (transportError) {
          message("連線已恢復，資料持續更新。");
          transportError = false;
        }
        if (results[0].value) {
          renderAnalysis(results[0].value);
          lastAnalysisAt = Date.now();
        }
        const job = results[1].value;
        renderBacktest(job);
        const runId = job.run_id || "saved-report";
        if (job.has_report && runId !== lastReportRunId) {
          const full = await request("backtest");
          renderBacktest(full);
          lastReportRunId = runId;
        }
      }
    } catch (error) {
      transportError = true;
      message(`資料更新失敗：${error.message} 將自動重試。`, true);
      if ($("page-status")) badge($("page-status"), "頁面連線中斷", "offline");
    } finally {
      refreshing = false;
    }
  }
  async function tick() {
    if (suspended) return;
    await refresh();
    clearTimeout(timer);
    const active = ["running", "queued", "training"].includes(backtestState);
    if (!suspended)
      timer = setTimeout(tick, page === "analysis" && !active ? 30000 : 5000);
  }
  document.querySelectorAll("[data-nav]").forEach((node) => {
    const active = node.dataset.nav === page;
    node.classList.toggle("active", active);
    if (active) node.setAttribute("aria-current", "page");
  });
  document.querySelectorAll("[data-action]").forEach((button) =>
    button.addEventListener("click", async () => {
      if (actionPending) return;
      actionPending = true;
      button.disabled = true;
      try {
        const data = await request(button.dataset.action, { method: "POST" });
        message(
          data.message ||
            {
              sync: "已觸發同步，最新資料將自動更新。",
              predict: "已觸發下一期預測；已封存的同一期號碼保持固定。",
              train: "已提交背景訓練。",
              "train/stop": "已提出停止訓練要求。",
            }[button.dataset.action],
        );
        await refresh();
      } catch (error) {
        message(error.message, true);
      } finally {
        actionPending = false;
        button.disabled = false;
        if (status && page === "dashboard")
          renderModels(status.models, status.training);
      }
    }),
  );
  document.querySelectorAll("[data-star]").forEach((button) =>
    button.addEventListener("click", () => {
      selectedStar = button.dataset.star;
      document.querySelectorAll("[data-star]").forEach((node) => {
        const active = node === button;
        node.classList.toggle("active", active);
        node.setAttribute("aria-pressed", active);
      });
      renderLeaderboard();
    }),
  );
  document.querySelectorAll("[data-board]").forEach((button) =>
    button.addEventListener("click", () => {
      selectedBoard = button.dataset.board;
      document.querySelectorAll("[data-board]").forEach((node) => {
        const active = node === button;
        node.classList.toggle("active", active);
        node.setAttribute("aria-pressed", active);
      });
      renderLeaderboard();
    }),
  );
  $("ranking-star")?.addEventListener("change", renderFullRanking);
  $("backtest-form")?.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (actionPending) return;
    const form = event.currentTarget,
      values = new FormData(form),
      payload = { count: Number(values.get("count")) };
    for (const key of ["start_term", "end_term"]) {
      const value = String(values.get(key) || "").trim();
      if (value) payload[key] = value;
    }
    if (
      payload.start_term &&
      payload.end_term &&
      BigInt(payload.start_term) > BigInt(payload.end_term)
    ) {
      message("起始期號必須小於或等於結束期號。", true);
      return;
    }
    actionPending = true;
    form.querySelector("button").disabled = true;
    try {
      await request("backtest", {
        method: "POST",
        body: JSON.stringify(payload),
      });
      message("盲測已提交至背景執行，完成後會自動顯示報告。");
      await refresh();
    } catch (error) {
      message(error.message, true);
    } finally {
      actionPending = false;
      form.querySelector("button").disabled = false;
      await refresh();
      clearTimeout(timer);
      if (!suspended) timer = setTimeout(tick, 5000);
    }
  });
  window.addEventListener("pagehide", () => {
    suspended = true;
    clearTimeout(timer);
  });
  window.addEventListener("pageshow", (event) => {
    if (event.persisted) {
      suspended = false;
      tick();
    }
  });
  tick();
})();
