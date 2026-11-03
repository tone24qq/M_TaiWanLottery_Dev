/* Neighbor research UI: historical rates stay distinct from forecast scores. */
(() => {
  "use strict";
  const root = document.getElementById("neighbors");
  if (!root) return;
  const $ = (id) => document.getElementById(id);
  const form = $("neighbor-form");
  const names = {
    neighbor: "鄰號／跨期轉移",
    random: "可重現隨機選號",
    frequency: "Frequency 頻率",
    cdm: "CDM 貝氏統計",
    river: "River 線上學習",
    lightgbm: "LightGBM",
    catboost: "CatBoost",
    transformer: "Transformer",
    dqn: "DQN",
    ensemble: "Ensemble 集成",
    equal: "Equal 等權",
    shrink: "Shrink 收縮加權",
    borda: "Borda 排名積分",
    majority: "Majority 多數決",
    association_a: "ABC 關聯 A",
    association_b: "ABC 關聯 B",
    association_c: "ABC 關聯 C",
  };
  let busy = false;
  let queued = false;
  let timer = null;
  let reportKey = null;
  let reportRequest = false;
  let latestSignature = null;
  let currentReference = null;

  function el(tag, value, cls) {
    const node = document.createElement(tag);
    if (value !== undefined && value !== null) node.textContent = String(value);
    if (cls) node.className = cls;
    return node;
  }
  function finite(value) {
    return value !== null && value !== undefined && Number.isFinite(Number(value));
  }
  function number(value, digits = 2) {
    return finite(value)
      ? Number(value).toLocaleString("zh-TW", { maximumFractionDigits: digits })
      : "—";
  }
  function percent(value, digits = 1) {
    return finite(value) ? `${number(Number(value) * 100, digits)}%` : "—";
  }
  function pp(value) {
    return finite(value)
      ? `${Number(value) > 0 ? "+" : ""}${number(Number(value) * 100, 2)} 個百分點`
      : "—";
  }
  function pvalue(value) {
    if (!finite(value)) return "—";
    const n = Number(value);
    return n > 0 && n < 0.001 ? n.toExponential(2) : number(n, 4);
  }
  function ball(value) {
    return String(value).padStart(2, "0");
  }
  function shift(value) {
    const n = Number(value);
    if (!Number.isFinite(n)) return "—";
    return n === 0 ? "0 原號重開" : `${n > 0 ? "+" : ""}${n} ${n > 0 ? "上升" : "下降"}${Math.abs(n) === 1 ? "／鄰接連號" : ""}`;
  }
  function range(row, prefix = "ci") {
    const low = row?.[`${prefix}_low`];
    const high = row?.[`${prefix}_high`];
    return finite(low) && finite(high) ? `${percent(low, 2)}～${percent(high, 2)}` : "—";
  }
  function query() {
    const params = new URLSearchParams();
    for (const name of ["window", "lag", "snapshot", "source"]) {
      const value = form.elements.namedItem(name)?.value;
      if (value !== "" && value !== undefined) params.set(name, value);
    }
    return params;
  }
  async function getJSON(url) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 12000);
    try {
      const response = await fetch(url, { signal: controller.signal, cache: "no-store" });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return await response.json();
    } finally {
      window.clearTimeout(timeout);
    }
  }
  function state(value, reason) {
    const labels = { ready: "統計已就緒", waiting: "等待歷史資料", running: "背景計算中", failed: "分析暫不可用" };
    const badge = $("neighbor-state");
    badge.textContent = labels[value] || "等待資料";
    badge.className = `badge ${value === "ready" ? "neutral" : value === "failed" ? "error" : "neutral"}`;
    const error = $("neighbor-error");
    error.hidden = !reason;
    error.textContent = reason || "";
  }
  function table(headers, rows, label) {
    const wrap = el("div", null, "neighbor-scroll");
    wrap.tabIndex = 0;
    wrap.setAttribute("role", "region");
    wrap.setAttribute("aria-label", `${label}（表格可橫向捲動）`);
    const grid = el("table", null, "neighbor-table");
    const thead = el("thead");
    const tr = el("tr");
    for (const heading of headers) {
      const th = el("th", heading);
      th.scope = "col";
      tr.append(th);
    }
    thead.append(tr);
    const body = el("tbody");
    for (const row of rows) body.append(row);
    grid.append(thead, body);
    wrap.append(grid);
    return wrap;
  }
  function sampleText(sample) {
    if (!sample) return "未提供分段計數";
    return `${number(sample.hit_count ?? sample.hits, 0)}／${number(sample.trigger_count ?? sample.trials ?? sample.triggers, 0)}`;
  }
  function metricCells(row, reference) {
    const rate = el("td", percent(row.conditional_rate, 2));
    rate.append(el("span", `獨立配對條件率 ${percent(row.inferential_rate, 2)}`, "subtle"));
    rate.append(el("span", `95% 家族同時區間 ${range(row)}`, "subtle"));
    const proof = el("td", row.independently_validated ? "通過獨立關係驗證" : "尚未證明優於隨機");
    if (row.discovery || row.confirmation) {
      proof.append(el("span", `發現 ${sampleText(row.discovery)}；獨立確認 ${sampleText(row.confirmation)}`, "subtle"));
    }
    const triggers = el("td", number(row.trigger_count, 0));
    triggers.append(el("span", `推論用 ${number(row.inferential_trigger_count, 0)}`, "subtle"));
    const hits = el("td", number(row.hit_count, 0));
    hits.append(el("span", `推論用 ${number(row.inferential_hit_count, 0)}`, "subtle"));
    const cells = [
      triggers,
      hits,
      rate,
      el("td", percent(row.random_baseline ?? 0.25)),
      el("td", pp(row.delta)),
      el("td", pvalue(row.adjusted_p_value)),
      proof,
    ];
    if (reference !== undefined) {
      cells.push(el("td", reference && finite(reference.conditional_rate) && finite(row.conditional_rate)
        ? pp(Number(reference.conditional_rate) - Number(row.conditional_rate))
        : "—"));
    }
    return cells;
  }
  function relationMap(view) {
    const map = new Map();
    for (const source of view?.by_source || []) {
      for (const row of source.relations || []) map.set(`${source.source}:${row.target}`, row);
    }
    return map;
  }
  function renderChanges(view, reference, params) {
    const holder = $("neighbor-changes");
    const snapshot = Number(params.get("snapshot") || 0);
    holder.hidden = snapshot === 0;
    holder.replaceChildren();
    if (snapshot === 0) return;
    holder.append(el("p", `正在查看前 ${snapshot} 期的資料截點：第 ${view.source_term || "—"} 期。此截點的統計不使用其後開獎答案。`));
    if (!reference) {
      holder.append(el("p", "最新截點尚未就緒，暫無同關係差異比較。"));
      return;
    }
    const observed = new Set(view.source_numbers || []);
    const latest = new Set(reference.source_numbers || []);
    const added = [...latest].filter((n) => !observed.has(n)).map(ball);
    const removed = [...observed].filter((n) => !latest.has(n)).map(ball);
    holder.append(el("p", `與最新第 ${reference.source_term || "—"} 期相比：最新新增 ${added.join("、") || "無"}；最新未再開出 ${removed.join("、") || "無"}。`));
    holder.append(el("p", `表格「最新−觀察期」比較相同來源／目標、視窗與跨期距離的歷史條件率；差異不是預測優勢。${params.get("source") ? "" : "目前顯示兩期的共同來源；選擇單一來源可查看全部目標的差異。"}`));
  }
  function render(view, reference, params) {
    const snapshot = Number(params.get("snapshot") || 0);
    const sourceValue = params.get("source");
    const windowValue = params.get("window") || "1000";
    const lagValue = params.get("lag") || "1";
    const history = view.history || {};
    const count = history.count ?? history.samples ?? history.draw_count;
    $("neighbor-context").textContent = `第 ${view.source_term || "—"} 期 · ${snapshot ? `前 ${snapshot} 期截點` : "最新截點"} · 最近 ${windowValue} 期視窗 · 後 ${lagValue} 期${finite(count) ? ` · 實際歷史 ${number(count, 0)} 期` : ""}。邊界與缺期配對已排除。`;
    const numbers = $("neighbor-numbers");
    numbers.replaceChildren();
    for (const n of view.source_numbers || []) numbers.append(el("span", ball(n), "neighbor-number"));
    renderChanges(view, reference, params);
    const refMap = relationMap(reference);
    const offsetRef = new Map((reference?.offset_statistics || []).map((row) => [Number(row.offset), row]));
    const extra = snapshot > 0 ? ["最新−觀察期"] : [];
    const offsets = (view.offset_statistics || []).map((row) => {
      const tr = el("tr");
      tr.append(el("td", shift(row.offset)), ...metricCells(row, snapshot ? offsetRef.get(Number(row.offset)) || null : undefined));
      return tr;
    });
    const offsetHolder = $("neighbor-offset-statistics");
    offsetHolder.replaceChildren();
    if (offsets.length) {
      offsetHolder.append(table(["位移／方向", "觸發次數", "命中次數", "歷史條件率／獨立配對推論", "隨機基準", "與隨機差距", "校正後 p 值", "證據狀態", ...extra], offsets, "位移命中統計"));
      offsetHolder.append(el("p", view.offset_statistics[0]?.testing_note || "位移彙總保留同一期相依性，不執行單號獨立二項檢定，校正後 p 值欄顯示 —。", "small subtle"));
    } else offsetHolder.append(el("p", "此條件尚無位移統計", "empty"));
    const relations = $("neighbor-relations");
    relations.replaceChildren();
    const entries = (view.by_source || []).filter((row) => sourceValue ? Number(row.source) === Number(sourceValue) : row.active);
    entries.forEach((entry, index) => {
      const details = el("details", null, "neighbor-card");
      details.open = index === 0;
      const rows = entry.relations || [];
      details.append(el("summary", `來源 ${ball(entry.source)}${entry.active ? "（觀察期開出）" : "（觀察期未開出）"} → ${sourceValue ? "全部 80 號" : `${rows.length} 個有效鄰號關係`}`));
      details.append(el("p", sourceValue ? "包含原號、全部上升與下降位移；每個目標只計算一次。" : "原號與 ±1、±2、±3、±5；超出號碼邊界的關係不建立。", "neighbor-card-note"));
      const rendered = rows.map((row) => {
        const tr = el("tr");
        tr.append(el("td", ball(row.target)), el("td", shift(row.offset)), ...metricCells(row, snapshot ? refMap.get(`${entry.source}:${row.target}`) || null : undefined));
        return tr;
      });
      details.append(table(["目標號碼", "位移／方向", "觸發次數", "命中次數", "歷史條件率／獨立配對推論", "隨機基準", "與隨機差距", "Bonferroni 校正 p", "獨立驗證", ...extra], rendered, `來源 ${ball(entry.source)} 的關係證據`));
      relations.append(details);
    });
    if (!entries.length) relations.append(el("p", "此截點沒有可顯示的來源號碼", "empty"));
    const methods = $("neighbor-methodology");
    methods.replaceChildren();
    const segments = view.validation_segments || {};
    const parameters = view.parameters || {};
    const family = view.validation?.family_hypotheses ?? parameters.multiple_tests ?? parameters.family_size ?? view.family_size;
    methods.append(el("p", `引擎版本：${view.version || "—"}${finite(family) ? `；完整檢定集合：${number(family, 0)} 種關係` : ""}。`));
    if (history.from || history.start_term) methods.append(el("p", `歷史範圍：${history.from || history.start_term}～${history.to || history.end_term || view.source_term}。`));
    if (segments.discovery || segments.confirmation) {
      methods.append(el("p", `發現截點：${segments.discovery?.to || "—"}；選擇驗證：${segments.selection_validation?.from || "—"}～${segments.selection_validation?.to || "—"}；獨立確認：${segments.confirmation?.from || "—"}～${segments.confirmation?.to || "—"}。所有視窗共用驗證界線。`));
    }
    if (view.validation?.descriptive_warning) methods.append(el("p", view.validation.descriptive_warning));
    if (view.validation?.ci_method) methods.append(el("p", view.validation.ci_method));
    if (view.warning) methods.append(el("p", view.warning));
  }
  function clearResults(reason) {
    $("neighbor-context").textContent = reason || "等待背景計算完成；完成後會自動更新。";
    $("neighbor-numbers").replaceChildren();
    $("neighbor-changes").hidden = true;
    $("neighbor-offset-statistics").replaceChildren(el("p", "等待可用統計", "empty"));
    $("neighbor-relations").replaceChildren(el("p", "等待可用關係", "empty"));
    $("neighbor-methodology").replaceChildren();
  }
  function renderReport(report) {
    const holder = $("neighbor-backtest-results");
    holder.replaceChildren();
    holder.append(el("p", `歷史樣本外模擬 ${number(report.count, 0)} 期。${report.selection_reason || "尚未證明優於隨機。"}`, "neighbor-report-note"));
    holder.append(el("p", "單號條件基準為 25%；三星、四星、五星的全中基準依無放回超幾何機率各自計算。回測與正式實戰分開。", "small subtle"));
    for (const star of ["3", "4", "5"]) {
      const rows = report.leaderboards?.cumulative?.[star] || report.leaderboards?.[star]?.cumulative || [];
      const section = el("section", null, "neighbor-report-star");
      section.append(el("h4", `${{ 3: "三星", 4: "四星", 5: "五星" }[star]}實績比較`));
      const evidence = report.validation?.[star];
      if (evidence) {
        section.append(el("p", evidence.status || "尚未證明優於隨機", "neighbor-report-note"));
        section.append(el("p", `前段發現 ${number(evidence.discovery?.samples, 0)} 期；後段獨立確認 ${number(evidence.confirmation?.samples, 0)} 期${evidence.confirmation_from ? `（自 ${evidence.confirmation_from} 期）` : ""}。${Number(evidence.confirmation?.samples || 0) < 100 ? "確認樣本不足 100 期，未通過證據門檻。" : ""}`, "small subtle"));
        section.append(el("p", `獨立確認平均命中 ${number(evidence.confirmation?.mean_hits, 3)}，保守區間 ${number(evidence.confirmation?.mean_ci_low, 3)}～${number(evidence.confirmation?.mean_ci_high, 3)}；全中率校正後 p 值 ${pvalue(evidence.confirmation?.adjusted_full_hit_p)}。`, "small subtle"));
      }
      const rendered = rows.map((row) => {
        const tr = el("tr");
        const mean = el("td", number(row.mean_hits, 3));
        mean.append(el("span", `95% 區間 ${number(row.mean_ci_low ?? row.ci_low, 3)}～${number(row.mean_ci_high ?? row.ci_high, 3)}`, "subtle"));
        const full = el("td", `${number(row.full_hits, 0)}／${number(row.samples, 0)}（${percent(row.full_hit_rate, 2)}）`);
        full.append(el("span", `隨機全中 ${percent(row.random_full_hit_rate ?? row.random_full_rate, 3)}`, "subtle"));
        full.append(el("span", `全中率區間 ${range(row, "full_hit_ci")}`, "subtle"));
        tr.append(el("td", names[row.name] || row.name || "—"), el("td", number(row.samples, 0)), mean, full, el("td", number(row.random_mean, 3)), el("td", finite(row.delta_vs_random) ? `${Number(row.delta_vs_random) > 0 ? "+" : ""}${number(row.delta_vs_random, 3)}` : "—"));
        return tr;
      });
      if (rendered.length) section.append(table(["研究候選／模型", "回測期數", "平均命中與區間", "實際全中／期數", "隨機平均命中", "與隨機平均差"], rendered, `${star}星回測比較`));
      else section.append(el("p", "此星數尚無可比較結果", "empty"));
      holder.append(section);
    }
  }
  async function refreshReport() {
    if (reportRequest) return;
    reportRequest = true;
    try {
      const summary = await getJSON("/api/v1/bingo/backtest?summary=true");
      if (!summary.has_report) return;
      const key = JSON.stringify([summary.run_id, summary.finished_at]);
      if (key === reportKey) return;
      const job = await getJSON("/api/v1/bingo/backtest");
      reportKey = key;
      if (job.report?.mode !== "neighbor_walk_forward") return;
      renderReport(job.report);
    } catch (_) {
      // Neighbor statistics remain available when no backtest report can be read.
    } finally {
      reportRequest = false;
    }
  }
  function schedule(delay) {
    window.clearTimeout(timer);
    if (!document.hidden) timer = window.setTimeout(refresh, delay);
  }
  async function refresh() {
    if (busy) {
      queued = true;
      return;
    }
    if (document.hidden) return;
    busy = true;
    void refreshReport();
    const params = query();
    const key = params.toString();
    let delay = 15000;
    try {
      const result = await getJSON(`/api/v1/bingo/neighbors?${key}`);
      if (key !== query().toString()) return;
      state(result.state, result.reason);
      if (result.state !== "ready" || !result.analysis) {
        clearResults(result.reason);
        delay = 5000;
        return;
      }
      const signature = `${result.analysis.snapshot_hash || result.source_term}:${params.get("window")}:${params.get("lag")}:${params.get("source") || "all"}`;
      let reference = null;
      if (Number(params.get("snapshot") || 0) > 0) {
        if (signature !== latestSignature) {
          const current = new URLSearchParams(params);
          current.set("snapshot", "0");
          const latest = await getJSON(`/api/v1/bingo/neighbors?${current}`);
          if (latest.state === "ready") {
            currentReference = latest.analysis;
            latestSignature = signature;
          } else currentReference = null;
        }
        reference = currentReference;
      }
      if (key === query().toString()) render(result.analysis, reference, params);
    } catch (error) {
      if (key === query().toString()) {
        const reason = error.name === "AbortError" ? "分析查詢逾時，將自動重試。" : `暫時無法讀取鄰號分析：${error.message}`;
        state("failed", reason);
        clearResults(reason);
      }
      delay = 5000;
    } finally {
      busy = false;
      if (queued || key !== query().toString()) {
        queued = false;
        schedule(0);
      } else schedule(delay);
    }
  }
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    void refresh();
  });
  form.addEventListener("change", () => void refresh());
  document.addEventListener("visibilitychange", () => {
    window.clearTimeout(timer);
    if (!document.hidden) void refresh();
  });
  void refresh();
})();
