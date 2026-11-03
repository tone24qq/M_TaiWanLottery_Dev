/* Strategy health is read-only: no browser action changes a model or forecast. */
(() => {
  "use strict";
  const panel = document.getElementById("intelligence-details");
  if (!panel) return;
  const $ = (id) => document.getElementById(id);
  const labels = { 3: "三星", 4: "四星", 5: "五星" };
  const names = {
    frequency: "Frequency 頻率",
    cdm: "CDM 貝氏統計",
    river: "River 線上學習",
    lightgbm: "LightGBM",
    catboost: "CatBoost",
    transformer: "Transformer",
    dqn: "DQN",
    ensemble: "Ensemble 集成",
    equal: "等權投票",
    shrink: "收縮加權",
    strategy_today: "Today Only 當日資料",
    strategy_historical: "Historical 跨日歷史",
    strategy_hybrid: "Adaptive Hybrid 自適應混合",
    production: "正式採用策略",
    association_a: "關聯方案 A",
    association_b: "關聯方案 B",
    association_c: "關聯方案 C",
    borda: "排名積分",
    majority: "多數決",
  };
  const phaseNames = {
    fixed: "固定基準",
    baseline: "固定基準",
    shadow: "影子驗證",
    selection: "候選選擇",
    selecting: "候選選擇",
    validation: "獨立驗證",
    validating: "獨立驗證",
    champion: "正式策略",
    challenger: "候選策略",
    rejected: "維持原策略",
    cooldown: "切換冷卻中",
    rollback: "已回退",
    promoted: "已通過切換",
    ready: "可用",
    waiting: "等待中",
    running: "執行中",
    training: "訓練中",
    fitting: "訓練中",
    queued: "等待執行",
    completed: "已完成",
    idle: "待命",
    failed: "失敗",
    stopped: "已停止",
    disabled: "未啟用",
    pending: "等待中",
    unavailable: "未提供",
    evaluating: "評估中",
    monitoring: "持續監控",
  };
  const eventNames = {
    promotion: "候選通過驗證",
    promoted: "候選通過驗證",
    switch: "策略切換",
    rollback: "回退已驗證版本",
    rejected: "保留正式策略",
    validation_failed: "候選未通過",
    validation_started: "開始獨立驗證",
    candidate_selected: "選定影子候選",
    selection: "比較候選",
    observe: "更新實績",
    diagnostic: "自動診斷",
  };
  let data = null,
    star = "3",
    refreshing = false,
    timer = null,
    controller = null;
  let suspended = false;

  function node(tag, text, cls) {
    const result = document.createElement(tag);
    if (text !== null && text !== undefined) result.textContent = String(text);
    if (cls) result.className = cls;
    return result;
  }
  function put(id, text) {
    if ($(id)) $(id).textContent = text ?? "—";
  }
  function finite(value) {
    return (
      value !== null &&
      value !== undefined &&
      value !== "" &&
      Number.isFinite(Number(value))
    );
  }
  function number(value, digits = 0) {
    return finite(value)
      ? Number(value).toLocaleString("zh-TW", {
          minimumFractionDigits: digits,
          maximumFractionDigits: digits,
        })
      : "—";
  }
  function fraction(value, digits = 1) {
    return finite(value) ? `${number(Number(value) * 100, digits)}%` : "—";
  }
  function signed(value) {
    return finite(value)
      ? `${Number(value) > 0 ? "+" : ""}${number(value, 3)}`
      : "—";
  }
  function duration(value) {
    if (!finite(value)) return "—";
    return Number(value) < 1
      ? `${number(Number(value) * 1000, 1)} 毫秒`
      : `${number(value, 2)} 秒`;
  }
  function date(value) {
    if (!value) return "—";
    let iso = String(value).replace(" ", "T");
    if (!/(Z|[+-]\d{2}:?\d{2})$/i.test(iso)) iso += "+08:00";
    const parsed = new Date(iso);
    if (Number.isNaN(parsed.getTime())) return "時間格式待確認";
    return new Intl.DateTimeFormat("zh-TW", {
      timeZone: "Asia/Taipei",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    }).format(parsed);
  }
  function list(value) {
    return Array.isArray(value) ? value : [];
  }
  function name(value) {
    return (
      names[String(value || "").toLowerCase()] || String(value || "未命名模型")
    );
  }
  function phase(value) {
    return phaseNames[String(value || "").toLowerCase()] || "狀態待確認";
  }
  function setBadge(id, text, tone = "neutral") {
    const target = $(id);
    if (target) {
      target.textContent = text;
      target.className = `badge ${tone}`;
    }
  }
  function active(job) {
    return ["running", "training", "fitting", "queued", "evaluating"].includes(
      String(job?.state || ""),
    );
  }

  function currentAdaptiveRows() {
    return list(data?.adaptive?.models).filter(
      (row) => String(row.stars) === star,
    );
  }
  function currentMonitorRows() {
    return list(data?.monitor?.models).filter(
      (row) => String(row.stars) === star,
    );
  }
  function statWindow(holder, value, key) {
    const cell = node("div"),
      samples = value?.samples;
    cell.append(node("span", `最近 ${key} 期`));
    cell.append(
      node(
        "strong",
        Number(samples) > 0
          ? number(value.mean_hits ?? value.average_hits, 3)
          : "—",
      ),
    );
    cell.append(
      node(
        "span",
        Number(samples) > 0
          ? `平均命中 · ${number(samples)} 個樣本`
          : "尚無已結算樣本",
      ),
    );
    if (Number(samples) > 0)
      cell.append(
        node(
          "span",
          `全中率 ${fraction(value.full_hit_rate)} · 隨機差 ${signed(value.delta_vs_random)}`,
        ),
      );
    holder.append(cell);
  }
  function renderModels() {
    const target = $("intelligence-models");
    const adaptive = currentAdaptiveRows(),
      monitors = currentMonitorRows();
    const keys = [
      ...new Set(
        [...adaptive, ...monitors]
          .map((row) => row.model || row.name)
          .filter(Boolean),
      ),
    ];
    target.replaceChildren();
    if (!keys.length) {
      target.append(node("p", "此星數尚無可稽核的視窗或已結算監控。", "empty"));
      return;
    }
    for (const key of keys) {
      const selected =
        adaptive.find((row) => (row.model || row.name) === key) || {};
      const monitor =
        monitors.find((row) => (row.model || row.name) === key) || {};
      const card = node("article", null, "intelligence-model");
      const heading = node("div", null, "model-top");
      heading.append(node("h3", name(key)));
      heading.append(
        node(
          "span",
          phase(selected.phase || (monitor.samples ? "monitoring" : "waiting")),
          "badge neutral",
        ),
      );
      card.append(heading);
      const requested = selected.window ?? monitor.window;
      const effective = selected.effective_window;
      let windowText = finite(requested)
        ? `參考視窗 ${number(requested)} 期`
        : "等待視窗資訊";
      if (finite(effective) && Number(effective) !== Number(requested))
        windowText += ` · 本次可用 ${number(effective)} 期`;
      card.append(node("p", windowText, "intelligence-window"));
      const training =
        selected.training_window ??
        selected.training_samples ??
        monitor.training_window;
      card.append(
        node(
          "p",
          finite(training)
            ? `模型工件訓練資料：${number(training)} 期`
            : "模型工件訓練範圍依封存版本紀錄，與即時參考視窗分開。",
          "model-description",
        ),
      );
      if (selected.reason || monitor.repetition_reason)
        card.append(
          node(
            "p",
            selected.reason || monitor.repetition_reason,
            "model-description",
          ),
        );
      if (finite(selected.candidate_window)) {
        const stages = `候選 ${number(selected.candidate_window)} 期 · 選擇 ${number(selected.selection_samples ?? 0)}／${number(data.adaptive?.min_selection)} · 獨立驗證 ${number(selected.validation_samples ?? 0)}／${number(data.adaptive?.min_validation)}`;
        card.append(node("p", stages, "model-description"));
      }
      const recent = node("div", null, "intelligence-recent");
      for (const period of ["10", "30", "100"])
        statWindow(recent, monitor.windows?.[period], period);
      card.append(recent);
      const stability = monitor.stability?.["10"] || {};
      const details = node("div", null, "intelligence-stability");
      const hasTransitions = Number(stability.transitions) > 0;
      details.append(
        node(
          "p",
          hasTransitions
            ? `近 10 期相同選號 ${fraction(stability.unchanged_rate)} · Top-K 重疊 ${fraction(stability.top_k_overlap)}`
            : "預測重複率：等待至少兩期已結算紀錄。",
        ),
      );
      if (hasTransitions)
        details.append(
          node(
            "p",
            `完整排名平均位移 ${number(stability.rank_displacement, 2)} 名 · 排名相關 ${number(stability.rank_spearman, 3)}`,
          ),
        );
      if (finite(monitor.full_miss_streak))
        details.append(
          node(
            "p",
            `連續未全中 ${number(monitor.full_miss_streak)} 期；僅觀測，不單獨判定故障。`,
          ),
        );
      const versions = [
        selected.version || monitor.model_version,
        monitor.feature_version,
      ].filter(Boolean);
      if (versions.length)
        details.append(node("p", `版本：${versions.join("／")}`));
      card.append(details);
      const advanced = node("details", null, "model-monitor-details");
      advanced.append(node("summary", "更多穩定性與 300 期成績"));
      const more = node("div", null, "model-monitor-content");
      if (hasTransitions)
        more.append(
          node(
            "p",
            `近 10 期：輸入更新 ${number(stability.input_updates)} 次 · 特徵更新 ${number(stability.feature_updates)} 次 · 模型版本更新 ${number(stability.version_updates)} 次`,
            "model-description",
          ),
        );
      const recentLong = node("div", null, "intelligence-recent long-window");
      statWindow(recentLong, monitor.windows?.["300"], "300");
      more.append(recentLong);
      for (const period of ["30", "100"]) {
        const row = monitor.stability?.[period];
        if (row && Number(row.transitions) > 0)
          more.append(
            node(
              "p",
              `近 ${period} 期：相同選號 ${fraction(row.unchanged_rate)}，完整排名平均位移 ${number(row.rank_displacement, 2)} 名。`,
              "model-description",
            ),
          );
      }
      const distribution = monitor.score_distribution || {};
      if (finite(distribution.score_span))
        more.append(
          node(
            "p",
            `80 號分數跨度 ${number(distribution.score_span, 4)} · 相對集中度 ${number(distribution.relative_concentration, 3)}${distribution.flat_scores ? " · 分數平坦，需檢視" : ""}`,
            "model-description",
          ),
        );
      if (finite(monitor.training_age_draws))
        more.append(
          node(
            "p",
            `工件訓練截止距今 ${number(monitor.training_age_draws)} 期。`,
            "model-description",
          ),
        );
      if (monitor.audit_available === false || monitor.audit_missing === true)
        more.append(
          node(
            "p",
            "部分紀錄缺少新版特徵稽核欄位；保留原始紀錄，不補造數值。",
            "model-description",
          ),
        );
      if (selected.last_change)
        more.append(
          node(
            "p",
            `最近調整：${date(selected.last_change)}`,
            "model-description",
          ),
        );
      if (selected.rollback_version)
        more.append(
          node(
            "p",
            `可回退版本：${selected.rollback_version}`,
            "model-description",
          ),
        );
      advanced.append(more);
      card.append(advanced);
      target.append(card);
    }
  }
  function renderAlerts() {
    const target = $("intelligence-alerts"),
      alerts = list(data?.monitor?.alerts).filter(
        (row) =>
          row.stars === undefined ||
          row.stars === null ||
          String(row.stars) === star,
      );
    target.replaceChildren();
    if (!alerts.length) {
      target.append(
        node(
          "p",
          data?.monitor?.updated_at
            ? "本次已回報的監控未發現警示；樣本少仍不能據此證明模型有效。"
            : "等待開獎後的實際監控紀錄。",
          "small subtle",
        ),
      );
      return;
    }
    for (const alert of alerts.slice(0, 12)) {
      const message =
        typeof alert === "string"
          ? alert
          : alert.message || alert.reason || "資料需進一步檢視";
      const label = alert.model ? `${name(alert.model)}：` : "";
      target.append(
        node(
          "p",
          `${label}${message}`,
          `intelligence-alert${["error", "critical", "high"].includes(alert.severity) ? " bad" : ""}`,
        ),
      );
    }
  }
  function parameterText(value) {
    if (value === null || value === undefined) return "—";
    if (typeof value !== "object") return String(value);
    const fields = {
      window: "參考視窗",
      context_window: "參考視窗",
      training_window: "訓練視窗",
      method: "投票",
      profile: "設定",
      version: "版本",
      retrain_every: "重訓間隔",
      model: "模型",
      weights: "權重",
    };
    const entries = Object.entries(value)
      .slice(0, 8)
      .map(([key, item]) => {
        const display =
          key === "model" || key === "method"
            ? name(item)
            : typeof item === "object"
              ? JSON.stringify(item)
              : String(item);
        return `${fields[key] || key} ${display}`;
      });
    return entries.join("；") || "—";
  }
  function renderEvents() {
    const target = $("intelligence-events");
    const events = list(data?.events).length
      ? data.events
      : list(data?.adaptive?.recent_events);
    const filtered = events.filter(
      (event) =>
        event.stars === undefined ||
        event.stars === null ||
        String(event.stars) === star,
    );
    target.replaceChildren();
    if (!filtered.length) {
      target.append(
        node("p", "尚無此星數的策略調整紀錄；原策略會持續保留。", "empty"),
      );
      return;
    }
    for (const event of filtered.slice(0, 10)) {
      const item = node("article", null, "intelligence-event");
      const kind = event.kind || event.action || event.type;
      item.append(
        node(
          "h3",
          `${event.model ? `${name(event.model)} · ` : ""}${eventNames[kind] || "策略驗證紀錄"}`,
        ),
      );
      if (event.reason || event.message)
        item.append(node("p", event.reason || event.message));
      const meta = node("dl");
      function line(label, value) {
        meta.append(node("dt", label), node("dd", value));
      }
      if (event.target_term || event.settled_target)
        line("依據期號", event.target_term || event.settled_target);
      if (event.created_at || event.at)
        line("紀錄時間", date(event.created_at || event.at));
      if (event.before !== undefined)
        line("原本設定", parameterText(event.before));
      if (event.after !== undefined) line("新設定", parameterText(event.after));
      const validation = event.validation || event.evidence;
      if (validation && typeof validation === "object") {
        line(
          "獨立驗證",
          `樣本 ${number(validation.samples ?? validation.n)} · 平均差 ${signed(validation.delta ?? validation.mean_delta ?? validation.mean_difference)} · 保守下界 ${signed(validation.lower_bound ?? validation.ci_low ?? validation.lower)}`,
        );
        if (finite(validation.overhead_ratio ?? validation.resource_ratio))
          line(
            "相對資源成本",
            `${number(validation.overhead_ratio ?? validation.resource_ratio, 2)} 倍`,
          );
        if (
          finite(validation.champion_mean_hits) ||
          finite(validation.candidate_mean_hits)
        )
          line(
            "正式／候選成績",
            `平均命中 ${number(validation.champion_mean_hits, 3)}／${number(validation.candidate_mean_hits, 3)} · 全中率 ${fraction(validation.champion_full_hit_rate)}／${fraction(validation.candidate_full_hit_rate)}`,
          );
      }
      if (event.rollback_version) line("回退版本", event.rollback_version);
      item.append(meta);
      target.append(item);
    }
  }
  function renderResources() {
    const resources = data?.resources || {},
      jobs = data?.jobs || {};
    put(
      "resource-cpu",
      finite(resources.cpu_percent)
        ? `${number(resources.cpu_percent, 1)}%`
        : "未提供量測",
    );
    put(
      "resource-memory",
      finite(resources.rss_mb ?? resources.memory_mb)
        ? `${number(resources.rss_mb ?? resources.memory_mb, 1)} MiB`
        : "未提供量測",
    );
    put("resource-training", duration(resources.training_seconds));
    put("resource-inference", duration(resources.inference_seconds));
    put("resource-backtest", duration(resources.backtest_seconds));
    const jobLabels = {
      training: "訓練",
      inference: "推論",
      backtest: "回測",
      evaluation: "候選評估",
    };
    const running = Object.entries(jobs)
      .filter(([, job]) => active(job))
      .map(
        ([key, job]) => `${jobLabels[key] || "背景工作"}：${phase(job.state)}`,
      );
    put(
      "resource-jobs",
      running.length
        ? running.join("／")
        : Object.keys(jobs).length
          ? "目前無執行中的背景工作"
          : "等待工作資訊",
    );
    const holder = $("resource-workers");
    holder.replaceChildren();
    for (const [key, worker] of Object.entries(resources.workers || {})) {
      if (!worker || typeof worker !== "object") continue;
      const parts = [jobLabels[key] || "工作程序", phase(worker.state)];
      if (finite(worker.cpu_percent))
        parts.push(`CPU ${number(worker.cpu_percent, 1)}%`);
      if (finite(worker.rss_mb)) parts.push(`${number(worker.rss_mb, 1)} MiB`);
      if (finite(worker.elapsed_seconds))
        parts.push(`已執行 ${duration(worker.elapsed_seconds)}`);
      holder.append(node("span", parts.join(" · "), "resource-worker"));
    }
  }
  function renderResearch() {
    const research=data?.research_strategies;
    const holder=$("parallel-research-cards");
    if (!holder) return;
    holder.replaceChildren();
    if (!research) { holder.append(node("p","尚無並行實戰資料","empty")); return; }
    const status=research.stars?.[star] || {};
    const today=research.today_status || {};
    const ids={today:"Today Only 當日資料",historical:"Historical 跨日歷史",hybrid:"Adaptive Hybrid 自適應混合"};
    put("parallel-research-status",`共同實戰 ${number(status.paired_samples || 0)}／300 期 · 已涵蓋 ${number(status.days || 0)}／3 日 · 正式策略：${ids[status.champion] || "Historical 跨日歷史"}。樣本不足或未通過保守驗證時不切換。`);
    for (const id of ["today","historical","hybrid"]) {
      const card=node("article",null,"parallel-research-card");
      card.append(node("h4",ids[id]));
      const pred=research.predictions?.[star]?.[id];
      const row=(status.leaderboard || []).find(r=>r.id===id);
      const picks=node("div",null,"parallel-research-numbers");
      for (const n of pred?.numbers || []) picks.append(node("span",String(n).padStart(2,"0"),"ball"));
      if (!pred) picks.append(node("span","尚無本期封存"));
      card.append(picks);
      card.append(node("p",row?.samples ? `${number(row.samples)} 期 · 平均命中 ${number(row.mean_hits,3)} · 全中 ${number(row.full_hits)} 次（${fraction(row.full_hit_rate)}）` : "尚無已結算實戰樣本","small"));
      if (id === "today") card.append(node("p",`目標日期：${today.target_date || "—"} · 當日已知 ${number(today.available_count ?? today.history_count)} 期，特徵使用 ${number(today.history_count)} 期${today.warmup ? " · 明示先驗暖機" : ""}` ,"small subtle"));
      if (id === "hybrid" && pred) card.append(node("p",`當日 ${fraction(pred.today_weight)}／歷史 ${fraction(pred.history_weight)}` ,"small subtle"));
      card.append(node("p",pred?.reason || "每期先封存再結算，不將歷史模擬計入實戰","small subtle"));
      holder.append(card);
    }
  }

  function render() {
    if (!data) return;
    const adaptive = data.adaptive || {},
      monitor = data.monitor || {};
    const selected = currentAdaptiveRows();
    const assessing =
      adaptive.evaluating ||
      monitor.evaluating ||
      active(data.jobs?.evaluation);
    const alerts = list(monitor.alerts).length;
    const windows = [
      ...new Set(selected.map((row) => row.window).filter(finite)),
    ].sort((a, b) => a - b);
    const hasRecord = monitor.updated_at || selected.length;
    setBadge(
      "intelligence-state",
      assessing
        ? "候選評估中"
        : alerts
          ? "有監控提示"
          : hasRecord
            ? "持續監控"
            : "等待實戰",
      alerts ? "warning" : "neutral",
    );
    put(
      "intelligence-summary",
      hasRecord
        ? `${labels[star]} · ${assessing ? "候選先驗證，正式策略受保護" : "依事前封存成績監控，不因連敗強制換號"}`
        : "等待事前封存與開獎後結算資料",
    );
    put(
      "intelligence-updated",
      monitor.updated_at
        ? `監控更新 ${date(monitor.updated_at)}`
        : "尚無結算監控",
    );
    put(
      "intelligence-champion",
      windows.length
        ? `${labels[star]}正式參考：${windows.map((value) => `${number(value)} 期`).join("／")}`
        : "固定基準，等待模型資訊",
    );
    const changes = selected
      .map((row) => row.last_change)
      .filter(Boolean)
      .sort();
    put(
      "intelligence-last-change",
      changes.length ? date(changes[changes.length - 1]) : "尚未調整",
    );
    put(
      "intelligence-selections",
      adaptive.selection_count !== undefined
        ? number(adaptive.selection_count)
        : "—",
    );
    put(
      "intelligence-evaluation",
      assessing
        ? "影子候選評估中"
        : hasRecord
          ? "等待足夠獨立樣本"
          : "等待資料",
    );
    if (adaptive.status || adaptive.reason)
      put("intelligence-policy", adaptive.status || adaptive.reason);
    const consensus = monitor.consensus?.[star];
    put(
      "intelligence-consensus",
      consensus && finite(consensus.mean_top_k_jaccard)
        ? `基礎模型 ${number(consensus.models)} 個 · 選號平均重疊 ${fraction(consensus.mean_top_k_jaccard)} · 排名平均相關 ${number(consensus.mean_rank_correlation, 3)} · 分歧 ${fraction(consensus.disagreement)}`
        : "模型間共識：等待至少兩個可比較的模型。",
    );
    renderAlerts();
    renderModels();
    renderEvents();
    renderResources();
    renderResearch();
  }
  async function refresh() {
    if (refreshing || suspended || document.hidden) return;
    refreshing = true;
    controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 12000);
    try {
      const response = await fetch("/api/v1/bingo/intelligence", {
        signal: controller.signal,
        headers: { Accept: "application/json" },
      });
      if (!response.ok) throw new Error(`監控服務回應 ${response.status}`);
      const next = await response.json();
      if (!next || typeof next !== "object" || Array.isArray(next))
        throw new Error("監控資料格式無法辨識");
      data = next;
      $("intelligence-error").hidden = true;
      render();
    } catch (error) {
      if (suspended || document.hidden) return;
      setBadge("intelligence-state", "監控資料未更新", "warning");
      put(
        "intelligence-error",
        `智慧監控暫時無法更新${data ? "，畫面保留上次資料" : ""}；稍後自動重試。`,
      );
      $("intelligence-error").hidden = false;
    } finally {
      clearTimeout(timeout);
      refreshing = false;
      controller = null;
    }
  }
  async function tick() {
    clearTimeout(timer);
    if (suspended || document.hidden) return;
    await refresh();
    // A visibility/toggle event can call tick while an earlier fetch finishes.
    // Keep only the timer from the last completion, without duplicate polling.
    clearTimeout(timer);
    if (!suspended && !document.hidden)
      timer = setTimeout(tick, panel.open ? 20000 : 30000);
  }
  document.querySelectorAll("[data-intelligence-star]").forEach((button) =>
    button.addEventListener("click", () => {
      star = button.dataset.intelligenceStar;
      document.querySelectorAll("[data-intelligence-star]").forEach((item) => {
        const isActive = item === button;
        item.classList.toggle("active", isActive);
        item.setAttribute("aria-pressed", String(isActive));
      });
      render();
    }),
  );
  panel.addEventListener("toggle", () => {
    if (panel.open && !data) tick();
  });
  document.addEventListener("visibilitychange", () => {
    clearTimeout(timer);
    if (document.hidden) controller?.abort();
    else tick();
  });
  window.addEventListener("pagehide", () => {
    suspended = true;
    clearTimeout(timer);
    controller?.abort();
  });
  window.addEventListener("pageshow", (event) => {
    if (event.persisted) {
      suspended = false;
      tick();
    }
  });
  tick();
})();
