/* ABC association candidates: read-only, sealed server records, independent UI. */
(() => {
  "use strict";
  const panel = document.getElementById("associations-details");
  if (!panel) return;
  const $ = (id) => document.getElementById(id);
  const labels = { 3: "三星", 4: "四星", 5: "五星" };
  const ids = ["A", "B", "C"];
  let snapshot = null,
    selectedStar = "3",
    selectedBoard = "cumulative";
  let refreshing = false,
    suspended = false,
    timer = null,
    controller = null;
  let renderedKey = null;

  function node(tag, text, cls) {
    const value = document.createElement(tag);
    if (text !== null && text !== undefined) value.textContent = String(text);
    if (cls) value.className = cls;
    return value;
  }
  function put(id, value) {
    if ($(id)) $(id).textContent = value ?? "—";
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
          maximumFractionDigits: digits,
          minimumFractionDigits: digits,
        })
      : "—";
  }
  function fraction(value) {
    return finite(value) ? `${number(Number(value) * 100, 1)}%` : "—";
  }
  function signed(value) {
    return finite(value)
      ? `${Number(value) > 0 ? "+" : ""}${number(value, 3)}`
      : "—";
  }
  function list(value) {
    return Array.isArray(value) ? value : [];
  }
  function date(value) {
    if (!value) return "—";
    let iso = String(value).replace(" ", "T");
    if (!/(Z|[+-]\d{2}:?\d{2})$/i.test(iso)) iso += "+08:00";
    const parsed = new Date(iso);
    if (Number.isNaN(parsed.getTime())) return "時間格式待確認";
    return new Intl.DateTimeFormat("zh-TW", {
      timeZone: "Asia/Taipei",
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    }).format(parsed);
  }
  function badge(text, tone = "neutral") {
    const target = $("association-state");
    target.textContent = text;
    target.className = `badge ${tone}`;
  }
  function confidence(value) {
    const labels = {
      insufficient: "樣本不足",
      insufficient_samples: "樣本不足",
      insufficient_evidence: "證據不足",
      uncertain: "不確定",
      exploratory: "探索性證據",
      not_proven: "尚未證明優勢",
      low: "低可信程度",
      medium: "初步觀察",
      high: "較高可信程度",
      validated: "完成規則驗證",
      reliable: "已有規則證據",
    };
    if (value === null || value === undefined || value === "")
      return "尚無可信程度資料";
    if (typeof value === "object") {
      const status = value.label || value.status || value.level;
      const reason = value.reason || value.message;
      return (
        [
          status ? labels[String(status).toLowerCase()] || String(status) : "",
          reason,
        ]
          .filter(Boolean)
          .join("；") || "尚無可信程度資料"
      );
    }
    return labels[String(value).toLowerCase()] || String(value);
  }
  function validNumbers(values) {
    return (
      Array.isArray(values) &&
      values.length === Number(selectedStar) &&
      new Set(values).size === values.length &&
      values.every((v) => Number.isInteger(v) && v >= 1 && v <= 80)
    );
  }
  function scoreFor(candidate, value) {
    if (Array.isArray(candidate.scores) && candidate.scores.length === 80)
      return candidate.scores[value - 1];
    const entry = list(candidate.ranking).find(
      (row) => row && typeof row === "object" && Number(row.number) === value,
    );
    return entry?.score;
  }
  function candidateName(model) {
    const match = String(model || "").match(/(?:association[_:\-])?([abc])$/i);
    return match
      ? `候選 ${match[1].toUpperCase()}`
      : String(model || "未命名候選");
  }
  function historyText(candidate) {
    const history = candidate.history || {};
    const start = candidate.history_start ?? history.from;
    const end = candidate.history_end ?? history.to;
    const count = candidate.history_count ?? history.count;
    const parts = [];
    if (start && end) parts.push(`歷史期號 ${start}～${end}`);
    if (finite(count)) parts.push(`共 ${number(count)} 期`);
    if (candidate.version) parts.push(`版本 ${candidate.version}`);
    return parts.join(" · ") || "尚無歷史範圍或版本資訊";
  }
  function evidenceRows(candidate) {
    const globalRules = Array.isArray(snapshot?.rules)
      ? snapshot.rules
      : Object.values(snapshot?.rules || {});
    const entries = [...list(candidate.evidence), ...list(candidate.rules)];
    const seen = new Set(),
      result = [];
    for (const value of entries) {
      let rule = value;
      if (typeof value === "string" || typeof value === "number")
        rule = globalRules.find(
          (item) => item && String(item.id ?? item.rule_id) === String(value),
        ) || { rule_id: String(value) };
      if (!rule || typeof rule !== "object" || Array.isArray(rule)) continue;
      const identity = rule.id ?? rule.rule_id;
      if (identity !== undefined && seen.has(String(identity))) continue;
      if (identity !== undefined) seen.add(String(identity));
      result.push(rule);
    }
    return result;
  }
  function evidenceStages(rule) {
    const stages = [
      ["discovery", "發現段"],
      ["selection_validation", "選擇驗證段"],
      ["confirmation", "獨立確認段"],
    ]
      .filter(([key]) => rule[key] && typeof rule[key] === "object")
      .map(([key, label]) => ({ key, label, stats: rule[key] }));
    if (stages.length) {
      // Pooled descriptive counts never replace any independent validation stage.
      if (finite(rule.trigger_count) || finite(rule.success_count))
        stages.unshift({ key: null, label: "全歷史描述", stats: rule });
      return stages;
    }
    const stageNames = {
      all_known_descriptive: "全歷史探索描述",
      discovery: "發現段",
      selection_validation: "選擇驗證段",
      confirmation: "獨立確認段",
    };
    return [
      { key: null, label: stageNames[rule.stage] || "封存統計", stats: rule },
    ];
  }
  function valuesText(value) {
    if (Array.isArray(value))
      return value
        .map((v) => (finite(v) ? String(v).padStart(2, "0") : String(v)))
        .join("、");
    return finite(value)
      ? String(value).padStart(2, "0")
      : value === null || value === undefined
        ? "—"
        : String(value);
  }
  function ruleText(evidence) {
    const title = evidence.description || evidence.reason;
    if (title) return String(title);
    const antecedent = valuesText(evidence.antecedent),
      consequent = valuesText(evidence.consequent ?? evidence.consequence);
    const lag = finite(evidence.lag)
      ? Number(evidence.lag) === 0
        ? "同期歷史關聯"
        : `間隔 ${number(evidence.lag)} 期的歷史關聯`
      : "歷史關聯";
    return `${antecedent} → ${consequent} · ${lag}`;
  }
  function cell(row, text, cls) {
    row.append(node("td", text, cls));
  }
  function renderEvidence(holder, candidate, id) {
    const evidence = evidenceRows(candidate);
    holder.replaceChildren();
    holder.append(
      node(
        "p",
        "以下為已知歷史的規則統計；條件頻率（可能含平滑）、Lift 與可信程度都不是下期勝率。發現、選擇驗證與獨立確認段分開列示，不混合樣本。",
        "small subtle",
      ),
    );
    if (evidence.length) {
      const scroll = node(
        "div",
        null,
        "table-scroll association-evidence-scroll",
      );
      const table = node("table", null, "association-evidence-table");
      const caption = node(
        "caption",
        `候選 ${id} 的歷史關聯規則證據`,
        "sr-only",
      );
      const head = node("thead"),
        headRow = node("tr");
      for (const label of [
        "歷史規則",
        "資料階段",
        "成功／觸發",
        "有效樣本",
        "條件頻率",
        "隨機基準",
        "Lift",
        "保守區間",
        "可信程度",
      ]) {
        const th = node("th", label);
        th.scope = "col";
        headRow.append(th);
      }
      head.append(headRow);
      const body = node("tbody");
      for (const rule of evidence.slice(0, 20)) {
        for (const stage of evidenceStages(rule)) {
          const stats = stage.stats,
            row = node("tr"),
            first = node("td");
          first.append(node("span", ruleText(rule)));
          if (rule.rule_id || rule.id)
            first.append(
              node("p", `規則 ${rule.rule_id || rule.id}`, "small subtle"),
            );
          if (rule.reliable === false)
            first.append(
              node("p", "探索規則，尚未通過全部獨立確認。", "small subtle"),
            );
          row.append(first);
          const phaseCell = node("td", stage.label),
            history = stage.key
              ? rule.split_history?.[stage.key]
              : rule.history;
          if (history?.from && history?.to)
            phaseCell.append(
              node("p", `${history.from}～${history.to}`, "small subtle"),
            );
          row.append(phaseCell);
          cell(
            row,
            `${number(stats.success_count ?? stats.successes)}／${number(stats.trigger_count ?? stats.triggers)}`,
          );
          cell(
            row,
            number(
              stats.effective_samples ?? stats.sample_count ?? stats.triggers,
            ),
          );
          cell(
            row,
            fraction(
              stats.conditional_frequency ?? stats.smoothed_rate ?? stats.rate,
            ),
          );
          cell(row, fraction(stats.random_baseline ?? rule.random_baseline));
          cell(row, number(stats.lift, 3));
          cell(
            row,
            finite(stats.ci_low) && finite(stats.ci_high)
              ? `${fraction(stats.ci_low)}～${fraction(stats.ci_high)}`
              : "—",
          );
          const reliability =
            stats.confidence ?? (stage.key ? undefined : rule.confidence);
          cell(
            row,
            reliability !== undefined
              ? confidence(reliability)
              : stats.approved === true
                ? "此段通過門檻"
                : stats.approved === false
                  ? "此段未通過門檻"
                  : "尚無可信程度資料",
          );
          body.append(row);
        }
      }
      table.append(caption, head, body);
      scroll.append(table);
      holder.append(scroll);
      if (evidence.length > 20)
        holder.append(
          node(
            "p",
            `顯示前 20／${number(evidence.length)} 條規則；完整證據可下載候選紀錄查看。`,
            "small subtle",
          ),
        );
    } else
      holder.append(
        node("p", "尚無可列出的規則證據；不補造支持數或可信程度。", "empty"),
      );
    const rankingDetails = node("details", null, "association-full-ranking");
    rankingDetails.append(node("summary", "完整 1～80 號排名與分數"));
    const rankingHolder = node("div");
    rankingDetails.append(rankingHolder);
    rankingDetails.addEventListener("toggle", () => {
      if (rankingDetails.open && !rankingHolder.dataset.loaded) {
        renderRanking(rankingHolder, candidate);
        rankingHolder.dataset.loaded = "true";
      }
    });
    holder.append(rankingDetails);
    const download = node("button", "下載完整候選紀錄", "association-download");
    download.type = "button";
    download.addEventListener("click", () => {
      const record = {
        candidate: id,
        stars: Number(selectedStar),
        version: snapshot.version,
        source_term: snapshot.source_term,
        target_term: snapshot.target_term,
        sealed_at: snapshot.sealed_at,
        ...candidate,
        resolved_rules: evidenceRows(candidate),
        window_selection: snapshot.window_selection?.[id] ?? null,
      };
      const url = URL.createObjectURL(
        new Blob([JSON.stringify(record, null, 2)], {
          type: "application/json",
        }),
      );
      const anchor = node("a");
      anchor.href = url;
      anchor.download = `bingo-association-${id}-${selectedStar}-${snapshot.target_term || "unsealed"}.json`;
      anchor.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    });
    holder.append(download);
  }
  function renderRanking(holder, candidate) {
    const ranking = list(candidate.ranking);
    const numbers = ranking.map((value) =>
      Number(
        typeof value === "object" && value !== null ? value.number : value,
      ),
    );
    if (
      numbers.length !== 80 ||
      new Set(numbers).size !== 80 ||
      !numbers.every(
        (value) => Number.isInteger(value) && value >= 1 && value <= 80,
      )
    ) {
      holder.append(node("p", "尚未提供完整有效的 80 號排名。", "empty"));
      return;
    }
    const scroll = node("div", null, "table-scroll association-ranking-scroll");
    const table = node("table", null, "association-ranking-table");
    const head = node("thead"),
      headRow = node("tr");
    for (const label of ["排名", "號碼", "相對關聯分數"]) {
      const th = node("th", label);
      th.scope = "col";
      headRow.append(th);
    }
    head.append(headRow);
    const body = node("tbody");
    ranking.forEach((entry, index) => {
      const row = node("tr"),
        value = numbers[index];
      cell(row, number(index + 1));
      cell(row, String(value).padStart(2, "0"));
      cell(
        row,
        number(
          typeof entry === "object" &&
            entry !== null &&
            entry.score !== undefined
            ? entry.score
            : scoreFor(candidate, value),
          4,
        ),
      );
      body.append(row);
    });
    table.append(head, body);
    scroll.append(table);
    holder.append(scroll);
  }
  function renderCards(candidates) {
    const holder = $("association-cards");
    const openCards = new Set(
      [...holder.querySelectorAll(".association-evidence[open]")].map(
        (detail) => detail.dataset.candidate,
      ),
    );
    holder.replaceChildren();
    for (const id of ids) {
      const candidate = candidates?.[id];
      const card = node("article", null, "association-card");
      card.dataset.candidate = id;
      const heading = node("div", null, "association-card-header");
      heading.append(node("h3", `候選 ${id}`));
      heading.append(
        node(
          "span",
          candidate
            ? confidence(candidate.reliability ?? candidate.label)
            : "等待候選",
          "badge neutral",
        ),
      );
      card.append(heading);
      const picks = node("div", null, "association-numbers");
      const valid = candidate && validNumbers(candidate.numbers);
      if (valid)
        for (const value of candidate.numbers)
          picks.append(node("span", String(value).padStart(2, "0"), "ball"));
      else
        picks.append(
          node(
            "p",
            candidate ? "候選資料不完整，等待有效結果。" : "尚無本期候選。",
            "empty",
          ),
        );
      card.append(picks);
      if (valid) {
        const scores = candidate.numbers
          .map((value) =>
            finite(scoreFor(candidate, value))
              ? `${String(value).padStart(2, "0")} 號 ${number(scoreFor(candidate, value), 4)}`
              : null,
          )
          .filter(Boolean);
        if (scores.length)
          card.append(
            node(
              "p",
              `相對關聯分數：${scores.join(" · ")}`,
              "association-score-line",
            ),
          );
      }
      const reasons = list(candidate?.reasons)
        .filter((value) => typeof value === "string")
        .slice(0, 3);
      if (reasons.length) {
        const why = node("ul", null, "association-reasons");
        for (const reason of reasons) why.append(node("li", reason));
        card.append(why);
      } else card.append(node("p", "尚無可顯示的選號依據。", "small subtle"));
      if (candidate) {
        card.append(node("p", historyText(candidate), "association-history"));
        const details = node("details", null, "association-evidence");
        details.dataset.candidate = id;
        details.append(
          node(
            "summary",
            `查看規則證據（${number(evidenceRows(candidate).length)} 筆）`,
          ),
        );
        const content = node("div", null, "association-evidence-content");
        details.append(content);
        const fill = () => {
          if (!content.dataset.loaded) {
            renderEvidence(content, candidate, id);
            content.dataset.loaded = "true";
          }
        };
        details.addEventListener("toggle", () => {
          if (details.open) fill();
        });
        if (openCards.has(id)) {
          details.open = true;
          fill();
        }
        card.append(details);
      }
      holder.append(card);
    }
  }
  function renderLeaderboard(rows) {
    const target = $("association-leaderboard");
    target.replaceChildren();
    if (!rows.length) {
      const row = node("tr"),
        empty = node("td", "此星數與期間尚無已結算關聯實戰。", "empty");
      empty.colSpan = 6;
      row.append(empty);
      target.append(row);
      return;
    }
    for (const record of rows) {
      const row = node("tr"),
        identity = node("td");
      identity.append(node("span", candidateName(record.model || record.name)));
      identity.append(
        node(
          "span",
          record.credibility ||
            (Number(record.samples) < 30 ? "樣本不足" : "需持續樣本外驗證"),
          "subtle",
        ),
      );
      identity.append(
        node(
          "span",
          record.evidence ||
            (record.proven_better_than_random
              ? "觀察區間高於隨機；仍須持續驗證"
              : "尚未證明優於隨機"),
          "subtle",
        ),
      );
      row.append(identity);
      cell(row, number(record.samples));
      const mean = node(
        "td",
        Number(record.samples) > 0
          ? number(record.mean_hits ?? record.average_hits, 3)
          : "—",
      );
      if (
        Number(record.samples) > 0 &&
        finite(record.ci_low) &&
        finite(record.ci_high)
      )
        mean.append(
          node(
            "span",
            `95% 區間 ${number(record.ci_low, 3)}～${number(record.ci_high, 3)}`,
            "subtle",
          ),
        );
      row.append(mean);
      cell(
        row,
        Number(record.samples) > 0
          ? `${number(record.full_hits)} 次／${fraction(record.full_hit_rate)}`
          : "—",
      );
      cell(row, number(record.partial_hits));
      const delta = node(
        "td",
        Number(record.samples) > 0 ? signed(record.delta_vs_random) : "—",
      );
      if (finite(record.random_mean))
        delta.append(
          node("span", `隨機平均 ${number(record.random_mean, 3)}`, "subtle"),
        );
      row.append(delta);
      target.append(row);
    }
  }
  function render() {
    if (!snapshot) return;
    const candidates = snapshot.by_star?.[selectedStar];
    const available = ids.some((id) => candidates?.[id]);
    const sealed =
      Boolean(snapshot.sealed_at) && available && snapshot.state !== "waiting";
    badge(
      sealed ? "已封存" : available ? "尚未封存" : "等待候選",
      sealed ? "good" : "neutral",
    );
    put(
      "association-summary",
      `${labels[selectedStar]}${snapshot.target_term ? ` · 目標第 ${snapshot.target_term} 期` : " · 目標期號尚未提供"}${snapshot.source_term ? ` · 資料截至 ${snapshot.source_term} 期` : ""}`,
    );
    put(
      "association-sealed",
      sealed
        ? `封存 ${date(snapshot.sealed_at)}`
        : "尚無封存時間，不視為正式實戰",
    );
    if (snapshot.error) {
      badge("本期關聯未完成", "warning");
      put("association-error", `本期關聯計算回報：${String(snapshot.error)}`);
      $("association-error").hidden = false;
    }
    const rows = list(snapshot.leaderboards?.[selectedStar]?.[selectedBoard]);
    const key = JSON.stringify([
      selectedStar,
      selectedBoard,
      snapshot.version,
      snapshot.source_term,
      snapshot.target_term,
      snapshot.sealed_at,
      candidates,
      snapshot.rules,
      rows,
    ]);
    if (key !== renderedKey) {
      renderCards(candidates);
      renderLeaderboard(rows);
      renderedKey = key;
    }
  }
  async function refresh() {
    if (refreshing || suspended || document.hidden) return;
    refreshing = true;
    controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 12000);
    try {
      const response = await fetch("/api/v1/bingo/associations", {
        headers: { Accept: "application/json" },
        signal: controller.signal,
      });
      if (!response.ok) throw new Error(`關聯資料服務回應 ${response.status}`);
      const data = await response.json();
      if (!data || typeof data !== "object" || Array.isArray(data))
        throw new Error("關聯資料格式無法辨識");
      snapshot = data;
      $("association-error").hidden = true;
      render();
    } catch (error) {
      if (suspended || document.hidden) return;
      badge("關聯資料未更新", "warning");
      put(
        "association-error",
        `關聯候選暫時無法更新${snapshot ? "，保留上次資料" : ""}；稍後自動重試。`,
      );
      $("association-error").hidden = false;
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
    clearTimeout(timer);
    if (!suspended && !document.hidden) timer = setTimeout(tick, 20000);
  }
  document.querySelectorAll("[data-association-star]").forEach((button) =>
    button.addEventListener("click", () => {
      selectedStar = button.dataset.associationStar;
      document.querySelectorAll("[data-association-star]").forEach((other) => {
        const active = other === button;
        other.classList.toggle("active", active);
        other.setAttribute("aria-pressed", String(active));
      });
      render();
    }),
  );
  document.querySelectorAll("[data-association-board]").forEach((button) =>
    button.addEventListener("click", () => {
      selectedBoard = button.dataset.associationBoard;
      document.querySelectorAll("[data-association-board]").forEach((other) => {
        const active = other === button;
        other.classList.toggle("active", active);
        other.setAttribute("aria-pressed", String(active));
      });
      render();
    }),
  );
  panel.addEventListener("toggle", () => {
    if (panel.open && !snapshot) tick();
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
