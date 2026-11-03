"""Prospective paired A/B/C association evidence, isolated from live rankings.

This tracker never selects numbers, adopts a rule, or changes production policy.
Its inputs are complete pre-draw decisions revealed only after their target has
settled. Cumulative histograms and bounded paired records survive a restart.
"""

from __future__ import annotations

import copy
from datetime import datetime
import hashlib
import json

import numpy as np

from taiwan_lottery.bingo_clock import to_taipei
from taiwan_lottery.ml.bingo_evaluation import summarize_outcomes

VERSION = "association-tracker-v1"
FAMILIES = ("a", "b", "c")
MODELS = {f"association_{family}": family for family in FAMILIES}
NAMES = {family: f"關聯方案 {family.upper()}" for family in FAMILIES}
STARS = (3, 4, 5)
HISTORY_LIMIT = 1000
REQUEST_COOLDOWN = 30


def _json_default(value):
    if isinstance(value, datetime):
        return to_taipei(value).isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported association evidence value: {type(value).__name__}")


def _json_copy(value):
    try:
        return json.loads(
            json.dumps(
                value, default=_json_default, ensure_ascii=False, allow_nan=False
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("關聯方案證據必須是有限且可序列化的資料") from exc


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _stamp(value, label):
    try:
        if isinstance(value, str):
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if not isinstance(value, datetime):
            raise ValueError(label)
        return to_taipei(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"關聯方案缺少有效的{label}") from exc


def _term(value, label):
    result = str(value)
    if (
        isinstance(value, bool)
        or not result.isdigit()
        or not 1 <= len(result) <= 20
        or int(result) <= 0
    ):
        raise ValueError(f"關聯方案{label}無效")
    return result


def _numbers(values, expected, label):
    if (
        not isinstance(values, (list, tuple))
        or len(values) != expected
        or len(set(values)) != expected
        or any(type(number) is not int or not 1 <= number <= 80 for number in values)
    ):
        raise ValueError(f"關聯方案{label}必須是 {expected} 個不重複的 1～80 號碼")
    return list(values)


def _rules(metadata):
    values = metadata.get("rules", metadata.get("rules_used", []))
    if isinstance(values, dict):
        values = list(values.values())
    return (
        [rule for rule in values if isinstance(rule, dict)]
        if isinstance(values, list)
        else []
    )


def _rule_expired(rule, target, target_time):
    if rule.get("expired") is True or rule.get("is_expired") is True:
        return True
    for name in ("valid_through_term", "expires_after_term", "last_valid_term"):
        value = rule.get(name)
        if value is not None and str(value).isdigit() and int(target) > int(value):
            return True
    for name in ("expires_at", "valid_until"):
        if rule.get(name):
            # Date-only values are not silently interpreted as a midnight cutoff.
            value = rule[name]
            if isinstance(value, datetime) or isinstance(value, str) and "T" in value:
                if target_time >= _stamp(value, "規則期限"):
                    return True
    return False


def _diagnostics(row):
    metadata = row["metadata"]
    rules = _rules(metadata)
    computed = sum(
        _rule_expired(rule, row["target_term"], _stamp(row["target_time"], "目標時間"))
        for rule in rules
    )
    declared = metadata.get("expired_rule_count", metadata.get("expired_rules", 0))
    if isinstance(declared, list):
        declared = len(declared)
    if isinstance(declared, dict):
        declared = len(declared)
    if not isinstance(declared, int) or isinstance(declared, bool) or declared < 0:
        declared = 0
    expired = max(computed, declared)
    score_span = float(np.ptp(row["scores"]))
    return dict(
        expired_rules=expired,
        rule_count=len(rules),
        flat_scores=score_span <= 1e-12,
        score_span=score_span,
    )


def _validated(row, target, is_live):
    family = MODELS.get(row.get("model"))
    if family is None or type(row.get("stars")) is not int or row["stars"] not in STARS:
        raise ValueError("關聯比較僅接受 A／B／C 的三星、四星、五星")
    if _term(row.get("target_term"), "目標期號") != target:
        raise ValueError("關聯比較目標期號不一致")
    source = _term(row.get("source_term"), "來源期號")
    if int(source) >= int(target):
        raise ValueError("關聯方案來源含當期或未來資料")
    metadata = _json_copy(row.get("metadata", {}))
    if not isinstance(metadata, dict):
        raise ValueError("關聯方案 metadata 必須是物件")
    marker = row.get("is_live", metadata.get("is_live"))
    simulation_marker = row.get("simulation", metadata.get("simulation"))
    if (
        marker is not None
        and (not isinstance(marker, bool) or marker != is_live)
        or simulation_marker is not None
        and (not isinstance(simulation_marker, bool) or simulation_marker == is_live)
    ):
        raise ValueError("關聯方案實戰與歷史模擬不得混合")
    created = _stamp(row.get("created_at"), "封存時間")
    deadline = _stamp(row.get("target_time"), "目標時間")
    if created >= deadline:
        raise ValueError("關聯方案必須於開獎前封存")
    if row.get("source_time") is not None:
        source_time = _stamp(row["source_time"], "來源時間")
        if source_time > created or source_time >= deadline:
            raise ValueError("關聯方案來源時間含未來資料")
    if (
        row.get("settled_at") is not None
        and _stamp(row["settled_at"], "結算時間") < deadline
    ):
        raise ValueError("關聯方案不得在開獎前結算")
    for name in (
        "training_cutoff",
        "trained_through",
        "feature_cutoff",
        "features_through",
        "history_start",
        "history_end",
        "history_start_term",
        "history_end_term",
    ):
        value = metadata.get(name)
        if value is not None and (not str(value).isdigit() or int(value) > int(source)):
            raise ValueError("關聯方案歷史截止期含未來資料")
    for rule in _rules(metadata):
        for name in (
            "training_cutoff",
            "trained_through",
            "validated_through",
            "evidence_through",
            "source_term",
        ):
            value = rule.get(name)
            if value is not None and (
                not str(value).isdigit() or int(value) > int(source)
            ):
                raise ValueError("關聯規則證據含未來資料")
    numbers = _numbers(row.get("numbers"), row["stars"], "選號")
    actual = _numbers(row.get("actual_numbers"), 20, "實際開獎")
    hits = row.get("hit_count")
    if (
        type(hits) is not int
        or not 0 <= hits <= row["stars"]
        or len(set(numbers) & set(actual)) != hits
    ):
        raise ValueError("關聯方案命中數與封存選號不一致")
    scores = np.asarray(row.get("scores"), dtype=float)
    if (
        scores.shape != (80,)
        or not np.isfinite(scores).all()
        or not np.isfinite(float(scores.max()) - float(scores.min()))
    ):
        raise ValueError("關聯方案必須提供完整 80 個有限分數")
    return dict(
        model=row["model"],
        family=family,
        stars=row["stars"],
        target_term=target,
        source_term=source,
        source_time=_stamp(row["source_time"], "來源時間").isoformat()
        if row.get("source_time")
        else None,
        created_at=created.isoformat(),
        target_time=deadline.isoformat(),
        numbers=numbers,
        scores=scores.tolist(),
        actual_numbers=sorted(actual),
        hit_count=hits,
        decision_hash=str(row.get("decision_hash", "")),
        version=str(row.get("version", "unknown")),
        settled_at=_stamp(row["settled_at"], "結算時間").isoformat()
        if row.get("settled_at")
        else None,
        metadata=metadata,
    )


class AssociationTracker:
    """Paired shadow evidence only; no action here can replace a formal model."""

    def __init__(self, state=None, simulation=False):
        self.state = (
            _json_copy(state)
            if state is not None
            else dict(
                version=VERSION,
                is_live=not simulation,
                last_target=None,
                last_target_time=None,
                observations=0,
                groups={},
                recent_pairs=[],
                processed={},
                diagnostics=[],
                revalidation_requests=[],
                last_requests={},
            )
        )
        if self.state.get("version") != VERSION:
            raise ValueError("關聯方案追蹤狀態版本不相容")
        if self.state.get("is_live") is not (not simulation):
            raise ValueError("關聯方案實戰與歷史模擬狀態不得混合")
        for name, default in (
            ("groups", {}),
            ("recent_pairs", []),
            ("processed", {}),
            ("diagnostics", []),
            ("revalidation_requests", []),
            ("last_requests", {}),
        ):
            self.state.setdefault(name, copy.deepcopy(default))

    def observe(self, target, rows):
        chosen = [
            row for row in rows if str(row.get("model", "")).startswith("association_")
        ]
        if not chosen:
            return []
        target = _term(target, "目標期號")
        indexed = {}
        for raw in chosen:
            row = _validated(raw, target, self.state["is_live"])
            key = (row["stars"], row["family"])
            if key in indexed:
                raise ValueError("關聯比較不得重複同一策略與星數")
            indexed[key] = row
        if len(indexed) != 9 or any(
            (stars, family) not in indexed for stars in STARS for family in FAMILIES
        ):
            raise ValueError("關聯比較必須具備三方案與三種星數的完整九列事前封存")
        target_times = {row["target_time"] for row in indexed.values()}
        actual_sets = {tuple(row["actual_numbers"]) for row in indexed.values()}
        if len(target_times) != 1 or len(actual_sets) != 1:
            raise ValueError("關聯方案的目標時間或實際開獎結果不一致")
        fingerprint = _digest(
            {f"{stars}:{family}": row for (stars, family), row in indexed.items()}
        )
        if target in self.state["processed"]:
            if self.state["processed"][target] != fingerprint:
                raise ValueError("關聯方案重複結算內容衝突")
            return []
        if self.state["last_target"] and int(target) <= int(self.state["last_target"]):
            raise ValueError("關聯方案必須依真實期號順序結算")
        target_time = next(iter(target_times))
        if self.state.get("last_target_time") and _stamp(
            target_time, "目標時間"
        ) <= _stamp(self.state["last_target_time"], "上期時間"):
            raise ValueError("關聯方案期號與時間順序不一致")
        # Validate the entire target before touching any cumulative state.
        diagnostics = {key: _diagnostics(row) for key, row in indexed.items()}
        observation = self.state["observations"] + 1
        hits = {str(stars): {} for stars in STARS}
        predictions = {str(stars): {} for stars in STARS}
        events = []
        for (stars, family), row in indexed.items():
            key = f"{stars}:{family}"
            group = self.state["groups"].setdefault(
                key,
                dict(
                    counts=[0] * (stars + 1),
                    versions={},
                    older_version_samples=0,
                    expired_rule_forecasts=0,
                    expired_rules=0,
                    flat_score_forecasts=0,
                    latest={},
                ),
            )
            group["counts"][row["hit_count"]] += 1
            group["versions"][row["version"]] = (
                group["versions"].get(row["version"], 0) + 1
            )
            while len(group["versions"]) > 64:
                oldest = next(iter(group["versions"]))
                group["older_version_samples"] += group["versions"].pop(oldest)
            checked = diagnostics[(stars, family)]
            group["expired_rule_forecasts"] += int(checked["expired_rules"] > 0)
            group["expired_rules"] += checked["expired_rules"]
            group["flat_score_forecasts"] += int(checked["flat_scores"])
            summary = dict(
                target_term=target,
                source_term=row["source_term"],
                target_time=target_time,
                version=row["version"],
                rule_version=row["metadata"].get("rule_version"),
                evidence=copy.deepcopy(row["metadata"].get("evidence", [])),
                numbers=list(row["numbers"]),
                hit_count=row["hit_count"],
                settled_at=row["settled_at"],
                decision_hash=row["decision_hash"],
                score_hash=_digest(row["scores"]),
                **checked,
            )
            group["latest"] = summary
            hits[str(stars)][family] = row["hit_count"]
            predictions[str(stars)][family] = summary
            if (
                checked["expired_rules"] or checked["flat_scores"]
            ) and observation - self.state["last_requests"].get(
                key, -REQUEST_COOLDOWN
            ) >= REQUEST_COOLDOWN:
                reasons = []
                if checked["expired_rules"]:
                    reasons.append("封存規則的有效期限已過，需使用獨立後續資料重新驗證")
                if checked["flat_scores"]:
                    reasons.append(
                        "80號分數平坦，需檢查規則／輸入有效性；不代表應強制換號"
                    )
                event = dict(
                    kind="association_revalidation_requested",
                    target_term=target,
                    model=row["model"],
                    stars=stars,
                    reason="；".join(reasons),
                    automatic_adoption=False,
                    requires_independent_validation=True,
                    evidence=checked,
                    version=row["version"],
                    rule_version=row["metadata"].get("rule_version"),
                )
                events.append(event)
                self.state["last_requests"][key] = observation
        self.state["observations"] = observation
        self.state["last_target"] = target
        self.state["last_target_time"] = target_time
        self.state["processed"][target] = fingerprint
        while len(self.state["processed"]) > HISTORY_LIMIT:
            self.state["processed"].pop(next(iter(self.state["processed"])))
        self.state["recent_pairs"].append(
            dict(
                target_term=target,
                target_time=target_time,
                day=target_time[:10],
                hits=hits,
                predictions=predictions,
                actual_numbers=list(next(iter(actual_sets))),
            )
        )
        self.state["recent_pairs"] = self.state["recent_pairs"][-HISTORY_LIMIT:]
        report = dict(
            target_term=target,
            expired_rule_forecasts=sum(
                value["expired_rules"] > 0 for value in diagnostics.values()
            ),
            expired_rules=sum(value["expired_rules"] for value in diagnostics.values()),
            flat_score_forecasts=sum(
                value["flat_scores"] for value in diagnostics.values()
            ),
            revalidation_requests=len(events),
            automatic_adoption=False,
        )
        self.state["diagnostics"] = (self.state["diagnostics"] + [report])[-30:]
        self.state["revalidation_requests"] = (
            self.state["revalidation_requests"] + events
        )[-30:]
        return copy.deepcopy(events)

    def status(self):
        stars_result = {}
        for stars in STARS:
            key = str(stars)
            leaderboard = []
            for family in FAMILIES:
                group = self.state["groups"].get(f"{stars}:{family}", {})
                histogram = [
                    dict(hit_count=hits, count=count)
                    for hits, count in enumerate(group.get("counts", []))
                ]
                cumulative = summarize_outcomes(histogram, stars, comparison_count=9)
                recent = {
                    str(window): summarize_outcomes(
                        [
                            dict(hit_count=row["hits"][key][family])
                            for row in self.state["recent_pairs"][-window:]
                        ],
                        stars,
                        comparison_count=9,
                    )
                    for window in (100, 300)
                }
                leaderboard.append(
                    dict(
                        cumulative,
                        id=family,
                        model=f"association_{family}",
                        name=NAMES[family],
                        cumulative=cumulative,
                        recent=recent,
                        updated_at=group.get("latest", {}).get("settled_at"),
                        latest=copy.deepcopy(group.get("latest", {})),
                        versions=copy.deepcopy(group.get("versions", {})),
                        older_version_samples=group.get("older_version_samples", 0),
                        diagnostics=dict(
                            expired_rule_forecasts=group.get(
                                "expired_rule_forecasts", 0
                            ),
                            expired_rules=group.get("expired_rules", 0),
                            flat_score_forecasts=group.get("flat_score_forecasts", 0),
                        ),
                    )
                )
            leaderboard.sort(key=lambda row: (-row["mean_hits"], row["id"]))
            stars_result[key] = dict(
                leaderboard=leaderboard,
                paired_samples=len(self.state["recent_pairs"]),
                total_paired_samples=self.state["observations"],
                automatic_adoption=False,
            )
        return dict(
            version=VERSION,
            is_live=self.state["is_live"],
            last_target=self.state["last_target"],
            observations=self.state["observations"],
            stars=stars_result,
            recent_pairs_count=len(self.state["recent_pairs"]),
            diagnostics=copy.deepcopy(self.state["diagnostics"]),
            revalidation_requests=copy.deepcopy(self.state["revalidation_requests"]),
            automatic_adoption=False,
            reason="ABC僅比較同一期事前封存樣本；規則到期或平坦分數只提出獨立再驗證，不改寫選號或取代正式策略",
        )

    def to_dict(self):
        return copy.deepcopy(self.state)
