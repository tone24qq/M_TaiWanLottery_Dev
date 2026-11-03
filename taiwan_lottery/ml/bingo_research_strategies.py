"""Three prospectively archived research strategies; no assumption of a day effect.

Today Only is trained on the target calendar day's known draws. Historical uses
cross-day artifacts. Hybrid weights depend only on prior paired live outcomes.
Historical remains production until >=300 common targets across >=3 dates pass
prespecified multiplicity/cooldown/noninferiority checks.
"""

from __future__ import annotations

from collections import defaultdict
import copy
from datetime import datetime
import hashlib
import json
import math

import numpy as np

from taiwan_lottery.ml.bingo_evaluation import normalize_scores, summarize_outcomes

FAMILIES = ("today", "historical", "hybrid")
NAMES = {
    "today": "Today Only 當日資料",
    "historical": "Historical 跨日歷史",
    "hybrid": "Adaptive Hybrid 自適應混合",
}
VERSION = "prospective-strategies-v1"


class ResearchStrategies:
    def __init__(self, state=None, *, simulation=False):
        self.state = (
            copy.deepcopy(state)
            if state
            else dict(
                version=VERSION,
                is_live=not simulation,
                last_target=None,
                observations=0,
                processed={},
                groups={},
                recent_pairs=[],
                dates=[],
                comparison_count=0,
                champions={str(k): "historical" for k in (3, 4, 5)},
                last_change={str(k): -100 for k in (3, 4, 5)},
                events=[],
            )
        )
        self.state.setdefault("is_live", not simulation)
        if self.state.get("version") != VERSION:
            raise ValueError("並行策略狀態版本不相容")

    def mix_weight(self, stars, available_today):
        if available_today == 0:
            return 0.0, "目標當天尚無已開期數，當日方案使用均勻先驗；混合保留歷史訊息"
        rows = self.state["recent_pairs"][-300:]
        if len(rows) < 30:
            return 0.5, "少於30期已結算配對樣本，維持等權混合"
        differences = np.asarray(
            [
                r["hits"][str(stars)]["today"] - r["hits"][str(stars)]["historical"]
                for r in rows
            ],
            dtype=float,
        )
        shrink = len(rows) / (len(rows) + 100.0)
        effect = float(differences.mean() / stars)
        weight = float(np.clip(0.5 + 0.2 * math.tanh(effect * shrink), 0.35, 0.65))
        return weight, f"依先前{len(rows)}期配對命中差收縮加權；未宣稱優勢"

    def score(self, today_scores, historical_scores, today_count):
        predictions = {}
        for stars in (3, 4, 5):
            key = str(stars)
            day = np.asarray(normalize_scores(today_scores[key]))
            historical = np.asarray(normalize_scores(historical_scores[key]))
            weight, reason = self.mix_weight(stars, today_count)
            vectors = {
                "today": day,
                "historical": historical,
                "hybrid": weight * day + (1 - weight) * historical,
            }
            predictions[key] = {}
            for name, vector in vectors.items():
                ranking = (np.lexsort((np.arange(80), -vector)) + 1).tolist()
                predictions[key][name] = dict(
                    name=NAMES[name],
                    numbers=ranking[:stars],
                    scores=vector.tolist(),
                    ranking=ranking,
                    today_weight=1.0
                    if name == "today"
                    else 0.0
                    if name == "historical"
                    else weight,
                    history_weight=0.0
                    if name == "today"
                    else 1.0
                    if name == "historical"
                    else 1 - weight,
                    reason=reason
                    if name == "hybrid"
                    else "只使用目標當天已知資料與當日工件"
                    if name == "today"
                    else "使用跨日歷史資料與歷史工件",
                )
        return predictions

    def observe(self, target_term, rows):
        chosen = [row for row in rows if row.get("model", "").startswith("strategy_")]
        if not chosen:
            return []
        indexed = {
            (r["stars"], r["model"].removeprefix("strategy_")): r for r in chosen
        }
        if (
            len(chosen) != 9
            or len(indexed) != 9
            or any(
                (stars, name) not in indexed for stars in (3, 4, 5) for name in FAMILIES
            )
        ):
            raise ValueError("並行比較必須具備三策略與三種星數的完整事前封存")
        for row in indexed.values():
            if self.state["is_live"] and row.get("is_live") is False:
                raise ValueError("歷史模擬不得計入並行實戰")
            if str(row["target_term"]) != str(target_term) or int(
                row["source_term"]
            ) >= int(target_term):
                raise ValueError("策略來源或目標期號含未來資料")
            if datetime.fromisoformat(row["created_at"]) >= datetime.fromisoformat(
                row["target_time"]
            ):
                raise ValueError("並行策略必須開獎前封存")
            if (
                len(row["numbers"]) != row["stars"]
                or len(row["scores"]) != 80
                or len(set(row["numbers"]) & set(row["actual_numbers"]))
                != row["hit_count"]
            ):
                raise ValueError("並行策略結算與封存不一致")
            metadata = row.get("metadata", {})
            if row["model"] == "strategy_today" and metadata.get(
                "history_dates"
            ) not in ([], [row["target_time"][:10]]):
                raise ValueError("當日策略不得混入其他日期資料")
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    f"{k[0]}:{k[1]}": {
                        "hits": r["hit_count"],
                        "numbers": r["numbers"],
                        "hash": r["decision_hash"],
                    }
                    for k, r in indexed.items()
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        if str(target_term) in self.state["processed"]:
            if self.state["processed"][str(target_term)] != fingerprint:
                raise ValueError("並行策略重複結算內容衝突")
            return []
        if self.state["last_target"] and int(target_term) <= int(
            self.state["last_target"]
        ):
            raise ValueError("並行策略必須依期號前進")
        hits = {str(k): {} for k in (3, 4, 5)}
        for (stars, name), row in indexed.items():
            group = self.state["groups"].setdefault(
                f"{stars}:{name}",
                dict(counts=[0] * (stars + 1), latest_target=None, updated_at=None),
            )
            group["counts"][row["hit_count"]] += 1
            group.update(
                latest_target=str(target_term), updated_at=row.get("settled_at")
            )
            hits[str(stars)][name] = row["hit_count"]
        day = next(iter(indexed.values()))["target_time"][:10]
        self.state["observations"] += 1
        self.state["last_target"] = str(target_term)
        self.state["recent_pairs"].append(
            dict(target_term=str(target_term), day=day, hits=hits)
        )
        self.state["recent_pairs"] = self.state["recent_pairs"][-1000:]
        self.state["dates"] = sorted(set(self.state["dates"] + [day]))[-90:]
        self.state["processed"][str(target_term)] = fingerprint
        while len(self.state["processed"]) > 1000:
            self.state["processed"].pop(next(iter(self.state["processed"])))
        events = []
        # Fixed checkpoints bound repeated selection; no reaction to one loss.
        if (
            self.state["observations"] >= 300
            and self.state["observations"] % 100 == 0
            and len(self.state["dates"]) >= 3
        ):
            pairs = self.state["recent_pairs"]
            for stars in (3, 4, 5):
                key = str(stars)
                current = self.state["champions"][key]
                for candidate in FAMILIES:
                    if candidate == current:
                        continue
                    self.state["comparison_count"] += 1
                    attempt = self.state["comparison_count"]
                    alpha = 0.05 / (18 * attempt * (attempt + 1))
                    differences = np.array(
                        [
                            r["hits"][key][candidate] - r["hits"][key][current]
                            for r in pairs
                        ],
                        dtype=float,
                    )
                    full = np.array(
                        [
                            float(r["hits"][key][candidate] == stars)
                            - float(r["hits"][key][current] == stars)
                            for r in pairs
                        ]
                    )
                    lower = float(differences.mean()) - 2 * stars * math.sqrt(
                        math.log(2 / alpha) / (2 * len(pairs))
                    )
                    full_lower = float(full.mean()) - 2 * math.sqrt(
                        math.log(2 / alpha) / (2 * len(pairs))
                    )
                    approved = (
                        lower > 0.03
                        and full_lower >= -0.02
                        and self.state["observations"] - self.state["last_change"][key]
                        >= 100
                    )
                    event = dict(
                        kind="research_strategy_promoted"
                        if approved
                        else "research_strategy_retained",
                        stars=stars,
                        target_term=str(target_term),
                        before=current,
                        after=candidate if approved else current,
                        reason="至少300期、跨至少3日的事前配對比較通過；僅影響未來正式策略"
                        if approved
                        else "數百期比較仍未通過平均命中／全中率保守門檻，維持原正式策略",
                        validation=dict(
                            samples=len(pairs),
                            days=len(set(r["day"] for r in pairs)),
                            mean_delta=float(differences.mean()),
                            ci_low=lower,
                            full_hit_delta=float(full.mean()),
                            full_hit_ci_low=full_lower,
                            alpha=alpha,
                        ),
                    )
                    events.append(event)
                    if approved:
                        self.state["champions"][key] = candidate
                        self.state["last_change"][key] = self.state["observations"]
                        break
        self.state["events"] = (self.state["events"] + events)[-30:]
        return events

    def status(self):
        result = {}
        for stars in (3, 4, 5):
            key = str(stars)
            ranks = []
            for name in FAMILIES:
                group = self.state["groups"].get(f"{stars}:{name}", {})
                values = [
                    dict(hit_count=i, count=n)
                    for i, n in enumerate(group.get("counts", []))
                ]
                stats = summarize_outcomes(values, stars, comparison_count=9)
                ranks.append(
                    dict(
                        stats,
                        id=name,
                        name=NAMES[name],
                        updated_at=group.get("updated_at"),
                    )
                )
            ranks.sort(key=lambda r: (-r["mean_hits"], r["id"]))
            result[key] = dict(
                champion=self.state["champions"][key],
                leaderboard=ranks,
                paired_samples=len(self.state["recent_pairs"]),
                days=len(self.state["dates"]),
                minimum_samples=300,
                minimum_days=3,
            )
        return dict(
            version=VERSION,
            is_live=self.state["is_live"],
            last_target=self.state["last_target"],
            observations=self.state["observations"],
            stars=result,
            events=copy.deepcopy(self.state["events"]),
            reason="三策略同一期事前封存並分開結算；至少數百期、多日比較後才評估正式切換",
        )

    def to_dict(self):
        return copy.deepcopy(self.state)
