"""Tests of evidence gates, future boundaries and prior-version rollback."""

import copy
from datetime import datetime, timedelta

from taiwan_lottery.ml.bingo_adaptive import AdaptiveController
from taiwan_lottery.ml.bingo_nested import CandidateProfile


def report():
    candidate = CandidateProfile(
        "recent",
        60,
        300,
        (0.55, 0.25, 0.15, 0.05),
        24,
        200,
        method="equal",
        cdm_prior_kind="recent20",
    ).payload()
    proposals = {}
    for stars in (3, 4, 5):
        pairs = [
            dict(
                target_term=str(115001001 + i),
                source_term=str(115001000 + i),
                training_cutoff=str(115001000 + i),
                feature_cutoff=str(115001000 + i),
                champion_hits=0,
                challenger_hits=stars,
            )
            for i in range(1000)
        ]
        proposals[str(stars)] = dict(
            profile=candidate,
            paired=pairs,
            selection_end_term="115000999",
            baseline_profile_id="fixed",
            resource_ratio=1,
            comparison_count=24,
            inner_validation={"approved": True, "samples": 100},
        )
    return dict(
        mode="nested_walk_forward",
        is_live=False,
        decision={"allowed_for_live": True, "proposals": proposals},
    )


def test_verified_profile_is_future_only_and_report_cannot_be_reselected():
    control = AdaptiveController()
    control.plan("115002001", history=1000)
    original = report()
    before = copy.deepcopy(original)
    outcome = control.accept_validation(original)
    assert all(r["kind"] == "profile_validated" for r in outcome["events"])
    assert original == before
    assert (
        control.plan("115002002", history=1000)["profiles_by_star"]["3"]["id"]
        == "recent"
    )
    count = control.state["selection_count"]
    assert control.accept_validation(original)["duplicate"]
    assert control.state["selection_count"] == count
    assert AdaptiveController(control.to_dict()).status() == control.status()


def test_future_or_overlapping_validation_cannot_replace_champion():
    for mutation in ("future", "overlap", "cost"):
        control = AdaptiveController()
        control.plan("115002001", history=1000)
        data = report()
        for proposal in data["decision"]["proposals"].values():
            if mutation == "future":
                proposal["paired"][0]["training_cutoff"] = proposal["paired"][0][
                    "target_term"
                ]
            elif mutation == "overlap":
                proposal["selection_end_term"] = proposal["paired"][0]["target_term"]
            else:
                proposal["resource_ratio"] = 10
        assert all(
            e["kind"] == "profile_retained"
            for e in control.accept_validation(data)["events"]
        )
        assert control.state["profiles_by_star"]["3"]["id"] == "fixed-v1"


def test_prior_profile_can_return_only_after_new_independent_archived_evidence():
    control = AdaptiveController()
    control.plan("115002001", history=1000)
    control.accept_validation(report())
    all_events = []
    for i in range(100):
        control.state["observation_count"] = i + 31
        at = datetime(2026, 1, 3, 7, 5) + timedelta(minutes=5 * i)
        target = str(115003001 + i)
        rows = [
            dict(
                model="policy",
                stars=k,
                target_term=target,
                source_term=str(int(target) - 1),
                created_at=(at - timedelta(minutes=4)).isoformat(),
                target_time=at.isoformat(),
                hit_count=k if name == "fixed-v1" else 0,
                metadata={"policy_profile_id": name, "scoring_cost_seconds": 0.001},
            )
            for k in (3, 4, 5)
            for name in ("fixed-v1", "recent")
        ]
        events = control.observe_profile_outcomes(target, rows)
        all_events.extend(events)
        if i < 99:
            assert not events
    assert {e["kind"] for e in all_events} == {"profile_rollback"}
    assert control.state["profiles_by_star"]["3"]["id"] == "fixed-v1"
