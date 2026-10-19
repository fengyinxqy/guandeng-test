"""Synthetic boundary fixtures only; these are not business evidence."""
import pandas as pd
import pytest

from main import evaluate, predict_flags, prepare, rule_mask, select_policy, split_time


def raw_frame():
    return pd.DataFrame({
        "request_id": [1, 2, 3],
        "request_time": ["2026-09-01T08:15:00+08:00", "2026-09-01T08:16:00+08:00", "2026-09-01T08:17:00+08:00"],
        "node_status_synced_at": ["2026-09-01T00:00:00Z"] * 3,
        "service_cluster": ["fixture"] * 3, "model_tier": ["fixture"] * 3,
        "available_slots": [4, 4, 3], "payload_kb": [0, 10, 10],
        "success": [1, 0, None],
    })


def test_timezone_boundaries_and_unknown_outcomes():
    frame, count = prepare(pd.concat([raw_frame(), raw_frame().iloc[[0]]], ignore_index=True))
    assert count == 1
    assert frame.status_age_minutes.tolist() == [15, 16, 17]
    assert rule_mask(frame, 15, None).tolist() == [False, True, True]
    assert rule_mask(frame, None, 3).tolist() == [False, False, True]
    result = evaluate(frame, rule_mask(frame, 15, 3))
    assert result["known_outcomes"] == 2 and result["unknown_outcomes"] == 1
    assert result["alerted_failures"] == 1 and result["alerted_unknowns"] == 1
    assert result["scenario_saving_yuan"] == 100  # 120 - 2 * 10
    assert result["unknown_as_failure_saving_yuan"] == 220
    assert evaluate(frame, rule_mask(frame, 15, 3), rescue_rate=0)["scenario_saving_yuan"] == -20
    assert frame.payload_kb.eq(0).sum() == 1


def test_conflicting_ids_are_not_silently_deduplicated():
    raw = raw_frame()
    raw.loc[1, "request_id"] = 1
    with pytest.raises(ValueError, match="Conflicting"):
        prepare(raw)


@pytest.mark.parametrize("column,value", [
    ("request_time", "not-a-time"), ("node_status_synced_at", "2026-09-02T00:00:00Z"),
    ("success", 2), ("available_slots", -1), ("payload_kb", -1),
])
def test_invalid_inputs_fail_explicitly(column, value):
    raw = raw_frame()
    raw.loc[0, column] = value
    with pytest.raises((ValueError, TypeError)):
        prepare(raw)


def test_predict_requires_no_outcome_and_unseen_cluster_uses_global_rule():
    frame, _ = prepare(raw_frame().drop(columns="success"), require_labels=False)
    policy = {"age_threshold_minutes": 15, "slot_threshold": 3,
              "cluster_modes": {}, "unseen_cluster_mode": "both"}
    assert predict_flags(frame, policy).tolist() == [False, True, True]
    policy["cluster_modes"] = {"fixture": "slots"}
    assert predict_flags(frame, policy).tolist() == [False, False, True]


def test_equal_timestamps_do_not_cross_split():
    raw = raw_frame()
    raw.loc[2, "request_time"] = raw.loc[1, "request_time"]
    frame, _ = prepare(raw)
    train, validation, cutoff = split_time(frame)
    assert train.request_dt.max() < validation.request_dt.min()
    assert len(train) == 1 and len(validation) == 2
    assert cutoff == validation.request_dt.min()


def test_policy_selection_is_deterministic_on_training_data():
    frame, _ = prepare(raw_frame())
    train, _, _ = split_time(frame)
    policy, candidates = select_policy(train)
    assert len(candidates) == 48
    assert policy == select_policy(train)[0]


def test_all_missing_labels_are_not_reported_as_zero_failure_rate():
    raw = raw_frame()
    raw["success"] = float("nan")
    frame, _ = prepare(raw)
    scores = evaluate(frame, rule_mask(frame, None, None))
    assert scores["observed_failure_rate"] is None
    assert scores["precision_known"] is None and scores["recall_known"] is None


def test_partial_rescue_new_failures_and_unknown_sensitivity():
    frame, _ = prepare(raw_frame())
    alerts = pd.Series(True, index=frame.index)
    scores = evaluate(frame, alerts, rescue_rate=0.5, new_failure_rate=0.05)
    assert scores["scenario_saving_yuan"] == pytest.approx(24)
    assert scores["unknown_as_success_saving_yuan"] == pytest.approx(18)
    assert scores["unknown_as_failure_saving_yuan"] == pytest.approx(84)
