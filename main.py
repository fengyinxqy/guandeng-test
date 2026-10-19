"""Reproducible gateway audit and training-only routing-rule selection."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

FEATURES = [
    "request_id", "request_time", "node_status_synced_at", "service_cluster",
    "model_tier", "available_slots", "payload_kb",
]
FAILURE_COST = 120
ACTION_COST = 10
AGE_GRID = [5, 10, 15, 20, 30, 45, 60, None]
SLOT_GRID = [None, 0, 1, 2, 3, 5]


def prepare(raw, require_labels=True):
    """Reject conflicts and invalid inputs; never impute unknown outcomes."""
    required = FEATURES + (["success"] if require_labels else [])
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if raw[FEATURES].isna().any().any():
        raise ValueError("Missing routing features; explicit fallback required")
    duplicate_rows = int(raw.duplicated().sum())
    frame = raw.drop_duplicates().copy()
    if frame.request_id.duplicated().any():
        raise ValueError("Conflicting records share a request_id")
    if require_labels and not frame.success.dropna().isin([0, 1]).all():
        raise ValueError("success must be 0, 1, or missing")
    frame["request_dt"] = pd.to_datetime(frame.request_time, utc=True, format="ISO8601", errors="raise")
    frame["sync_dt"] = pd.to_datetime(frame.node_status_synced_at, utc=True, format="ISO8601", errors="raise")
    frame["status_age_minutes"] = (frame.request_dt - frame.sync_dt).dt.total_seconds() / 60
    if (frame.status_age_minutes < 0).any():
        raise ValueError("Health sync timestamp is after the routing decision")
    for name in ["available_slots", "payload_kb"]:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if not frame[name].map(math.isfinite).all() or (frame[name] < 0).any():
            raise ValueError(f"{name} must be finite and nonnegative")
    if frame.available_slots.mod(1).ne(0).any():
        raise ValueError("available_slots must be an integer")
    return frame.sort_values(["request_dt", "request_id"]).reset_index(drop=True), duplicate_rows


def split_time(frame, train_fraction=0.7):
    if not 0 < train_fraction < 1 or len(frame) < 2:
        raise ValueError("Need at least 2 requests and a train fraction between 0 and 1")
    cutoff = frame.request_dt.iloc[int(len(frame) * train_fraction)]
    train = frame[frame.request_dt < cutoff].copy()
    validation = frame[frame.request_dt >= cutoff].copy()
    if train.empty or validation.empty:
        raise ValueError("Cannot separate timestamps into two nonempty periods")
    return train, validation, cutoff


def rule_mask(frame, age_threshold, slot_threshold, mode="both"):
    age = pd.Series(False, index=frame.index)
    slots = age.copy()
    if age_threshold is not None:
        age = frame.status_age_minutes.gt(age_threshold)
    if slot_threshold is not None:
        slots = frame.available_slots.le(slot_threshold)
    if mode == "age":
        return age
    if mode == "slots":
        return slots
    if mode == "none":
        return pd.Series(False, index=frame.index)
    if mode != "both":
        raise ValueError(f"Unknown rule mode: {mode}")
    return age | slots


def evaluate(frame, alerts, rescue_rate=1.0, new_failure_rate=0.0):
    """Unknowns remain unknown; charge every alert in this cost proxy."""
    if not 0 <= rescue_rate <= 1 or not 0 <= new_failure_rate <= 1:
        raise ValueError("Scenario rates must be between 0 and 1")
    failed, succeeded, unknown = frame.success.eq(0), frame.success.eq(1), frame.success.isna()
    failures, tp = int(failed.sum()), int((alerts & failed).sum())
    fp, actions = int((alerts & succeeded).sum()), int(alerts.sum())
    uncertain_alerts = int((alerts & unknown).sum())
    baseline = FAILURE_COST * failures
    savings = FAILURE_COST * (rescue_rate * tp - new_failure_rate * fp) - ACTION_COST * actions
    known_count = int((~unknown).sum())
    return {
        "requests": len(frame), "known_outcomes": known_count,
        "unknown_outcomes": int(unknown.sum()), "failures": failures,
        "observed_failure_rate": failures / known_count if known_count else None,
        "alerts": actions, "alerted_failures": tp, "alerted_successes": fp,
        "alerted_unknowns": uncertain_alerts, "missed_failures": failures - tp,
        "coverage": actions / len(frame) if len(frame) else None,
        "precision_known": tp / (tp + fp) if tp + fp else None,
        "recall_known": tp / failures if failures else None,
        "baseline_observed_loss_yuan": baseline,
        "action_cost_proxy_yuan": ACTION_COST * actions,
        "scenario_loss_yuan": baseline - savings,
        "scenario_saving_yuan": savings,
        "break_even_rescue_rate": ACTION_COST * actions / (FAILURE_COST * tp) if tp else None,
        "unknown_as_success_saving_yuan": savings - FAILURE_COST * new_failure_rate * uncertain_alerts,
        "unknown_as_failure_saving_yuan": (
            FAILURE_COST * (rescue_rate * (tp + uncertain_alerts) - new_failure_rate * fp)
            - ACTION_COST * actions
        ),
    }


def select_policy(train):
    """Global coarse search, then select each cluster's signal using train only."""
    candidates = []
    for age in AGE_GRID:
        for slots in SLOT_GRID:
            scores = evaluate(train, rule_mask(train, age, slots))
            candidates.append({"age_threshold_minutes": age, "slot_threshold": slots, **scores})
    best = max(candidates, key=lambda r: (r["scenario_saving_yuan"], -r["alerts"]))
    age, slots = best["age_threshold_minutes"], best["slot_threshold"]
    modes = {}
    for cluster, group in train.groupby("service_cluster", sort=True):
        choices = []
        for mode in ["none", "age", "slots", "both"]:
            scores = evaluate(group, rule_mask(group, age, slots, mode))
            choices.append((scores["scenario_saving_yuan"], -scores["alerts"], mode))
        modes[cluster] = max(choices, key=lambda v: v[:2])[2]
    policy = {
        "age_threshold_minutes": age, "slot_threshold": slots,
        "cluster_modes": modes, "unseen_cluster_mode": "both",
        "rule_semantics": "age > threshold OR slots <= threshold; cluster mode selects terms",
        "status": "offline candidate; no production deployment or causal evidence",
    }
    return policy, pd.DataFrame(candidates)


def predict_flags(features, policy):
    """Uses decision-time features only, never success or future outcomes."""
    age, slots = policy["age_threshold_minutes"], policy["slot_threshold"]
    flags = rule_mask(features, age, slots, policy["unseen_cluster_mode"])
    for cluster, mode in policy["cluster_modes"].items():
        belongs = features.service_cluster.eq(cluster)
        flags.loc[belongs] = rule_mask(features.loc[belongs], age, slots, mode)
    return flags


def group_stats(frame, keys):
    work = frame.assign(failure=frame.success.eq(0).astype(int))
    result = work.groupby(keys, dropna=False, observed=True).agg(
        requests=("request_id", "size"), known_outcomes=("success", "count"),
        failures=("failure", "sum"), mean_age_minutes=("status_age_minutes", "mean"),
    ).reset_index()
    result["unknown_outcomes"] = result.requests - result.known_outcomes
    result["failure_rate_known"] = result.failures.div(result.known_outcomes.where(result.known_outcomes.ne(0)))
    return result


def run(input_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(input_path)
    frame, duplicates = prepare(raw)
    train, validation, cutoff = split_time(frame)
    policy, candidates = select_policy(train)
    candidates.to_csv(output_dir / "training_candidates.csv", index=False)
    age, slots = policy["age_threshold_minutes"], policy["slot_threshold"]
    split_names = {"train": train, "validation": validation, "all_descriptive": frame}
    comparisons = []
    for name, group in split_names.items():
        masks = {
            "no_action": rule_mask(group, None, None),
            "flag_all": pd.Series(True, index=group.index),
            "age_only": rule_mask(group, age, None),
            "slots_only": rule_mask(group, None, slots),
            "global_or": rule_mask(group, age, slots),
            "cluster_policy": predict_flags(group[FEATURES + ["status_age_minutes"]], policy),
        }
        for rule_name, alerts in masks.items():
            comparisons.append({"split": name, "rule": rule_name, **evaluate(group, alerts)})
    metrics = pd.DataFrame(comparisons)
    metrics.to_csv(output_dir / "policy_comparison.csv", index=False)

    frame["stale_gt_15m"] = frame.status_age_minutes.gt(15)
    frame["slots_le_3"] = frame.available_slots.le(3)
    for filename, keys in [
        ("age_groups.csv", ["stale_gt_15m"]),
        ("cluster_age_groups.csv", ["service_cluster", "stale_gt_15m"]),
        ("cluster_slot_groups.csv", ["service_cluster", "slots_le_3"]),
        ("cluster_cross_groups.csv", ["service_cluster", "stale_gt_15m", "slots_le_3"]),
        ("model_age_groups.csv", ["model_tier", "stale_gt_15m"]),
    ]:
        group_stats(frame, keys).to_csv(output_dir / filename, index=False)
    frame["payload_band_kb"] = pd.cut(frame.payload_kb, [-1, 0, 30, 50, 70, 90, float("inf")])
    group_stats(frame, ["payload_band_kb"]).to_csv(output_dir / "payload_groups.csv", index=False)

    flags = predict_flags(frame, policy)
    decisions = frame[FEATURES + ["success", "status_age_minutes"]].copy()
    decisions["split"] = frame.request_dt.ge(cutoff).map({True: "validation", False: "train"})
    decisions["flag_target_unavailable"] = flags
    decisions["rule_mode"] = frame.service_cluster.map(policy["cluster_modes"]).fillna(policy["unseen_cluster_mode"])
    decisions.to_csv(output_dir / "request_decisions.csv", index=False)

    scenarios = []
    for split_name, group in split_names.items():
        mask = predict_flags(group, policy)
        for q in [0, 0.25, 0.5, 0.75, 1.0]:
            for r in [0, 0.01, 0.05]:
                scenarios.append({"split": split_name, "rescue_rate": q, "new_failure_rate": r, **evaluate(group, mask, q, r)})
    pd.DataFrame(scenarios).to_csv(output_dir / "sensitivity.csv", index=False)

    policy["selected_using"] = "earlier 70% timestamp partition only"
    policy["training_cutoff_utc"] = cutoff.isoformat()
    (output_dir / "policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit = {
        "source": input_path.name, "sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "raw_rows": len(raw), "exact_duplicate_rows_removed": duplicates,
        "unique_requests": len(frame), "known_outcomes": int(frame.success.notna().sum()),
        "successes": int(frame.success.eq(1).sum()), "failures": int(frame.success.eq(0).sum()),
        "unknown_outcomes": int(frame.success.isna().sum()),
        "zero_payload_rows_retained": int(frame.payload_kb.eq(0).sum()),
        "zero_slot_rows_retained": int(frame.available_slots.eq(0).sum()),
        "request_time_min_utc": frame.request_dt.min().isoformat(),
        "request_time_max_utc": frame.request_dt.max().isoformat(),
        "split_cutoff_utc": cutoff.isoformat(), "train_requests": len(train),
        "validation_requests": len(validation),
        "notes": [
            "Timezone-normalized UTC arithmetic. Conflicting IDs and invalid inputs fail explicitly.",
            "Missing success is unknown, not success/failure; separate sensitivity endpoint provided.",
            "10 yuan per alerted request is a proxy; no node_id to count unique unavailable events.",
            "Scenario q=1,r=0 is optimistic; reroute capacity/outcomes and waiting duration are unobserved.",
            "Full-data EDA preceded design; temporal replay is not an untouched confirmatory test.",
        ],
    }
    (output_dir / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"audit": audit, "policy": policy}, ensure_ascii=True, indent=2))
    print(metrics[["split", "rule", "alerts", "alerted_failures", "alerted_successes", "alerted_unknowns", "scenario_saving_yuan"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/gateway_requests.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    run(args.input, args.output_dir)
