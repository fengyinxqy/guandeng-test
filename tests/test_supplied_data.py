"""Integration reconciliation uses the actual supplied CSV, not generated evidence."""
import csv
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from main import run


def test_supplied_csv_end_to_end_and_independent_reconciliation(tmp_path):
    source = Path(__file__).resolve().parents[1] / "data" / "gateway_requests.csv"
    run(source, tmp_path)
    audit = json.loads((tmp_path / "audit.json").read_text(encoding="utf-8"))
    assert (audit["raw_rows"], audit["exact_duplicate_rows_removed"], audit["unique_requests"]) == (1078, 51, 1027)
    assert (audit["successes"], audit["failures"], audit["unknown_outcomes"]) == (850, 126, 51)
    # Independently calculate validation decisions with stdlib CSV and datetime.
    with source.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    unique = {tuple(row.values()): row for row in records}
    cutoff = datetime.fromisoformat(audit["split_cutoff_utc"])
    counts = {"requests": 0, "alerts": 0, "failures": 0, "tp": 0, "fp": 0, "unknown": 0}
    north = "\u534e\u5317\u5f02\u6784\u96c6\u7fa4"
    for row in unique.values():
        requested = datetime.fromisoformat(row["request_time"])
        if requested < cutoff:
            continue
        age = (requested - datetime.fromisoformat(row["node_status_synced_at"])).total_seconds() / 60
        flagged = int(row["available_slots"]) <= 3 if row["service_cluster"] == north else age > 15
        counts["requests"] += 1
        counts["alerts"] += flagged
        counts["failures"] += row["success"] == "0"
        counts["tp"] += flagged and row["success"] == "0"
        counts["fp"] += flagged and row["success"] == "1"
        counts["unknown"] += flagged and row["success"] == ""
    assert counts == {"requests": 310, "alerts": 79, "failures": 36, "tp": 34, "fp": 41, "unknown": 4}
    table = pd.read_csv(tmp_path / "policy_comparison.csv")
    result = table[table.split.eq("validation") & table.rule.eq("cluster_policy")].iloc[0]
    assert result.alerts == counts["alerts"]
    assert result.alerted_failures == counts["tp"]
    assert result.scenario_saving_yuan == 120 * counts["tp"] - 10 * counts["alerts"] == 3290
