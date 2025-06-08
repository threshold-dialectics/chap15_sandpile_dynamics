import json
import os
import subprocess
import sys
import math


def test_summary_generation(tmp_path):
    env = os.environ.copy()
    env["NUM_SIM_STEPS"] = "200"
    env["BURN_IN_STEPS"] = "10"
    # run the simulation
    subprocess.run([sys.executable, "run_experiment_15A.py"], check=True, env=env)
    summary_file = os.path.join("Experiment_15A_results", "summary.json")
    assert os.path.exists(summary_file)
    with open(summary_file, "r") as f:
        data = json.load(f)

    # Check top-level keys
    for key in [
        "metadata",
        "event_counts",
        "event_overlap",
        "avalanche_stats",
        "strain_stats",
        "summary_by_metric",
        "quadrant_counts",
        "tightening_loop",
        "derivative_sign_patterns",
        "mean_safety_margin",
        "mean_beta_lever_p",
        "mean_g_lever_p",
        "fcrit_slack",
        "cumulative_cost",
        "trend_slopes",
    ]:
        assert key in data
        assert data[key] is not None

    # simple numeric checks
    assert data["metadata"]["total_steps"] == 200
    assert not math.isnan(data["event_counts"]["n_large_avalanches"])
