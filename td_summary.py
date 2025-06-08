# td_summary.py
from __future__ import annotations
import numpy as np
from typing import Dict
import json


def make_summary(df, params: Dict, analysis_results: Dict) -> Dict:
    # --- 1. Metadata ----------------------------------------
    summary = {
        "metadata": {
            "experiment_id": params.get("EXPERIMENT_NAME"),
            "grid_size": list(params.get("GRID_SIZE", [])),
            "burn_in_steps": params.get("BURN_IN_STEPS"),
            "total_steps": params.get("NUM_SIM_STEPS"),
            "rng_seed": params.get("SEED"),
        }
    }

    # --- 2. Event counts & overlap --------------------------
    lav = df["is_large_avalanche"].astype(bool)
    br = df["theta_T_breach"].astype(bool)
    inter = int((lav & br).sum())
    union = int((lav | br).sum())
    summary["event_counts"] = {
        "n_large_avalanches": int(lav.sum()),
        "n_thetaT_breaches": int(br.sum()),
        "jaccard_index_overlap": (inter / union) if union else None,
    }
    summary["event_overlap"] = {
        "lav_and_breach": int((lav & br).sum()),
        "lav_only": int((lav & ~br).sum()),
        "breach_only": int((~lav & br).sum()),
        "neither": int((~lav & ~br).sum()),
    }

    # --- 3. Avalanche size stats ----------------------------
    aval = df["avalanche_size"].loc[df["avalanche_size"] > 0]
    if not aval.empty:
        from powerlaw import Fit
        from scipy.stats import linregress

        fit = Fit(aval.values, xmin=1, discrete=True)

        # calculate an approximate R^2 on the log-log CCDF
        sorted_sizes = np.sort(aval.values)
        ranks = np.arange(len(sorted_sizes), 0, -1)
        log_x = np.log10(sorted_sizes)
        log_y = np.log10(ranks)
        if len(sorted_sizes) > 1:
            r2 = linregress(log_x, log_y).rvalue ** 2
        else:
            r2 = np.nan

        aval_stats = {
            "n_total": int(len(aval)),
            "n_large": int(lav.sum()),
            "size_min": float(aval.min()),
            "size_median": float(aval.median()),
            "size_mean": float(aval.mean()),
            "size_max": float(aval.max()),
            "powerlaw_exponent": float(-fit.alpha),
            "powerlaw_r2": float(r2),
        }
    else:
        aval_stats = {
            "n_total": 0,
            "n_large": int(lav.sum()),
            "size_min": np.nan,
            "size_median": np.nan,
            "size_mean": np.nan,
            "size_max": np.nan,
            "powerlaw_exponent": np.nan,
            "powerlaw_r2": np.nan,
        }
    summary["avalanche_stats"] = aval_stats

    # --- 4. Strain histogram numbers ------------------------
    strain = df["avg_delta_P_p"]
    descr = strain.describe()
    summary["strain_stats"] = {
        "min": float(descr["min"]),
        "q25": float(strain.quantile(0.25)),
        "median": float(descr["50%"]),
        "mean": float(descr["mean"]),
        "q75": float(strain.quantile(0.75)),
        "max": float(descr["max"]),
        "skew": float(strain.skew()),
    }

    # --- 5. Generic helper for bar-chart metrics ------------
    def _metric_block(pre_LA, pre_B, base):
        from scipy.stats import mannwhitneyu
        out: Dict[str, float] = {}

        def s(arr):
            arr = np.asarray(arr, dtype=float)
            return float(np.mean(arr)) if len(arr) else np.nan, (
                float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
            )

        m, se = s(base)
        out["baseline_mean"], out["baseline_se"] = m, se

        if len(pre_LA):
            m, se = s(pre_LA)
            stat, p = mannwhitneyu(pre_LA, base, alternative="greater")
            out.update(pre_LA_mean=m, pre_LA_se=se, pre_LA_u=float(stat), pre_LA_p=float(p))
        else:
            out.update(pre_LA_mean=np.nan, pre_LA_se=0.0, pre_LA_u=np.nan, pre_LA_p=np.nan)

        if len(pre_B):
            m, se = s(pre_B)
            stat, p = mannwhitneyu(pre_B, base, alternative="greater")
            out.update(pre_Θ_mean=m, pre_Θ_se=se, pre_Θ_u=float(stat), pre_Θ_p=float(p))
        else:
            out.update(pre_Θ_mean=np.nan, pre_Θ_se=0.0, pre_Θ_u=np.nan, pre_Θ_p=np.nan)
        return out

    summary["summary_by_metric"] = {
        "speed_p_smooth": _metric_block(
            analysis_results.get("pre_speed_LA", []),
            analysis_results.get("pre_speed_B", []),
            analysis_results.get("baseline_speed", []),
        ),
        "couple_p": _metric_block(
            analysis_results.get("pre_couple_LA", []),
            analysis_results.get("pre_couple_B", []),
            analysis_results.get("baseline_couple", []),
        ),
        "hazard_rate_p_clipped": _metric_block(
            analysis_results.get("pre_hazard_B", []),
            analysis_results.get("pre_hazard_B", []),
            analysis_results.get("baseline_hazard", []),
        ),
    }

    # --- Derivative sign patterns at strong coupling --------
    summary["derivative_sign_patterns"] = {
        "high_positive_couple": {
            "N": int(analysis_results.get("high_pos_signs", {}).get("N", 0)),
            "both_pos": int(analysis_results.get("high_pos_signs", {}).get("both_pos", 0)),
            "beta_neg": int(analysis_results.get("high_pos_signs", {}).get("beta_neg", 0)),
            "both_neg": int(analysis_results.get("high_pos_signs", {}).get("both_neg", 0)),
            "tightening": int(analysis_results.get("high_pos_signs", {}).get("tightening", 0)),
        },
        "high_negative_couple": {
            "N": int(analysis_results.get("high_neg_signs", {}).get("N", 0)),
            "q1": int(analysis_results.get("high_neg_signs", {}).get("q1", 0)),
            "q2": int(analysis_results.get("high_neg_signs", {}).get("q2", 0)),
            "q3": int(analysis_results.get("high_neg_signs", {}).get("q3", 0)),
            "q4": int(analysis_results.get("high_neg_signs", {}).get("q4", 0)),
        },
    }

    # --- 6. Quadrant counts ---------------------------------
    summary["quadrant_counts"] = {
        "pre_event": np.asarray(analysis_results.get("quadrant_pre", []), dtype=float).tolist(),
        "baseline": np.asarray(analysis_results.get("quadrant_base", []), dtype=float).tolist(),
        "chi2": float(analysis_results.get("quadrant_chi2", np.nan)),
        "p": float(analysis_results.get("quadrant_p", np.nan)),
    }

    # --- 7. Tightening loop ---------------------------------
    summary["tightening_loop"] = {
        "baseline_prop": float(analysis_results.get("tight_base", np.nan)),
        "pre_LA_prop": float(analysis_results.get("tight_LA", np.nan)),
        "pre_Θ_prop": float(analysis_results.get("tight_B", np.nan)),
        "pvals": {
            "pre_LA_vs_base": float(analysis_results.get("tight_p_LA", np.nan)),
            "pre_Θ_vs_base": float(analysis_results.get("tight_p_B", np.nan)),
        },
    }

    summary["mean_safety_margin"] = float(analysis_results.get("mean_safety_margin", np.nan))
    summary["mean_beta_lever_p"] = float(analysis_results.get("mean_beta_lever_p", np.nan))
    summary["mean_g_lever_p"] = float(analysis_results.get("mean_g_lever_p", np.nan))

    summary["fcrit_slack"] = {
        "baseline_mean": float(analysis_results.get("fcrit_baseline_mean", np.nan)),
        "baseline_sd": float(analysis_results.get("fcrit_baseline_sd", np.nan)),
        "post_burnin_trend_slope": float(analysis_results.get("fcrit_trend_slope", np.nan)),
        "final_value": float(analysis_results.get("fcrit_final_value", np.nan)),
    }

    summary["cumulative_cost"] = {
        "final_beta_cost": float(analysis_results.get("final_beta_cost", np.nan)),
        "final_g_cost": float(analysis_results.get("final_g_cost", np.nan)),
        "final_total_cost": float(analysis_results.get("final_total_cost", np.nan)),
    }

    summary["trend_slopes"] = {
        "avg_delta_P_p": float(analysis_results.get("strain_trend_slope", np.nan)),
        "theta_T_p": float(analysis_results.get("theta_trend_slope", np.nan)),
    }
    return summary
