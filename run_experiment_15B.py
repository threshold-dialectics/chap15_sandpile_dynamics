# run_experiment_15B.py
import os
import argparse
from collections import deque

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

from td_core import (
    SandpileBTW,
    calculate_raw_instantaneous_strain_p,
    calculate_energetic_slack_p,
    calculate_tolerance_sheet_p,
    calculate_derivatives_savgol,
)

# -----------------------------------------------------------------------------
# Configuration Constants
# -----------------------------------------------------------------------------
SOTC_BETA_ADAPT_STRAIN_LOW_THRESH = 0.05
SOTC_BETA_ADAPT_STRAIN_HIGH_THRESH = 0.15
SOTC_BETA_ADAPT_GREED_RATE = 0.0065
SOTC_BETA_ADAPT_REACTION_RATE = 0.015
SOTC_BETA_ADAPT_K_TH_MIN = 3
SOTC_BETA_ADAPT_K_TH_MAX = 5
AVALANCHE_REACTION_MULT = 1.2

GRID_SIZE = (30, 30)
INITIAL_BETA_LEVER_P_CONTINUOUS = 3.5
G_LEVER_P_FIXED_VALUE = 1.0

W_G_P = 0.1
W_BETA_P = 0.4
W_FCRIT_P = 0.5
C_P_SCALE = 1.0
THETA_T_SCALING_FACTOR = 0.0012

# SOTC identification parameters
G_UPPER_SOTC_BOUND = 0.05
G_LOWER_SOTC_BOUND = -0.20
MIN_SOTC_DURATION = 30

# Diagnostic calculation windows
STRAIN_AVG_WINDOW = 10
DERIVATIVE_WINDOW_LEN = 11
COUPLE_WINDOW_LEN = 30

# Large avalanche threshold
LARGE_AVALANCHE_THRESH = GRID_SIZE[0] * GRID_SIZE[1] * 0.03

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def identify_sustained_sotc_periods(
    df_log: pd.DataFrame, g_upper_bound: float, g_lower_bound: float, min_duration: int
) -> pd.DataFrame:
    """Identify sustained SOTC periods where G stays within bounds for a minimum duration."""
    in_bounds = (df_log["safety_margin_g"] < g_upper_bound) & (
        df_log["safety_margin_g"] > g_lower_bound
    )
    group_ids = (in_bounds != in_bounds.shift()).cumsum()
    sustained = pd.Series(False, index=df_log.index)
    for _, group in df_log.groupby(group_ids):
        if in_bounds.loc[group.index].all() and len(group) >= min_duration:
            sustained.loc[group.index] = True
    df_log["is_sustained_sotc_period"] = sustained
    return df_log


# -----------------------------------------------------------------------------
# Main Simulation Routine
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run Experiment 15B SOTC simulation")
    parser.add_argument("--experiment_name", default="Experiment_15B", help="Base experiment name")
    parser.add_argument("--run_id", default="0", help="Run identifier / seed")
    parser.add_argument("--tuning", action="store_true", help="Use shorter tuning run")
    parser.add_argument(
        "--parent_dir",
        default=".",
        help="Parent directory to store results subfolder",
    )
    args = parser.parse_args()

    is_tuning = args.tuning
    experiment_name = f"{args.experiment_name}_Run{args.run_id}{'_tuning' if is_tuning else ''}"
    results_dir = os.path.join(args.parent_dir, f"{experiment_name}_results")
    os.makedirs(results_dir, exist_ok=True)

    try:
        seed_val = int(args.run_id)
    except ValueError:
        seed_val = 0
    np.random.seed(seed_val)

    if is_tuning:
        burn_in_steps = 1000
        num_sim_steps = 20000
    else:
        burn_in_steps = 2000
        num_sim_steps = 50000
    grain_add_interval = 2

    sandpile_model = SandpileBTW(
        grid_size=GRID_SIZE,
        k_th=np.clip(int(round(INITIAL_BETA_LEVER_P_CONTINUOUS)), SOTC_BETA_ADAPT_K_TH_MIN, SOTC_BETA_ADAPT_K_TH_MAX),
        p_topple=G_LEVER_P_FIXED_VALUE,
    )

    beta_lever_p_continuous = INITIAL_BETA_LEVER_P_CONTINUOUS
    unstable_history_for_strain_calc = deque(maxlen=STRAIN_AVG_WINDOW)
    simulation_log_list = []

    # Log initial state (time_step=0)
    k_th_for_log = sandpile_model.k_th
    f_crit_init = calculate_energetic_slack_p(sandpile_model.grid, k_th_for_log)
    raw_strain_init = calculate_raw_instantaneous_strain_p(sandpile_model.grid, k_th_for_log)
    unstable_history_for_strain_calc.append(raw_strain_init)
    avg_delta_p = np.mean(list(unstable_history_for_strain_calc)) if unstable_history_for_strain_calc else 0.0
    theta_t_init = calculate_tolerance_sheet_p(
        G_LEVER_P_FIXED_VALUE,
        beta_lever_p_continuous,
        f_crit_init,
        W_G_P,
        W_BETA_P,
        W_FCRIT_P,
        C_P_SCALE,
        scaling=THETA_T_SCALING_FACTOR,
    )
    simulation_log_list.append(
        {
            "time_step": 0,
            "g_lever_p_topple_prob": G_LEVER_P_FIXED_VALUE,
            "beta_lever_p": beta_lever_p_continuous,
            "actual_k_th": k_th_for_log,
            "f_crit_p": f_crit_init,
            "avg_delta_P_p": avg_delta_p,
            "theta_T_p": theta_t_init,
            "avalanche_size": 0,
            "is_large_avalanche": False,
            "theta_T_breach": avg_delta_p > theta_t_init,
            "safety_margin_g": theta_t_init - avg_delta_p,
            "speed_p": np.nan,
            "couple_p": np.nan,
            "dot_fcrit_p": np.nan,
            "dot_beta_p": np.nan,
            "dot_g_p": 0.0,
            "num_unstable_pre_relax": np.sum(sandpile_model.grid >= k_th_for_log),
            "cost_beta_change": 0.0,
            "cost_g_maintenance": 0.0,
        }
    )

    # ------------------------------------------------------------------
    # Burn-in phase (no logging of intermediate steps)
    # ------------------------------------------------------------------
    print(f"Starting burn-in for {burn_in_steps} steps ({experiment_name})...")
    for i in range(burn_in_steps):
        current_k_th = np.clip(int(round(beta_lever_p_continuous)), SOTC_BETA_ADAPT_K_TH_MIN, SOTC_BETA_ADAPT_K_TH_MAX)
        sandpile_model.k_th = current_k_th
        if (i + 1) % grain_add_interval == 0:
            sandpile_model.add_grain()
        if (i + 1) % 1 == 0:
            raw_strain = calculate_raw_instantaneous_strain_p(sandpile_model.grid, current_k_th)
            if raw_strain < SOTC_BETA_ADAPT_STRAIN_LOW_THRESH:
                beta_lever_p_continuous += SOTC_BETA_ADAPT_GREED_RATE
            elif raw_strain > SOTC_BETA_ADAPT_STRAIN_HIGH_THRESH:
                beta_lever_p_continuous -= SOTC_BETA_ADAPT_REACTION_RATE
            beta_lever_p_continuous = np.clip(
                beta_lever_p_continuous,
                SOTC_BETA_ADAPT_K_TH_MIN - 0.49,
                SOTC_BETA_ADAPT_K_TH_MAX + 0.49,
            )
        sandpile_model.topple_and_relax()
        if burn_in_steps >= 5 and (i + 1) % (burn_in_steps // 5) == 0:
            print(f"  Burn-in step {i+1}/{burn_in_steps}")

    print("Burn-in complete. Starting main simulation.")

    main_sim_start_time = 0
    for t_idx in range(1, num_sim_steps + 1):
        current_sim_time = main_sim_start_time + t_idx
        actual_k_th = np.clip(int(round(beta_lever_p_continuous)), SOTC_BETA_ADAPT_K_TH_MIN, SOTC_BETA_ADAPT_K_TH_MAX)
        sandpile_model.k_th = actual_k_th
        avalanche_size = 0
        if t_idx % grain_add_interval == 0:
            sandpile_model.add_grain()

        raw_strain = calculate_raw_instantaneous_strain_p(sandpile_model.grid, actual_k_th)
        unstable_history_for_strain_calc.append(raw_strain)
        avg_delta_p = np.mean(list(unstable_history_for_strain_calc)) if unstable_history_for_strain_calc else 0.0
        num_unstable = np.sum(sandpile_model.grid >= actual_k_th)

        if t_idx % 1 == 0:
            reacted = False
            if simulation_log_list and simulation_log_list[-1]["is_large_avalanche"]:
                beta_lever_p_continuous -= SOTC_BETA_ADAPT_REACTION_RATE * AVALANCHE_REACTION_MULT
                reacted = True
            if not reacted:
                if avg_delta_p < SOTC_BETA_ADAPT_STRAIN_LOW_THRESH:
                    beta_lever_p_continuous += SOTC_BETA_ADAPT_GREED_RATE
                elif avg_delta_p > SOTC_BETA_ADAPT_STRAIN_HIGH_THRESH:
                    beta_lever_p_continuous -= SOTC_BETA_ADAPT_REACTION_RATE
            beta_lever_p_continuous = np.clip(
                beta_lever_p_continuous,
                SOTC_BETA_ADAPT_K_TH_MIN - 0.49,
                SOTC_BETA_ADAPT_K_TH_MAX + 0.49,
            )
            actual_k_th = np.clip(int(round(beta_lever_p_continuous)), SOTC_BETA_ADAPT_K_TH_MIN, SOTC_BETA_ADAPT_K_TH_MAX)
            sandpile_model.k_th = actual_k_th

        if num_unstable > 0:
            avalanche_size = sandpile_model.topple_and_relax()
        is_large = avalanche_size > LARGE_AVALANCHE_THRESH

        f_crit_val = calculate_energetic_slack_p(sandpile_model.grid, actual_k_th)
        current_step = {
            "time_step": current_sim_time,
            "g_lever_p_topple_prob": G_LEVER_P_FIXED_VALUE,
            "beta_lever_p": beta_lever_p_continuous,
            "actual_k_th": actual_k_th,
            "f_crit_p": f_crit_val,
            "avg_delta_P_p": avg_delta_p,
            "avalanche_size": avalanche_size,
            "num_unstable_pre_relax": num_unstable,
            "cost_beta_change": 0.0,
            "cost_g_maintenance": 0.0,
            "is_large_avalanche": is_large,
        }
        current_step["theta_T_p"] = calculate_tolerance_sheet_p(
            current_step["g_lever_p_topple_prob"],
            current_step["beta_lever_p"],
            current_step["f_crit_p"],
            W_G_P,
            W_BETA_P,
            W_FCRIT_P,
            C_P_SCALE,
            scaling=THETA_T_SCALING_FACTOR,
        )
        current_step["theta_T_breach"] = current_step["avg_delta_P_p"] > current_step["theta_T_p"]
        current_step["safety_margin_g"] = current_step["theta_T_p"] - current_step["avg_delta_P_p"]

        if len(simulation_log_list) + 1 >= DERIVATIVE_WINDOW_LEN:
            beta_hist = np.array([log["beta_lever_p"] for log in simulation_log_list[-(DERIVATIVE_WINDOW_LEN - 1):]] + [current_step["beta_lever_p"]])
            fcrit_hist = np.array([log["f_crit_p"] for log in simulation_log_list[-(DERIVATIVE_WINDOW_LEN - 1):]] + [current_step["f_crit_p"]])
            dot_beta_full = calculate_derivatives_savgol(beta_hist, window_length=DERIVATIVE_WINDOW_LEN)
            dot_fcrit_full = calculate_derivatives_savgol(fcrit_hist, window_length=DERIVATIVE_WINDOW_LEN)
            current_step["dot_beta_p"] = dot_beta_full[-1] if not np.all(np.isnan(dot_beta_full)) else np.nan
            current_step["dot_fcrit_p"] = dot_fcrit_full[-1] if not np.all(np.isnan(dot_fcrit_full)) else np.nan
            current_step["dot_g_p"] = 0.0
            if not np.isnan(current_step["dot_beta_p"]) and not np.isnan(current_step["dot_fcrit_p"]):
                current_step["speed_p"] = np.sqrt(current_step["dot_beta_p"] ** 2 + current_step["dot_fcrit_p"] ** 2)
            else:
                current_step["speed_p"] = np.nan
            if len(simulation_log_list) >= COUPLE_WINDOW_LEN - 1:
                beta_seg = [log.get("dot_beta_p", np.nan) for log in simulation_log_list[-(COUPLE_WINDOW_LEN - 1):]] + [current_step["dot_beta_p"]]
                fcrit_seg = [log.get("dot_fcrit_p", np.nan) for log in simulation_log_list[-(COUPLE_WINDOW_LEN - 1):]] + [current_step["dot_fcrit_p"]]
                mask = ~np.isnan(beta_seg) & ~np.isnan(fcrit_seg)
                beta_seg = np.array(beta_seg)[mask]
                fcrit_seg = np.array(fcrit_seg)[mask]
                if len(beta_seg) >= COUPLE_WINDOW_LEN * 0.8:
                    if np.var(beta_seg) > 1e-9 and np.var(fcrit_seg) > 1e-9:
                        current_step["couple_p"] = pearsonr(beta_seg, fcrit_seg)[0]
                    else:
                        current_step["couple_p"] = 0.0
                else:
                    current_step["couple_p"] = np.nan
            else:
                current_step["couple_p"] = np.nan
        else:
            current_step["dot_beta_p"] = np.nan
            current_step["dot_fcrit_p"] = np.nan
            current_step["dot_g_p"] = 0.0
            current_step["speed_p"] = np.nan
            current_step["couple_p"] = np.nan

        simulation_log_list.append(current_step)
        if t_idx % max(1, num_sim_steps // 100) == 0:
            print(
                f"Step {t_idx}/{num_sim_steps}: G_p={current_step['safety_margin_g']:.3f}, "
                f"AvgStrain={avg_delta_p:.3f}, k_th={actual_k_th}, beta_cont={beta_lever_p_continuous:.3f}"
            )

    # ------------------------------------------------------------------
    # Finalise dataframe and compute additional fields
    # ------------------------------------------------------------------
    df_log = pd.DataFrame(simulation_log_list)
    df_log["dot_beta_p"] = pd.to_numeric(df_log["dot_beta_p"], errors="coerce")
    df_log["dot_fcrit_p"] = pd.to_numeric(df_log["dot_fcrit_p"], errors="coerce")

    df_log["Q1_both_pos"] = (df_log["dot_beta_p"] > 0) & (df_log["dot_fcrit_p"] > 0)
    df_log["Q2_beta_neg_fcrit_pos"] = (df_log["dot_beta_p"] < 0) & (df_log["dot_fcrit_p"] > 0)
    df_log["Q3_both_neg"] = (df_log["dot_beta_p"] < 0) & (df_log["dot_fcrit_p"] < 0)
    df_log["Q4_beta_pos_fcrit_neg"] = (df_log["dot_beta_p"] > 0) & (df_log["dot_fcrit_p"] < 0)

    df_log = identify_sustained_sotc_periods(df_log, G_UPPER_SOTC_BOUND, G_LOWER_SOTC_BOUND, MIN_SOTC_DURATION)
    df_log.loc[df_log["time_step"] <= burn_in_steps, "is_sustained_sotc_period"] = False

    df_log.to_csv(os.path.join(results_dir, f"sotc_simulation_log_{experiment_name}.csv"), index=False)

    sotc_frac = df_log.loc[df_log["time_step"] > burn_in_steps, "is_sustained_sotc_period"].mean()
    print(
        f"H1: Fraction of post-burn-in time in sustained SOTC state (bounds {G_LOWER_SOTC_BOUND} to {G_UPPER_SOTC_BOUND}, min {MIN_SOTC_DURATION} steps): {sotc_frac:.2%}"
    )

    # ------------------------------------------------------------------
    # Basic Plots
    # ------------------------------------------------------------------
    plt.figure(figsize=(15, 6))
    plt.plot(df_log["time_step"], df_log["safety_margin_g"], label="Safety Margin G_p", linewidth=0.8)
    plt.axhline(G_UPPER_SOTC_BOUND, color="r", linestyle="--", label="Upper Bound")
    plt.axhline(G_LOWER_SOTC_BOUND, color="r", linestyle="--")
    if df_log["is_sustained_sotc_period"].any():
        plt.fill_between(
            df_log["time_step"],
            df_log["safety_margin_g"].min(),
            df_log["safety_margin_g"].max(),
            where=df_log["is_sustained_sotc_period"],
            color="lightcoral",
            alpha=0.3,
            interpolate=True,
            label="SOTC Period",
        )
    plt.xlabel("Time Step")
    plt.ylabel("Safety Margin G_p")
    plt.title(f"Safety Margin and Identified SOTC Periods ({experiment_name})")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "sotc_safety_margin_G_p.png"))
    plt.close()

    avalanche_sizes = df_log.loc[df_log["avalanche_size"] > 0, "avalanche_size"]
    if not avalanche_sizes.empty:
        min_val = avalanche_sizes.min()
        max_val = avalanche_sizes.max()
        if max_val < 1:
            max_val = 1
        if min_val < 1:
            min_val = 1
        bins = (
            np.logspace(np.log10(min_val), np.log10(max_val), 50)
            if max_val > min_val
            else np.array([min_val, min_val + 1])
        )
        counts, edges = np.histogram(avalanche_sizes, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        plt.figure(figsize=(8, 6))
        plt.scatter(centers[counts > 0], counts[counts > 0], edgecolor="k")
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Avalanche Size")
        plt.ylabel("Frequency")
        plt.title(f"Avalanche Size Distribution (Log-Log) - {experiment_name}")
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, "sotc_avalanche_distribution_loglog.png"))
        plt.close()

    # Full diagnostics 7-panel plot
    fig, axs = plt.subplots(7, 1, figsize=(15, 24), sharex=True)
    axs[0].plot(df_log["time_step"], df_log["avg_delta_P_p"], label="Systemic Strain", linewidth=1.0)
    axs[0].plot(df_log["time_step"], df_log["theta_T_p"], label="Tolerance Sheet", linestyle="--", linewidth=1.0)
    ax0_twin = axs[0].twinx()
    ax0_twin.plot(df_log["time_step"], df_log["g_lever_p_topple_prob"], color="green", linestyle=":", label="g_lever_p")
    axs[0].set_ylabel("Strain / Tolerance")
    lines1, labels1 = axs[0].get_legend_handles_labels()
    lines2, labels2 = ax0_twin.get_legend_handles_labels()
    axs[0].legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    axs[1].plot(df_log["time_step"], df_log["safety_margin_g"], label="G_p", color="darkgreen")
    axs[1].axhline(G_UPPER_SOTC_BOUND, color="red", linestyle=":")
    axs[1].axhline(G_LOWER_SOTC_BOUND, color="red", linestyle=":")
    axs[1].fill_between(
        df_log["time_step"],
        axs[1].get_ylim()[0],
        axs[1].get_ylim()[1],
        where=df_log["is_sustained_sotc_period"],
        color="lightcoral",
        alpha=0.2,
        interpolate=True,
        label="SOTC Period",
    )
    axs[1].set_ylabel("G_p")
    axs[1].legend(loc="upper left")

    axs[2].plot(df_log["time_step"], df_log["f_crit_p"], label="Fcrit_p", color="teal")
    axs[2].set_ylabel("Fcrit_p")
    axs[2].legend(loc="upper left")

    axs[3].plot(df_log["time_step"], df_log["beta_lever_p"], label="Beta Lever", color="blueviolet")
    axs[3].plot(df_log["time_step"], df_log["actual_k_th"], label="Actual k_th", linestyle="--", color="indigo")
    axs[3].set_ylabel("Beta / k_th")
    axs[3].legend(loc="upper left")

    axs[4].plot(df_log["time_step"], df_log["speed_p"], label="Speed Index", color="crimson")
    axs[4].set_ylabel("Speed_p")
    axs[4].legend(loc="upper left")

    axs[5].plot(df_log["time_step"], df_log["couple_p"], label="Couple Index", color="purple")
    axs[5].set_ylabel("Couple_p")
    axs[5].set_ylim(-1.1, 1.1)
    axs[5].legend(loc="upper left")

    axs[6].plot(df_log["time_step"], df_log["avalanche_size"], label="Avalanche Size", color="forestgreen")
    axs[6].set_ylabel("Avalanche Size")
    axs[6].set_xlabel("Time Step")
    axs[6].legend(loc="upper left")

    plt.suptitle(f"Full TD Diagnostics and Sandpile Dynamics ({experiment_name})", fontsize=18, y=0.99)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(os.path.join(results_dir, f"sotc_full_diagnostics_timeseries_{experiment_name}.png"))
    plt.close()

    print(f"\nSOTC Simulation {experiment_name} complete. Results in {results_dir}")


if __name__ == "__main__":
    main()
