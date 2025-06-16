# run_experiment_15A.py
import os
import json
from collections import deque

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, mannwhitneyu, chi2_contingency, linregress
import matplotlib.pyplot as plt

# Global plot style settings
plt.rcParams.update({
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

SAVEFIG_DPI = 350

from td_core import (
    SandpileBTW,
    calculate_raw_instantaneous_strain_p,
    calculate_energetic_slack_p,
    calculate_tolerance_sheet_p,
    calculate_derivatives_savgol,
)

# --- Experimental Configuration Parameters ---
EXPERIMENT_NAME = "Experiment_15A"
RESULTS_DIR = f"{EXPERIMENT_NAME}_results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Lever adaptation flags
ADAPTIVE_BETA_LEVER = True  # Adaptive beta lever
ADAPTIVE_G_LEVER = False    # g lever fixed for Experiment 15A

# Initial value and adaptation parameters for perception gain (g_lever_p)
G_LEVER_P_INITIAL = 1.0
G_LEVER_ADAPT_FREQUENCY = 1
AVG_DELTA_P_LOWER_THRESHOLD_FOR_G_ADAPT = 0.15
AVG_DELTA_P_UPPER_THRESHOLD_FOR_G_ADAPT = 0.6
G_LEVER_ADAPT_RATE = 0.05
G_LEVER_MIN_VAL = 0.1
G_LEVER_MAX_VAL = 1.0

# Costs for adapting/maintaining levers
BETA_LEVER_COST_PER_CHANGE_UNIT = 0.0
G_LEVER_PHI1_COST_EXPONENT = 0.75
G_LEVER_COST_COEFF = 0.0


# --- Main Simulation Parameters & Loop ---
np.random.seed(42)

GRID_SIZE = (30, 30)
INITIAL_K_TH = 4
G_LEVER_P_FIXED = 1.0 

W_G_P = 0.1
W_BETA_P = 0.2
W_FCRIT_P = 0.7
C_P_SCALE = 1.0
# Adjusted THETA_T_SCALING_FACTOR as per Recommendation 2.1
# This is an initial estimate and will likely need further tuning.
THETA_T_SCALING_FACTOR = 0.0025 # Old value was 0.005

SPEED_SMOOTHING_WINDOW_ANALYSIS = 11 # For post-simulation analysis smoothing

BURN_IN_STEPS = int(os.environ.get("BURN_IN_STEPS", 1000))
POST_BURN_IN_STABILIZATION_STEPS = 500
NUM_SIM_STEPS = int(os.environ.get("NUM_SIM_STEPS", 20000))
STRAIN_AVG_WINDOW = 20 # Now used for time-averaging systemic strain
LARGE_AVALANCHE_THRESH = GRID_SIZE[0] * GRID_SIZE[1] * 0.02 # Example: 2% of grid size
# DERIVATIVE_WINDOW_LEN and COUPLE_WINDOW_LEN might need review after observing new dynamics.
DERIVATIVE_WINDOW_LEN = 11
COUPLE_WINDOW_LEN = 25

simulation_log_list = []
# Initialized deque for strain history as per Recommendation 1.2
unstable_history_for_strain_calc = deque(maxlen=STRAIN_AVG_WINDOW) 

# Initialize continuous beta_lever_p as per Recommendation 3.1
beta_lever_p_continuous = float(INITIAL_K_TH)

# Initialize perception gain (g_lever) value
current_p_topple = G_LEVER_P_INITIAL

# Initialize sandpile model with the initial k_th and p_topple
sandpile_model = SandpileBTW(
    grid_size=GRID_SIZE,
    k_th=int(round(beta_lever_p_continuous)),
    p_topple=current_p_topple,
)

# Initial state (t=0)
k_th_for_init_step = int(round(beta_lever_p_continuous))
f_crit_p_val_init = calculate_energetic_slack_p(sandpile_model.grid, k_th_for_init_step)

# Initial raw strain calculation for avg_delta_P_p initialization
raw_strain_init = calculate_raw_instantaneous_strain_p(sandpile_model.grid, k_th_for_init_step)
unstable_history_for_strain_calc.append(raw_strain_init)
avg_delta_P_p_val_init = np.mean(unstable_history_for_strain_calc) if unstable_history_for_strain_calc else 0.0

num_unstable_init = np.sum(sandpile_model.grid >= k_th_for_init_step)

theta_T_p_val_init = calculate_tolerance_sheet_p(
    current_p_topple,
    beta_lever_p_continuous,  # Use continuous beta for ThetaT
    f_crit_p_val_init,
    W_G_P, W_BETA_P, W_FCRIT_P, C_P_SCALE,
    scaling=THETA_T_SCALING_FACTOR)

simulation_log_list.append({
    'time_step': 0,
    'g_lever_p_topple_prob': current_p_topple,
    'beta_lever_p': beta_lever_p_continuous, # Log continuous beta
    'actual_k_th': k_th_for_init_step, # Log the k_th used by sandpile
    'f_crit_p': f_crit_p_val_init,
    'avg_delta_P_p': avg_delta_P_p_val_init,
    'theta_T_p': theta_T_p_val_init,
    'avalanche_size': 0,
    'is_large_avalanche': False,
    'theta_T_breach': avg_delta_P_p_val_init > theta_T_p_val_init,
    'speed_p': np.nan,
    'couple_p': np.nan,
    'dot_fcrit_p': np.nan,
    'dot_beta_p': np.nan,
    'dot_g_p': np.nan,
    'num_unstable_pre_relax': num_unstable_init,
    'cost_beta_change': 0.0,
    'cost_g_maintenance': 0.0
})

print(f"Starting burn-in for {BURN_IN_STEPS} steps...")
for i in range(BURN_IN_STEPS):
    sandpile_model.k_th = int(round(beta_lever_p_continuous)) # Ensure sandpile uses current k_th
    sandpile_model.add_grain()
    sandpile_model.topple_and_relax() # Uses self.k_th
    if (i+1) % (BURN_IN_STEPS//5) == 0:
        print(f"Burn-in step {i+1}/{BURN_IN_STEPS}")
print("Burn-in complete. Starting main simulation.")
main_sim_start_time = sandpile_model.time_step_counter

# K_th adaptation thresholds (initial placeholders, need tuning)
# Recommendation 3.2: These values need tuning based on the new scale of avg_delta_P_p.
AVG_DELTA_P_LOWER_THRESHOLD_FOR_K_TH_ADAPT = 0.2
AVG_DELTA_P_UPPER_THRESHOLD_FOR_K_TH_ADAPT = 0.8
K_TH_ADAPT_FREQUENCY = 1 

is_large_avalanche_flag = False      # ← no avalanche yet

for t_idx in range(1, NUM_SIM_STEPS + 1):
    current_sim_time = main_sim_start_time + t_idx

    # Ensure sandpile model uses current lever values
    sandpile_model.k_th = np.clip(int(round(beta_lever_p_continuous)), 3, 6)
    k_th_for_current_step = sandpile_model.k_th
    sandpile_model.p_topple = current_p_topple

    sandpile_model.add_grain()

    raw_strain_this_step = calculate_raw_instantaneous_strain_p(
        sandpile_model.grid,
        k_th_for_current_step,
    )
    unstable_history_for_strain_calc.append(raw_strain_this_step)
    avg_delta_P_p_val_for_step = (
        np.mean(unstable_history_for_strain_calc) if unstable_history_for_strain_calc else 0.0
    )

    # Adapt beta lever if enabled
    if ADAPTIVE_BETA_LEVER and t_idx % K_TH_ADAPT_FREQUENCY == 0:
        if avg_delta_P_p_val_for_step < AVG_DELTA_P_LOWER_THRESHOLD_FOR_K_TH_ADAPT \
           and sandpile_model.grid.mean() > k_th_for_current_step * 0.5:
            beta_lever_p_continuous = min(6.0, beta_lever_p_continuous + 0.1)

        elif avg_delta_P_p_val_for_step > AVG_DELTA_P_UPPER_THRESHOLD_FOR_K_TH_ADAPT \
             or is_large_avalanche_flag:         # ← safe to reference now
            beta_lever_p_continuous = max(3.0, beta_lever_p_continuous - 0.1)

        sandpile_model.k_th = np.clip(int(round(beta_lever_p_continuous)), 3, 6)
        k_th_for_current_step = sandpile_model.k_th

    # Adapt g lever if enabled
    if ADAPTIVE_G_LEVER and t_idx % G_LEVER_ADAPT_FREQUENCY == 0:
        if avg_delta_P_p_val_for_step < AVG_DELTA_P_LOWER_THRESHOLD_FOR_G_ADAPT:
            current_p_topple = max(G_LEVER_MIN_VAL, current_p_topple - G_LEVER_ADAPT_RATE)
        elif avg_delta_P_p_val_for_step > AVG_DELTA_P_UPPER_THRESHOLD_FOR_G_ADAPT:
            current_p_topple = min(G_LEVER_MAX_VAL, current_p_topple + G_LEVER_ADAPT_RATE)
        sandpile_model.p_topple = current_p_topple

    num_unstable_pre_relax_val = np.sum(sandpile_model.grid >= k_th_for_current_step)

    avalanche_size_current_event = sandpile_model.topple_and_relax()
    is_large_avalanche_flag = avalanche_size_current_event > LARGE_AVALANCHE_THRESH

    f_crit_p_val_for_step = calculate_energetic_slack_p(sandpile_model.grid, k_th_for_current_step)
    is_large_avalanche_flag = avalanche_size_current_event > LARGE_AVALANCHE_THRESH

    # Lever adaptation costs
    cost_beta_change_this_step = 0.0
    if ADAPTIVE_BETA_LEVER and BETA_LEVER_COST_PER_CHANGE_UNIT > 0 and simulation_log_list:
        prev_k_th = simulation_log_list[-1]['actual_k_th']
        delta_k = abs(k_th_for_current_step - prev_k_th)
        if delta_k > 0:
            cost_beta_change_this_step = BETA_LEVER_COST_PER_CHANGE_UNIT * delta_k
            f_crit_p_val_for_step -= cost_beta_change_this_step

    cost_g_maintenance_this_step = 0.0
    if G_LEVER_COST_COEFF > 0:
        cost_g_maintenance_this_step = G_LEVER_COST_COEFF * (current_p_topple ** G_LEVER_PHI1_COST_EXPONENT)
        f_crit_p_val_for_step -= cost_g_maintenance_this_step

    f_crit_p_val_for_step = max(0.0, f_crit_p_val_for_step)
    
    current_step_metrics = {
        'time_step': current_sim_time,
        'g_lever_p_topple_prob': current_p_topple,
        'beta_lever_p': beta_lever_p_continuous,
        'actual_k_th': k_th_for_current_step,
        'f_crit_p': f_crit_p_val_for_step,
        'avg_delta_P_p': avg_delta_P_p_val_for_step,
        'avalanche_size': avalanche_size_current_event,
        'num_unstable_pre_relax': num_unstable_pre_relax_val,
        'cost_beta_change': cost_beta_change_this_step,
        'cost_g_maintenance': cost_g_maintenance_this_step,
    }

    current_step_metrics['theta_T_p'] = calculate_tolerance_sheet_p(
        current_step_metrics['g_lever_p_topple_prob'],
        current_step_metrics['beta_lever_p'], # Use continuous beta_lever_p for ThetaT
        current_step_metrics['f_crit_p'],
        W_G_P, W_BETA_P, W_FCRIT_P, C_P_SCALE,
        scaling=THETA_T_SCALING_FACTOR
    )
    current_step_metrics['is_large_avalanche'] = is_large_avalanche_flag

    # Post-relaxation raw strain for DEBUG print (should be 0 if relaxation is complete for k_th_for_current_step)
    raw_strain_debug_val_post_relax = calculate_raw_instantaneous_strain_p(sandpile_model.grid, k_th_for_current_step)
    
    #print(
    #    f"DEBUG t_idx={t_idx}, avg_delta_P_p={current_step_metrics['avg_delta_P_p']:.3f}, "
    #    f"theta_T_p={current_step_metrics['theta_T_p']:.3f}, "
    #    f"num_unstable={num_unstable_pre_relax_val}, raw_strain_post_relax={raw_strain_debug_val_post_relax:.3f}"
    #)
    
    current_step_metrics['theta_T_breach'] = (
        current_step_metrics['avg_delta_P_p'] > current_step_metrics['theta_T_p']
    )
    
    # Calculate derivatives and Speed/Couple
    if len(simulation_log_list) + 1 >= DERIVATIVE_WINDOW_LEN:
        f_crit_hist_for_deriv = np.array(
            [log['f_crit_p'] for log in simulation_log_list[-(DERIVATIVE_WINDOW_LEN-1):]]
            + [current_step_metrics['f_crit_p']]
        )
        beta_hist_for_deriv = np.array(
            [log['beta_lever_p'] for log in simulation_log_list[-(DERIVATIVE_WINDOW_LEN-1):]]
            + [current_step_metrics['beta_lever_p']]
        )
        g_hist_for_deriv = np.array(
            [log['g_lever_p_topple_prob'] for log in simulation_log_list[-(DERIVATIVE_WINDOW_LEN-1):]]
            + [current_step_metrics['g_lever_p_topple_prob']]
        )
        
        dot_f_crit_p_full = calculate_derivatives_savgol(
            f_crit_hist_for_deriv, window_length=DERIVATIVE_WINDOW_LEN
        )
        dot_beta_lever_p_full = calculate_derivatives_savgol(
            beta_hist_for_deriv, window_length=DERIVATIVE_WINDOW_LEN
        )
        dot_g_p_full = calculate_derivatives_savgol(
            g_hist_for_deriv, window_length=DERIVATIVE_WINDOW_LEN
        )
        
        dot_f_crit_p_current = dot_f_crit_p_full[-1] if (dot_f_crit_p_full is not None and not np.all(np.isnan(dot_f_crit_p_full))) else np.nan
        dot_beta_lever_p_current = dot_beta_lever_p_full[-1] if (dot_beta_lever_p_full is not None and not np.all(np.isnan(dot_beta_lever_p_full))) else np.nan
        dot_g_p_current = dot_g_p_full[-1] if (dot_g_p_full is not None and not np.all(np.isnan(dot_g_p_full))) else np.nan
        # Changed default for dot_beta to nan if calculation fails, consistent with dot_fcrit

        current_step_metrics['dot_fcrit_p'] = dot_f_crit_p_current
        current_step_metrics['dot_beta_p'] = dot_beta_lever_p_current
        current_step_metrics['dot_g_p'] = dot_g_p_current
        
        if not np.isnan(dot_f_crit_p_current) and not np.isnan(dot_beta_lever_p_current):
             current_step_metrics['speed_p'] = np.sqrt(dot_f_crit_p_current**2 + dot_beta_lever_p_current**2)
        else:
             current_step_metrics['speed_p'] = np.nan
        
        if len(simulation_log_list) >= COUPLE_WINDOW_LEN:
            # Ensure 'dot_fcrit_p' and 'dot_beta_p' exist in past logs, defaulting to nan
            dot_f_crit_past_segment = [log.get('dot_fcrit_p', np.nan) for log in simulation_log_list[-COUPLE_WINDOW_LEN:]]
            if not np.isnan(dot_f_crit_p_current):
                dot_f_crit_past_segment_for_corr = np.array(dot_f_crit_past_segment[1:] + [dot_f_crit_p_current])
            else:
                dot_f_crit_past_segment_for_corr = np.array(dot_f_crit_past_segment)

            dot_beta_past_segment = [log.get('dot_beta_p', np.nan) for log in simulation_log_list[-COUPLE_WINDOW_LEN:]]
            if not np.isnan(dot_beta_lever_p_current):
                 dot_beta_past_segment_for_corr = np.array(dot_beta_past_segment[1:] + [dot_beta_lever_p_current])
            else:
                 dot_beta_past_segment_for_corr = np.array(dot_beta_past_segment)

            valid_mask = ~np.isnan(dot_f_crit_past_segment_for_corr) & ~np.isnan(dot_beta_past_segment_for_corr)
            segment_f = dot_f_crit_past_segment_for_corr[valid_mask]
            segment_beta = dot_beta_past_segment_for_corr[valid_mask]
            
            if len(segment_f) >= COUPLE_WINDOW_LEN * 0.8: # Need sufficient valid points
                # Check for variance before calculating correlation
                if np.var(segment_f) > 1e-9 and np.var(segment_beta) > 1e-9:
                    corr, _ = pearsonr(segment_f, segment_beta)
                    current_step_metrics['couple_p'] = corr
                elif np.var(segment_f) < 1e-9 and np.var(segment_beta) < 1e-9 : # If both have no variance, they are "perfectly" (though trivially) correlated if constant
                     current_step_metrics['couple_p'] = 0.0 # Or 1.0, but 0.0 is safer for interpretation
                else: # If only one has no variance, correlation is undefined or 0
                     current_step_metrics['couple_p'] = 0.0 # Setting to 0.0 to avoid NaN propagation where possible
            else:
                 current_step_metrics['couple_p'] = np.nan
        else:
            current_step_metrics['couple_p'] = np.nan
    else:
        current_step_metrics['dot_fcrit_p'] = np.nan
        current_step_metrics['dot_beta_p'] = np.nan
        current_step_metrics['dot_g_p'] = np.nan
        current_step_metrics['speed_p'] = np.nan
        current_step_metrics['couple_p'] = np.nan

    simulation_log_list.append(current_step_metrics)

    if t_idx % (NUM_SIM_STEPS // 25) == 0 or t_idx == NUM_SIM_STEPS :
        print(f"Step {t_idx}/{NUM_SIM_STEPS}: Fcrit={f_crit_p_val_for_step:.1f}, AvgStrain={avg_delta_P_p_val_for_step:.2f} (from {num_unstable_pre_relax_val} unstable), "
              f"k_th={k_th_for_current_step}, beta_cont={beta_lever_p_continuous:.2f}, g_p={current_p_topple:.2f}, "
              f"ThetaT={current_step_metrics['theta_T_p']:.2f}, Speed={current_step_metrics.get('speed_p', float('nan')):.2f}, "
              f"Couple={current_step_metrics.get('couple_p', float('nan')):.2f}, Avalanche={avalanche_size_current_event}")

df_log = pd.DataFrame(simulation_log_list)
df_log['safety_margin_p'] = df_log['theta_T_p'] - df_log['avg_delta_P_p']
# Ensure dot_beta_p and dot_fcrit_p are numeric before comparison
df_log['dot_beta_p'] = pd.to_numeric(df_log['dot_beta_p'], errors='coerce')
df_log['dot_fcrit_p'] = pd.to_numeric(df_log['dot_fcrit_p'], errors='coerce')
df_log['is_tightening_loop'] = (df_log['dot_beta_p'] > 0) & (df_log['dot_fcrit_p'] < 0)
df_log['cum_cost_beta'] = df_log['cost_beta_change'].cumsum()
df_log['cum_cost_g'] = df_log['cost_g_maintenance'].cumsum()
df_log['cum_cost_total'] = df_log['cum_cost_beta'] + df_log['cum_cost_g']


# --- Plotting and Analysis (largely unchanged, but results will differ) ---

# Quick diagnostic plots for the NEW strain proxy
plt.figure(figsize=(10,4))
plt.plot(df_log['time_step'], df_log['avg_delta_P_p'], color='steelblue', linewidth=1)
plt.xlabel('Time Step')
plt.ylabel('Time-Averaged Systemic Strain (avg_delta_P_p)')
plt.title(f'Time-Averaged Systemic Strain over Time ({EXPERIMENT_NAME})')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'averaged_strain_timeseries_experiment15A.png'), dpi=SAVEFIG_DPI)
plt.close() # Close plot to free memory

non_zero_averaged_strain = df_log['avg_delta_P_p'][df_log['avg_delta_P_p'] > 1e-6] # Use a small epsilon for "non-zero"
if not non_zero_averaged_strain.empty:
    plt.figure(figsize=(8,4))
    # Determine appropriate bins based on data range
    min_strain = non_zero_averaged_strain.min()
    max_strain = non_zero_averaged_strain.max()
    if max_strain > min_strain: # Ensure there's a range to bin
        bins = np.linspace(min_strain, max_strain, 30)
        plt.hist(non_zero_averaged_strain, bins=bins, edgecolor='black', color='steelblue')
    else: # If all values are the same (or very close)
        plt.hist(non_zero_averaged_strain, bins=10, edgecolor='black', color='steelblue') # Fallback bins
    plt.xlabel('Time-Averaged Systemic Strain (avg_delta_P_p)')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Non-zero Time-Averaged Systemic Strain ({EXPERIMENT_NAME})')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'averaged_strain_histogram_experiment15A.png'), dpi=SAVEFIG_DPI)
    plt.close()


print("\n--- Post-Simulation Analysis ---")

# 1. Avalanche Size Distribution
plt.figure(figsize=(10, 7))
avalanche_data = df_log['avalanche_size'][df_log['avalanche_size'] > 0]
if not avalanche_data.empty:
    min_val_aval = avalanche_data.min()
    max_val_aval = avalanche_data.max()
    if max_val_aval < 1: max_val_aval = 1 
    if min_val_aval < 1: min_val_aval = 1
    if max_val_aval > min_val_aval :
        bins = np.logspace(np.log10(min_val_aval), np.log10(max_val_aval), 50)
    else: # Handle cases where all avalanches are size 1, or very few distinct sizes
        bins = np.array([min_val_aval, min_val_aval +1]) if min_val_aval == max_val_aval else 10

    plt.hist(avalanche_data, bins=bins, log=True, edgecolor='black')
    plt.xscale('log')
    plt.xlabel("Avalanche Size (Number of Toppled Cells)")
    plt.ylabel("Frequency (log scale)")
    plt.title(f"Avalanche Size Distribution ({EXPERIMENT_NAME})")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig(os.path.join(RESULTS_DIR, "avalanche_distribution_experiment15A.png"), dpi=SAVEFIG_DPI)
    plt.close()
else:
    print("No avalanches recorded to plot histogram.")

# 2. Time Series Plots
print("\nGenerating Time Series Plots...")
fig, axs = plt.subplots(6, 1, figsize=(15, 21), sharex=True)
plt.style.use('seaborn-v0_8-whitegrid')

axs[0].plot(df_log['time_step'], df_log['avg_delta_P_p'], label='Systemic Strain (avgDeltaP_p)', linewidth=1.5, color='dodgerblue')
axs[0].plot(df_log['time_step'], df_log['theta_T_p'], label='Tolerance Sheet (ThetaT_p)', linestyle='--', linewidth=1.5, color='darkorange')
ax0_twin = axs[0].twinx()
ax0_twin.plot(df_log['time_step'], df_log['g_lever_p_topple_prob'], label='g_lever_p', color='green', linestyle=':', linewidth=1.2)
ax0_twin.set_ylabel('g_lever_p', color='green')
ax0_twin.tick_params(axis='y', labelcolor='green')
axs[0].set_ylabel("Strain / Tolerance")
lines1, labels1 = axs[0].get_legend_handles_labels()
lines2, labels2 = ax0_twin.get_legend_handles_labels()
axs[0].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
breach_times = df_log['time_step'][df_log['theta_T_breach']]
for bt in breach_times:
    axs[0].axvline(bt, color='magenta', linestyle='--', linewidth=0.8, alpha=0.7, label='_nolegend_') # Thinner line for breaches
if len(breach_times) > 0:
    axs[0].plot([], [], color='magenta', linestyle='--', linewidth=2.0, label='ThetaT Breach') # For legend
axs[0].legend(loc='upper left')

axs[1].plot(df_log['time_step'], df_log['safety_margin_p'], label='Safety Margin (G_p)', color='darkgreen', linewidth=1.5)
axs[1].axhline(0, color='gray', linestyle='--', linewidth=1)
axs[1].set_ylabel("G_p")
axs[1].legend(loc='upper left')

axs[2].plot(df_log['time_step'], df_log['f_crit_p'], label='Energetic Slack (Fcrit_p)', color='teal', linewidth=1.5)
ax2_twin = axs[2].twinx()
ax2_twin.plot(df_log['time_step'], df_log['cum_cost_total'], label='Cumulative Cost', color='brown', linestyle=':', linewidth=1.2)
ax2_twin.set_ylabel('Cum Cost', color='brown')
ax2_twin.tick_params(axis='y', labelcolor='brown')
axs[2].set_ylabel("Fcrit_p")
lines1, labels1 = axs[2].get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
axs[2].legend(lines1 + lines2, labels1 + labels2, loc='upper left')

axs[3].plot(df_log['time_step'], df_log['speed_p'], label='Speed Index (Speed_p)', color='crimson', linewidth=1.5)
axs[3].set_ylabel("Speed_p")
axs[3].legend(loc='upper left')

ax4_twin = axs[4].twinx()
axs[4].plot(df_log['time_step'], df_log['couple_p'], label='Couple Index (Couple_p)', color='purple', linewidth=1.5, alpha=0.7)
axs[4].set_ylabel("Couple_p")
axs[4].legend(loc='upper left')
axs[4].set_ylim(-1.1, 1.1)
ax4_twin.plot(df_log['time_step'], df_log['theta_T_breach'].astype(int),
              label='ThetaT Breach Event', color='sandybrown', drawstyle='steps-post', linewidth=1.5)
ax4_twin.set_ylabel("Breach Event (1=True)", color='sandybrown')
ax4_twin.tick_params(axis='y', labelcolor='sandybrown')
ax4_twin.legend(loc='upper right')
ax4_twin.set_ylim(-0.1, 1.1)

axs[5].plot(df_log['time_step'], df_log['avalanche_size'], label='Avalanche Size', color='forestgreen', alpha=0.8, linewidth=1.5)
axs[5].set_ylabel("Avalanche Size")
axs[5].set_xlabel("Time Step")
axs[5].legend(loc='upper left')
large_avalanche_times = df_log['time_step'][df_log['is_large_avalanche']]
for lat in large_avalanche_times:
    axs[5].axvline(lat, color='black', linestyle='--', alpha=0.7, linewidth=1.0, label='_nolegend_') # Thinner line
if not large_avalanche_times.empty: # Check if series is not empty
    axs[5].plot([], [], color='black', linestyle='--', alpha=0.7, linewidth=1.5, label='Large Avalanche')
axs[5].legend(loc='upper left')

plt.suptitle(f"TD Diagnostics and Sandpile Dynamics ({EXPERIMENT_NAME})", fontsize=18, y=0.99)
fig.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.savefig(os.path.join(RESULTS_DIR, "td_diagnostics_timeseries_experiment15A.png"), dpi=SAVEFIG_DPI)
plt.close()

# --- Event Overlap Analysis ---
analysis_mask = df_log['time_step'] > (BURN_IN_STEPS + POST_BURN_IN_STABILIZATION_STEPS)
lav_flag = df_log.loc[analysis_mask, 'is_large_avalanche']
breach_flag = df_log.loc[analysis_mask, 'theta_T_breach']

# Ensure flags are boolean and same length for logical operations
lav_flag = lav_flag.astype(bool)
breach_flag = breach_flag.astype(bool)

if len(lav_flag) == len(breach_flag): # Ensure they are compatible for direct comparison
    cont_table_vals = [
        np.sum(lav_flag & breach_flag),
        np.sum(lav_flag & ~breach_flag),
        np.sum(~lav_flag & breach_flag),
        np.sum(~lav_flag & ~breach_flag)
    ]
    cont_table_keys = [
        'LargeAvalanche_AND_ThetaBreach', 'LargeAvalanche_AND_NOT_ThetaBreach',
        'NOT_LargeAvalanche_AND_ThetaBreach', 'NOT_LargeAvalanche_AND_NOT_ThetaBreach'
    ]
    cont_table = dict(zip(cont_table_keys, cont_table_vals))

    print("\nEvent Type Overlap after burn-in:")
    for k, v in cont_table.items():
        print(f"  {k}: {v}")
    intersection = cont_table['LargeAvalanche_AND_ThetaBreach']
    union = cont_table['LargeAvalanche_AND_NOT_ThetaBreach'] + \
            cont_table['NOT_LargeAvalanche_AND_ThetaBreach'] + \
            intersection
    if union > 0:
        jaccard = intersection / union
        print(f"  Jaccard Index (Large Avalanche vs ThetaT Breach): {jaccard:.3f}")
    else:
        print("  No events in union for Jaccard Index calculation.")
else:
    print("  Error: Mismatch in lengths of event flags for overlap analysis.")


# --- Pre-Event Speed Analysis ---
print("\n--- Pre-Event Speed Analysis ---")
df_log['speed_p_smooth'] = df_log['speed_p'].rolling(window=SPEED_SMOOTHING_WINDOW_ANALYSIS, center=True, min_periods=3).mean()

event_times_large_avalanche = df_log.loc[df_log['is_large_avalanche'], 'time_step'].values
event_times_theta_breach = df_log.loc[df_log['theta_T_breach'], 'time_step'].values
print(f"Detected {len(event_times_theta_breach)} ThetaT breach timesteps for analysis")

LOOKBACK_WINDOW_ANALYSIS = 50 
MIN_EVENTS_FOR_STAT_TEST = 5

pre_event_speeds_lav_list = []
for etime in event_times_large_avalanche:
    pre_event_df = df_log[(df_log['time_step'] >= etime - LOOKBACK_WINDOW_ANALYSIS) & (df_log['time_step'] < etime)]
    if not pre_event_df.empty and not pre_event_df['speed_p_smooth'].dropna().empty:
        pre_event_speeds_lav_list.append(pre_event_df['speed_p_smooth'].mean())

pre_event_speeds_breach_list = []
for etime in event_times_theta_breach:
    pre_event_df = df_log[(df_log['time_step'] >= etime - LOOKBACK_WINDOW_ANALYSIS) & (df_log['time_step'] < etime)]
    if not pre_event_df.empty and not pre_event_df['speed_p_smooth'].dropna().empty:
        pre_event_speeds_breach_list.append(pre_event_df['speed_p_smooth'].mean())

is_event_period = pd.Series(False, index=df_log.index)
all_event_times = np.unique(np.concatenate((event_times_large_avalanche, event_times_theta_breach)))
for etime in all_event_times:
    event_indices = df_log.index[df_log['time_step'] == etime]
    if not event_indices.empty:
        idx = event_indices[0]
        start_pre_event_exclusion = max(0, idx - LOOKBACK_WINDOW_ANALYSIS)
        is_event_period.iloc[start_pre_event_exclusion : idx + 1] = True

baseline_speeds_vals = df_log.loc[
    (~is_event_period) &
    (df_log['time_step'] > (BURN_IN_STEPS + POST_BURN_IN_STABILIZATION_STEPS)),
    'speed_p_smooth'
].dropna().values

pre_event_speeds_lav_list = [s for s in pre_event_speeds_lav_list if not np.isnan(s)]
pre_event_speeds_breach_list = [s for s in pre_event_speeds_breach_list if not np.isnan(s)]

print(f"Number of large avalanche events considered for speed analysis: {len(pre_event_speeds_lav_list)}")
if pre_event_speeds_lav_list:
    mean_pre_lav_speed = np.mean(pre_event_speeds_lav_list)
    print(f"  Mean Smoothed Speed_p ({LOOKBACK_WINDOW_ANALYSIS} steps) before Large Avalanches: {mean_pre_lav_speed:.3f}")
    if baseline_speeds_vals.size >= MIN_EVENTS_FOR_STAT_TEST and len(pre_event_speeds_lav_list) >= MIN_EVENTS_FOR_STAT_TEST:
        stat, p_val = mannwhitneyu(pre_event_speeds_lav_list, baseline_speeds_vals, alternative='greater', nan_policy='propagate')
        print(f"    Mann-Whitney U vs Baseline (Large Avalanche): Statistic={stat:.2f}, p-value={p_val:.4f}")

print(f"\nNumber of ThetaT breach events considered for speed analysis: {len(pre_event_speeds_breach_list)}")
if pre_event_speeds_breach_list:
    mean_pre_breach_speed = np.mean(pre_event_speeds_breach_list)
    print(f"  Mean Smoothed Speed_p ({LOOKBACK_WINDOW_ANALYSIS} steps) before ThetaT Breaches: {mean_pre_breach_speed:.3f}")
    if baseline_speeds_vals.size >= MIN_EVENTS_FOR_STAT_TEST and len(pre_event_speeds_breach_list) >= MIN_EVENTS_FOR_STAT_TEST:
        stat, p_val = mannwhitneyu(pre_event_speeds_breach_list, baseline_speeds_vals, alternative='greater', nan_policy='propagate')
        print(f"    Mann-Whitney U vs Baseline (ThetaT Breach): Statistic={stat:.2f}, p-value={p_val:.4f}")

if baseline_speeds_vals.size >= MIN_EVENTS_FOR_STAT_TEST :
    print(f"\nMean Smoothed Speed_p during Baseline periods: {np.mean(baseline_speeds_vals):.3f} (N={len(baseline_speeds_vals)})")
else:
    print(f"\nNot enough baseline data to calculate mean baseline speed robustly. N={len(baseline_speeds_vals)}")

plt.figure(figsize=(10,7))
categories = []
means = []
errors = [] 
if pre_event_speeds_lav_list:
    categories.append("Pre-Large Avalanche")
    means.append(np.mean(pre_event_speeds_lav_list))
    errors.append(np.std(pre_event_speeds_lav_list, ddof=1) / np.sqrt(len(pre_event_speeds_lav_list)) if len(pre_event_speeds_lav_list)>1 else 0)
if pre_event_speeds_breach_list:
    categories.append("Pre-ThetaT Breach")
    means.append(np.mean(pre_event_speeds_breach_list))
    errors.append(np.std(pre_event_speeds_breach_list, ddof=1) / np.sqrt(len(pre_event_speeds_breach_list)) if len(pre_event_speeds_breach_list)>1 else 0)
if baseline_speeds_vals.size > 0: # Changed to > 0 from >= MIN_EVENTS_FOR_STAT_TEST for plotting
    categories.append("Baseline")
    means.append(np.mean(baseline_speeds_vals))
    errors.append(np.std(baseline_speeds_vals, ddof=1) / np.sqrt(len(baseline_speeds_vals)) if len(baseline_speeds_vals)>1 else 0)

if categories:
    bars = plt.bar(categories, means, yerr=errors, capsize=5, color=['skyblue', 'lightcoral', 'lightgreen'][:len(categories)], edgecolor='black')
    plt.ylabel("Mean Smoothed Speed Index (Speed_p)")
    plt.title(f"Average Speed Index Prior to Events vs. Baseline (Lookback={LOOKBACK_WINDOW_ANALYSIS} steps)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    max_mean_val = max(means) if means else 1 # Avoid error if means is empty
    for bar_idx, bar in enumerate(bars):
        yval = bar.get_height()
        err_val = errors[bar_idx] if errors and len(errors) > bar_idx else 0
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + err_val + 0.05 * max_mean_val , f'{yval:.3f}', ha='center', va='bottom')
    plt.savefig(os.path.join(RESULTS_DIR, "avg_speed_comparison_experiment15A.png"), dpi=SAVEFIG_DPI)
    plt.close()
else:
    print("No valid event or baseline data recorded to plot average speeds comparison.")

# ... (Rest of the analysis code for Couple Index, Tightening Loop, etc. should follow a similar pattern of checks for data availability) ...

print("\n--- Pre-Event Couple Index Analysis ---")
pre_event_couples_lav_list = []
for etime in event_times_large_avalanche:
    pre_event_df = df_log[(df_log['time_step'] >= etime - LOOKBACK_WINDOW_ANALYSIS) & (df_log['time_step'] < etime)]
    vals = pre_event_df['couple_p'].dropna()
    if not vals.empty:
        pre_event_couples_lav_list.append(vals.mean())

pre_event_couples_breach_list = []
for etime in event_times_theta_breach:
    pre_event_df = df_log[(df_log['time_step'] >= etime - LOOKBACK_WINDOW_ANALYSIS) & (df_log['time_step'] < etime)]
    vals = pre_event_df['couple_p'].dropna()
    if not vals.empty:
        pre_event_couples_breach_list.append(vals.mean())

baseline_couple_vals = df_log.loc[(~is_event_period) & 
                                  (df_log['time_step'] > (BURN_IN_STEPS + POST_BURN_IN_STABILIZATION_STEPS)), 
                                  'couple_p'].dropna().values

print(f"Number of large avalanche events considered for couple analysis: {len(pre_event_couples_lav_list)}")
if pre_event_couples_lav_list:
    mean_lav_couple = np.mean(pre_event_couples_lav_list)
    print(f"  Mean CoupleIndex_p ({LOOKBACK_WINDOW_ANALYSIS} steps) before Large Avalanches: {mean_lav_couple:.3f}")
    if baseline_couple_vals.size >= MIN_EVENTS_FOR_STAT_TEST and len(pre_event_couples_lav_list) >= MIN_EVENTS_FOR_STAT_TEST:
        stat, p_val = mannwhitneyu(pre_event_couples_lav_list, baseline_couple_vals, alternative='two-sided', nan_policy='propagate')
        print(f"    Mann-Whitney U vs Baseline (Large Avalanche): Statistic={stat:.2f}, p-value={p_val:.4f}")

print(f"\nNumber of ThetaT breach events considered for couple analysis: {len(pre_event_couples_breach_list)}")
if pre_event_couples_breach_list:
    mean_breach_couple = np.mean(pre_event_couples_breach_list)
    print(f"  Mean CoupleIndex_p ({LOOKBACK_WINDOW_ANALYSIS} steps) before ThetaT Breaches: {mean_breach_couple:.3f}")
    if baseline_couple_vals.size >= MIN_EVENTS_FOR_STAT_TEST and len(pre_event_couples_breach_list) >= MIN_EVENTS_FOR_STAT_TEST:
        stat, p_val = mannwhitneyu(pre_event_couples_breach_list, baseline_couple_vals, alternative='two-sided', nan_policy='propagate')
        print(f"    Mann-Whitney U vs Baseline (ThetaT Breach): Statistic={stat:.2f}, p-value={p_val:.4f}")

if baseline_couple_vals.size >= MIN_EVENTS_FOR_STAT_TEST:
    print(f"\nMean CoupleIndex_p during Baseline periods: {np.mean(baseline_couple_vals):.3f} (N={len(baseline_couple_vals)})")
else:
    print(f"\nNot enough baseline data for Couple Index. N={len(baseline_couple_vals)}")


plt.figure(figsize=(10,7))
categories_c = []
means_c = []
errors_c = []
if pre_event_couples_lav_list:
    categories_c.append("Pre-Large Avalanche")
    means_c.append(np.mean(pre_event_couples_lav_list))
    errors_c.append(np.std(pre_event_couples_lav_list, ddof=1) / np.sqrt(len(pre_event_couples_lav_list)) if len(pre_event_couples_lav_list)>1 else 0)
if pre_event_couples_breach_list:
    categories_c.append("Pre-ThetaT Breach")
    means_c.append(np.mean(pre_event_couples_breach_list))
    errors_c.append(np.std(pre_event_couples_breach_list, ddof=1) / np.sqrt(len(pre_event_couples_breach_list)) if len(pre_event_couples_breach_list)>1 else 0)
if baseline_couple_vals.size > 0: # Changed for plotting
    categories_c.append("Baseline")
    means_c.append(np.mean(baseline_couple_vals))
    errors_c.append(np.std(baseline_couple_vals, ddof=1) / np.sqrt(len(baseline_couple_vals)) if len(baseline_couple_vals)>1 else 0)

if categories_c:
    bars_c = plt.bar(categories_c, means_c, yerr=errors_c, capsize=5, color=['skyblue','lightcoral','lightgreen'][:len(categories_c)], edgecolor='black')
    plt.ylabel("Mean Couple Index (Couple_p)")
    plt.title(f"Average Couple Index Prior to Events vs. Baseline (Lookback={LOOKBACK_WINDOW_ANALYSIS} steps)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    max_mean_c_val = max(np.abs(means_c)) if means_c else 1 # Use abs for y-axis text offset calculation for CoupleIndex
    for bar_idx, bar in enumerate(bars_c):
        yval = bar.get_height()
        err_val = errors_c[bar_idx] if errors_c and len(errors_c) > bar_idx else 0
        text_y_pos = yval + (np.sign(yval) * err_val if err_val > 0 else 0) + np.sign(yval) * 0.05 * max_mean_c_val # Adjust text based on sign
        if yval == 0: text_y_pos = 0.05 * max_mean_c_val # Avoid placing text at 0 if bar is 0
        plt.text(bar.get_x() + bar.get_width()/2.0, text_y_pos , f'{yval:.3f}', ha='center', va='bottom' if yval >=0 else 'top')
    plt.ylim(min(0, min(means_c)-0.1) if means_c else -0.1, max(0, max(means_c)+0.1) if means_c else 0.1) # Adjust y-limits for CoupleIndex
    plt.savefig(os.path.join(RESULTS_DIR, "avg_couple_comparison_experiment15A.png"), dpi=SAVEFIG_DPI)
    plt.close()
else:
    print("No valid event or baseline data recorded to plot average couple index comparison.")


# --- Tightening Loop Motif Analysis ---
pre_tight_lav = []
for etime in event_times_large_avalanche:
    pre_df = df_log[(df_log['time_step'] >= etime - LOOKBACK_WINDOW_ANALYSIS) & (df_log['time_step'] < etime)]
    if not pre_df.empty and not pre_df['is_tightening_loop'].dropna().empty:
        pre_tight_lav.append(pre_df['is_tightening_loop'].mean())

pre_tight_breach = []
for etime in event_times_theta_breach:
    pre_df = df_log[(df_log['time_step'] >= etime - LOOKBACK_WINDOW_ANALYSIS) & (df_log['time_step'] < etime)]
    if not pre_df.empty and not pre_df['is_tightening_loop'].dropna().empty:
        pre_tight_breach.append(pre_df['is_tightening_loop'].mean())

baseline_tight_mask = (
    ~is_event_period &
    (df_log['time_step'] > (BURN_IN_STEPS + POST_BURN_IN_STABILIZATION_STEPS))
)
baseline_tight_values = df_log.loc[baseline_tight_mask, 'is_tightening_loop'].dropna()
baseline_tight_proportion = baseline_tight_values.mean() if not baseline_tight_values.empty else np.nan
num_baseline_tight_timesteps = baseline_tight_values.sum() if not baseline_tight_values.empty else 0


print("\n--- Tightening Loop Prevalence ---")
print(f"  Baseline tightening proportion: {baseline_tight_proportion:.3f} (N={int(num_baseline_tight_timesteps)})")

if pre_tight_lav:
    mean_pre_tight_lav = np.mean(pre_tight_lav)
    print(f"  Mean proportion before Large Avalanches: {mean_pre_tight_lav:.3f}")
    if not np.isnan(baseline_tight_proportion) and len(pre_tight_lav) >= MIN_EVENTS_FOR_STAT_TEST and baseline_tight_values.size >= MIN_EVENTS_FOR_STAT_TEST:
        # Perform test against the mean proportion observed in baseline, not a list of identical values
        # This might require a different test or interpretation if baseline_tight_values isn't a distribution.
        # For now, let's compare the distribution of pre_tight_lav with the distribution of baseline_tight_values
        stat, p_val = mannwhitneyu(pre_tight_lav, baseline_tight_values, alternative='greater', nan_policy='propagate')
        print(f"    Mann-Whitney U vs Baseline values (Large Avalanche): Statistic={stat:.2f}, p-value={p_val:.4f}")

if pre_tight_breach:
    mean_pre_tight_breach = np.mean(pre_tight_breach)
    print(f"  Mean proportion before ThetaT Breaches: {mean_pre_tight_breach:.3f}")
    if not np.isnan(baseline_tight_proportion) and len(pre_tight_breach) >= MIN_EVENTS_FOR_STAT_TEST and baseline_tight_values.size >= MIN_EVENTS_FOR_STAT_TEST:
        stat, p_val = mannwhitneyu(pre_tight_breach, baseline_tight_values, alternative='greater', nan_policy='propagate')
        print(f"    Mann-Whitney U vs Baseline values (ThetaT Breach): Statistic={stat:.2f}, p-value={p_val:.4f}")


# --- Positive Coupling Sign Analysis ---
COUPLE_POS_THRESHOLD = 0.3
combined_pre_event_mask = pd.Series(False, index=df_log.index)
for etime in all_event_times: # Use all_event_times which is unique and combined
    mask_indices = df_log.index[(df_log['time_step'] >= etime - LOOKBACK_WINDOW_ANALYSIS) & (df_log['time_step'] < etime)]
    if not mask_indices.empty:
        combined_pre_event_mask.loc[mask_indices] = True


high_couple_df = df_log[combined_pre_event_mask & (df_log['couple_p'].notna()) & (df_log['couple_p'] > COUPLE_POS_THRESHOLD)].dropna(subset=['dot_beta_p', 'dot_fcrit_p'])

if not high_couple_df.empty:
    q1_pos = np.sum((high_couple_df['dot_beta_p'] > 0) & (high_couple_df['dot_fcrit_p'] > 0))
    q2_pos = np.sum((high_couple_df['dot_beta_p'] < 0) & (high_couple_df['dot_fcrit_p'] > 0))
    q3_pos = np.sum((high_couple_df['dot_beta_p'] < 0) & (high_couple_df['dot_fcrit_p'] < 0))
    q4_pos = np.sum((high_couple_df['dot_beta_p'] > 0) & (high_couple_df['dot_fcrit_p'] < 0))
    total_high = len(high_couple_df)
    other_combo = total_high - q1_pos - q2_pos - q3_pos - q4_pos
    print("\n--- Positive Coupling Sign Patterns (CoupleIndex_p > {:.2f}) ---".format(COUPLE_POS_THRESHOLD))
    print(f"  (during pre-event windows, N_high_couple_timesteps = {total_high})")
    print(f"  Both derivatives > 0: {q1_pos/total_high:.2%} ")
    print(f"  Both derivatives < 0: {q3_pos/total_high:.2%} ")
    print(f"  Mixed signs       : {other_combo/total_high:.2%} ")

    plt.figure(figsize=(8,6))
    sc = plt.scatter(high_couple_df['dot_beta_p'], high_couple_df['dot_fcrit_p'],
                     c=high_couple_df['couple_p'], cmap='coolwarm', edgecolor='k', alpha=0.7, vmin=-1, vmax=1) # Set vmin/vmax for cmap
    plt.axvline(0, color='gray', linestyle='--', linewidth=1)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.colorbar(sc, label='CoupleIndex_p')
    plt.xlabel('dot_beta_p')
    plt.ylabel('dot_fcrit_p')
    plt.title('Derivative Relationship Prior to Events\n(CoupleIndex_p > {:.2f})'.format(COUPLE_POS_THRESHOLD))
    # Add some padding to axis limits if derivatives are very small
    xlim_curr = plt.xlim()
    ylim_curr = plt.ylim()
    plt.xlim(xlim_curr[0] - abs(xlim_curr[0])*0.1 if xlim_curr[0]!=0 else -1e-9, xlim_curr[1] + abs(xlim_curr[1])*0.1 if xlim_curr[1]!=0 else 1e-9)
    plt.ylim(ylim_curr[0] - abs(ylim_curr[0])*0.1 if ylim_curr[0]!=0 else -1e-9, ylim_curr[1] + abs(ylim_curr[1])*0.1 if ylim_curr[1]!=0 else 1e-9)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'pre_event_derivative_scatter_experiment15A.png'), dpi=SAVEFIG_DPI)
    plt.close()
    high_pos_sign_dict = {
        'N': int(total_high),
        'both_pos': int(q1_pos),
        'beta_neg': int(q2_pos),
        'both_neg': int(q3_pos),
        'tightening': int(q4_pos)
    }
else:
    print(f"\nNo data points with CoupleIndex_p > {COUPLE_POS_THRESHOLD} in pre-event windows.")
    high_pos_sign_dict = {
        'N': 0,
        'both_pos': 0,
        'beta_neg': 0,
        'both_neg': 0,
        'tightening': 0
    }


COUPLE_NEG_THRESHOLD = -0.3
neg_couple_df = df_log[combined_pre_event_mask & (df_log['couple_p'].notna()) & (df_log['couple_p'] < COUPLE_NEG_THRESHOLD)].dropna(subset=['dot_beta_p', 'dot_fcrit_p'])
if not neg_couple_df.empty:
    print("\n--- Negative Coupling Sign Patterns (CoupleIndex_p < {:.2f}) ---".format(COUPLE_NEG_THRESHOLD))
    quad1 = np.sum((neg_couple_df['dot_beta_p'] > 0) & (neg_couple_df['dot_fcrit_p'] > 0))
    quad2 = np.sum((neg_couple_df['dot_beta_p'] < 0) & (neg_couple_df['dot_fcrit_p'] > 0))
    quad3 = np.sum((neg_couple_df['dot_beta_p'] < 0) & (neg_couple_df['dot_fcrit_p'] < 0))
    quad4 = np.sum((neg_couple_df['dot_beta_p'] > 0) & (neg_couple_df['dot_fcrit_p'] < 0))
    tot_neg = len(neg_couple_df)
    print(f"  (during pre-event windows, N_neg_couple_timesteps = {tot_neg})")
    print(f"  Q1 (dot_beta > 0, dot_fcrit > 0): {quad1/tot_neg:.2%}")
    print(f"  Q2 (dot_beta < 0, dot_fcrit > 0): {quad2/tot_neg:.2%}")
    print(f"  Q3 (dot_beta < 0, dot_fcrit < 0): {quad3/tot_neg:.2%}")
    print(f"  Q4 (dot_beta > 0, dot_fcrit < 0) [Tightening Loop]: {quad4/tot_neg:.2%}")


    plt.figure(figsize=(8,6))
    sc = plt.scatter(neg_couple_df['dot_beta_p'], neg_couple_df['dot_fcrit_p'],
                     c=neg_couple_df['couple_p'], cmap='coolwarm_r', edgecolor='k', alpha=0.7, vmin=-1, vmax=1) # Reversed cmap, set vmin/vmax
    plt.axvline(0, color='gray', linestyle='--', linewidth=1)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.colorbar(sc, label='CoupleIndex_p')
    plt.xlabel('dot_beta_p')
    plt.ylabel('dot_fcrit_p')
    plt.title('Derivative Relationship Prior to Events\n(CoupleIndex_p < {:.2f})'.format(COUPLE_NEG_THRESHOLD))
    xlim_curr_neg = plt.xlim()
    ylim_curr_neg = plt.ylim()
    plt.xlim(xlim_curr_neg[0] - abs(xlim_curr_neg[0])*0.1 if xlim_curr_neg[0]!=0 else -1e-9, xlim_curr_neg[1] + abs(xlim_curr_neg[1])*0.1 if xlim_curr_neg[1]!=0 else 1e-9)
    plt.ylim(ylim_curr_neg[0] - abs(ylim_curr_neg[0])*0.1 if ylim_curr_neg[0]!=0 else -1e-9, ylim_curr_neg[1] + abs(ylim_curr_neg[1])*0.1 if ylim_curr_neg[1]!=0 else 1e-9)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'pre_event_negative_deriv_scatter_experiment15A.png'), dpi=SAVEFIG_DPI)
    plt.close()
    high_neg_sign_dict = {
        'N': int(tot_neg),
        'q1': int(quad1),
        'q2': int(quad2),
        'q3': int(quad3),
        'q4': int(quad4)
    }
else:
    print(f"\nNo data points with CoupleIndex_p < {COUPLE_NEG_THRESHOLD} in pre-event windows.")
    high_neg_sign_dict = {
        'N': 0,
        'q1': 0,
        'q2': 0,
        'q3': 0,
        'q4': 0
    }


# --- Quadrant Distribution Analysis ---
def quadrant_counts(df_segment):
    q1 = np.sum((df_segment['dot_beta_p'] > 0) & (df_segment['dot_fcrit_p'] > 0)) # ++
    q2 = np.sum((df_segment['dot_beta_p'] < 0) & (df_segment['dot_fcrit_p'] > 0)) # -+
    q3 = np.sum((df_segment['dot_beta_p'] < 0) & (df_segment['dot_fcrit_p'] < 0)) # --
    q4 = np.sum((df_segment['dot_beta_p'] > 0) & (df_segment['dot_fcrit_p'] < 0)) # +- (Tightening)
    return np.array([q1, q2, q3, q4], dtype=float)

pre_event_df_full = df_log[combined_pre_event_mask].dropna(subset=['dot_beta_p','dot_fcrit_p'])
pre_event_counts = quadrant_counts(pre_event_df_full) if not pre_event_df_full.empty else np.zeros(4)

baseline_df_for_quadrants = df_log.loc[baseline_tight_mask].dropna(subset=['dot_beta_p','dot_fcrit_p'])
baseline_counts = quadrant_counts(baseline_df_for_quadrants) if not baseline_df_for_quadrants.empty else np.zeros(4)

print("\n--- Derivative Quadrant Analysis (All Pre-Event vs Baseline) ---")
if pre_event_counts.sum() > 0 and baseline_counts.sum() > 0 :
    pre_event_props = pre_event_counts / pre_event_counts.sum()
    base_props = baseline_counts / baseline_counts.sum()
    
    print(f"  Pre-Event Quadrant Proportions (N={pre_event_counts.sum()}): Q1(++)={pre_event_props[0]:.2%}, Q2(-+)={pre_event_props[1]:.2%}, Q3(--)={pre_event_props[2]:.2%}, Q4(+-)={pre_event_props[3]:.2%}")
    print(f"  Baseline Quadrant Proportions  (N={baseline_counts.sum()}): Q1(++)={base_props[0]:.2%}, Q2(-+)={base_props[1]:.2%}, Q3(--)={base_props[2]:.2%}, Q4(+-)={base_props[3]:.2%}")
    
    cont_table = np.vstack([pre_event_counts, baseline_counts])
    # Check for zeros in expected frequencies before calling chi2_contingency
    expected_freq = chi2_contingency(cont_table)[3] if cont_table.sum() > 0 else None
    if expected_freq is not None and np.all(expected_freq > 0):
        chi2, p, _, _ = chi2_contingency(cont_table)
        print(f"  Chi-squared test for difference in distributions: chi2={chi2:.2f}, p-value={p:.4f}")
    else:
        print("  Chi-squared test skipped due to zero expected frequencies or no data.")
else:
    print("  Not enough data for Quadrant Distribution Analysis.")


# --- Conceptual Hazard Rate Formulation ---
df_log['hazard_rate_p_raw'] = (df_log['speed_p_smooth']**2) / (1 - df_log['couple_p'].fillna(0) + 1e-6) # Fill NaN couple_p with 0 for this calculation
# Filter out extreme values for plotting and analysis, e.g. due to (1-CoupleIndex) being very small
hazard_percentiles = df_log['hazard_rate_p_raw'].quantile([0.01, 0.99]).dropna()
if not hazard_percentiles.empty:
    min_hazard_plot = hazard_percentiles.iloc[0]
    max_hazard_plot = hazard_percentiles.iloc[1]
    df_log['hazard_rate_p_clipped'] = df_log['hazard_rate_p_raw'].clip(lower=min_hazard_plot, upper=max_hazard_plot)
else:
    df_log['hazard_rate_p_clipped'] = df_log['hazard_rate_p_raw'] # No clipping if quantiles fail

plt.figure(figsize=(12,4))
plt.plot(df_log['time_step'], df_log['hazard_rate_p_clipped'], color='darkred', linewidth=0.8)
plt.xlabel('Time Step')
plt.ylabel('Hazard_Rate_p (Clipped)')
plt.title(f'Conceptual Hazard Rate over Time ({EXPERIMENT_NAME})')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'hazard_rate_timeseries_experiment15A.png'), dpi=SAVEFIG_DPI)
plt.close()


pre_hazard_breach = []
for etime in event_times_theta_breach: # Assuming event_times_theta_breach is defined
    pre_df = df_log[(df_log['time_step'] >= etime - LOOKBACK_WINDOW_ANALYSIS) & (df_log['time_step'] < etime)]
    if not pre_df.empty and 'hazard_rate_p_clipped' in pre_df.columns:
        vals = pre_df['hazard_rate_p_clipped'].dropna()
        if not vals.empty:
            pre_hazard_breach.append(vals.mean())

baseline_hazard_series = df_log.loc[baseline_tight_mask, 'hazard_rate_p_clipped'].dropna() # baseline_tight_mask defined earlier

if pre_hazard_breach and not baseline_hazard_series.empty:
    print(f"\nMean hazard rate before ThetaT breaches: {np.mean(pre_hazard_breach):.3f} (N_events={len(pre_hazard_breach)})")
    print(f"Mean hazard rate during baseline: {baseline_hazard_series.mean():.3f} (N_baseline_points={len(baseline_hazard_series)})")
    if len(pre_hazard_breach) >= MIN_EVENTS_FOR_STAT_TEST and len(baseline_hazard_series) >= MIN_EVENTS_FOR_STAT_TEST:
        stat, p_val = mannwhitneyu(pre_hazard_breach, baseline_hazard_series, alternative='greater')
        print(f"  Mann-Whitney U (Pre-Breach Hazard > Baseline Hazard): stat={stat:.2f}, p={p_val:.4f}")
elif not pre_hazard_breach:
    print("\nNo ThetaT breach events to analyze for pre-event hazard rate.")
elif baseline_hazard_series.empty:
    print("\nNo baseline data to compare pre-event hazard rate against.")


print(f"\nSimulation {EXPERIMENT_NAME} Complete.")

# Calculate post burn-in averages for summary
post_burn_mask_full = df_log['time_step'] > (BURN_IN_STEPS + POST_BURN_IN_STABILIZATION_STEPS)
mean_safety_margin_val = float(df_log.loc[post_burn_mask_full, 'safety_margin_p'].mean()) if not df_log.loc[post_burn_mask_full, 'safety_margin_p'].empty else float('nan')
mean_beta_lever_val = float(df_log.loc[post_burn_mask_full, 'beta_lever_p'].mean()) if not df_log.loc[post_burn_mask_full, 'beta_lever_p'].empty else float('nan')
mean_g_lever_val = float(df_log.loc[post_burn_mask_full, 'g_lever_p_topple_prob'].mean()) if not df_log.loc[post_burn_mask_full, 'g_lever_p_topple_prob'].empty else float('nan')

# --- Additional summary statistics for slack and cost ---
fcrit_baseline_vals = df_log.loc[baseline_tight_mask, 'f_crit_p'].dropna()
fcrit_baseline_mean = float(fcrit_baseline_vals.mean()) if not fcrit_baseline_vals.empty else float('nan')
fcrit_baseline_sd = float(fcrit_baseline_vals.std(ddof=1)) if len(fcrit_baseline_vals) > 1 else float('nan')

if df_log.loc[post_burn_mask_full, 'f_crit_p'].dropna().shape[0] > 1:
    fcrit_trend_slope_val = float(
        linregress(
            df_log.loc[post_burn_mask_full, 'time_step'],
            df_log.loc[post_burn_mask_full, 'f_crit_p'],
        ).slope
    )
else:
    fcrit_trend_slope_val = float('nan')
fcrit_final_val = float(df_log['f_crit_p'].iloc[-1]) if not df_log['f_crit_p'].empty else float('nan')

final_beta_cost_val = float(df_log['cum_cost_beta'].iloc[-1]) if not df_log['cum_cost_beta'].empty else float('nan')
final_g_cost_val = float(df_log['cum_cost_g'].iloc[-1]) if not df_log['cum_cost_g'].empty else float('nan')
final_total_cost_val = float(df_log['cum_cost_total'].iloc[-1]) if not df_log['cum_cost_total'].empty else float('nan')

if df_log.loc[post_burn_mask_full, 'avg_delta_P_p'].dropna().shape[0] > 1:
    strain_trend_slope_val = float(
        linregress(
            df_log.loc[post_burn_mask_full, 'time_step'],
            df_log.loc[post_burn_mask_full, 'avg_delta_P_p'],
        ).slope
    )
else:
    strain_trend_slope_val = float('nan')
if df_log.loc[post_burn_mask_full, 'theta_T_p'].dropna().shape[0] > 1:
    theta_trend_slope_val = float(
        linregress(
            df_log.loc[post_burn_mask_full, 'time_step'],
            df_log.loc[post_burn_mask_full, 'theta_T_p'],
        ).slope
    )
else:
    theta_trend_slope_val = float('nan')

# --- Write compact numerical summary for LLM reporting ---
import td_summary
summary_params = {
    "EXPERIMENT_NAME": EXPERIMENT_NAME,
    "GRID_SIZE": GRID_SIZE,
    "BURN_IN_STEPS": BURN_IN_STEPS,
    "NUM_SIM_STEPS": NUM_SIM_STEPS,
    "SEED": 42,
}

# Re-compute p-values for tightening loop comparisons if possible
tight_p_LA_val = np.nan
if pre_tight_lav and not baseline_tight_values.empty:
    _, tight_p_LA_val = mannwhitneyu(pre_tight_lav, baseline_tight_values, alternative="greater")

tight_p_B_val = np.nan
if pre_tight_breach and not baseline_tight_values.empty:
    _, tight_p_B_val = mannwhitneyu(pre_tight_breach, baseline_tight_values, alternative="greater")

analysis_results = {
    "pre_speed_LA": pre_event_speeds_lav_list,
    "pre_speed_B": pre_event_speeds_breach_list,
    "baseline_speed": baseline_speeds_vals,
    "pre_couple_LA": pre_event_couples_lav_list,
    "pre_couple_B": pre_event_couples_breach_list,
    "baseline_couple": baseline_couple_vals,
    "pre_hazard_B": pre_hazard_breach,
    "baseline_hazard": baseline_hazard_series,
    "quadrant_pre": pre_event_counts,
    "quadrant_base": baseline_counts,
    "quadrant_chi2": chi2 if 'chi2' in locals() else np.nan,
    "quadrant_p": p if 'p' in locals() else np.nan,
    "tight_base": baseline_tight_proportion,
    "tight_LA": np.mean(pre_tight_lav) if pre_tight_lav else np.nan,
    "tight_B": np.mean(pre_tight_breach) if pre_tight_breach else np.nan,
    "tight_p_LA": tight_p_LA_val,
    "tight_p_B": tight_p_B_val,
    "high_pos_signs": high_pos_sign_dict,
    "high_neg_signs": high_neg_sign_dict,
    "mean_safety_margin": mean_safety_margin_val,
    "mean_beta_lever_p": mean_beta_lever_val,
    "mean_g_lever_p": mean_g_lever_val,
    "fcrit_baseline_mean": fcrit_baseline_mean,
    "fcrit_baseline_sd": fcrit_baseline_sd,
    "fcrit_trend_slope": fcrit_trend_slope_val,
    "fcrit_final_value": fcrit_final_val,
    "final_beta_cost": final_beta_cost_val,
    "final_g_cost": final_g_cost_val,
    "final_total_cost": final_total_cost_val,
    "strain_trend_slope": strain_trend_slope_val,
    "theta_trend_slope": theta_trend_slope_val,
}

summary_dict = td_summary.make_summary(df_log, summary_params, analysis_results)
with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
    json.dump(summary_dict, f, indent=2, sort_keys=True)
print(f"summary.json written to {RESULTS_DIR}")
