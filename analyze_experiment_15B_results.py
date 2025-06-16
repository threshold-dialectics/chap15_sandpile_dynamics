# analyze_experiment_15B_results.py
import os
import glob 
import argparse
from collections import defaultdict
import json

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency, pearsonr, linregress
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

from td_core import calculate_derivatives_savgol

G_UPPER_SOTC_BOUND = 0.05
G_LOWER_SOTC_BOUND = -0.20
MIN_SOTC_DURATION = 30
BURN_IN_STEPS = 2000
LARGE_AVALANCHE_THRESH = 27  # 30x30 grid *0.03
LOOKBACK_WINDOW_H3 = 100
PRE_EVENT_WINDOW_H4 = 50
MIN_AVALS_FOR_ALPHA_FIT = 100


def identify_sustained_sotc_periods(df_log: pd.DataFrame, g_upper: float, g_lower: float, min_duration: int) -> pd.DataFrame:
    in_bounds = (df_log['safety_margin_g'] < g_upper) & (df_log['safety_margin_g'] > g_lower)
    group_ids = (in_bounds != in_bounds.shift()).cumsum()
    sustained = pd.Series(False, index=df_log.index)
    for _, grp in df_log.groupby(group_ids):
        if in_bounds.loc[grp.index].all() and len(grp) >= min_duration:
            sustained.loc[grp.index] = True
    df_log['is_sustained_sotc_period'] = sustained
    return df_log


def fit_power_law_robust(avalanche_sizes: np.ndarray, xmin_method="LikelihoodRatioTest", discrete: bool = True):
    """Fit a discrete power-law distribution and return alpha, KS-D and xmin."""
    avals = avalanche_sizes[avalanche_sizes > 0]
    if len(avals) < 50:
        return np.nan, np.nan, np.nan
    try:
        import powerlaw

        if xmin_method == "LikelihoodRatioTest" and len(np.unique(avals)) > 1:
            fit = powerlaw.Fit(avals, discrete=discrete, verbose=False)
            xmin_choice = fit.xmin
        elif isinstance(xmin_method, (int, float)):
            xmin_choice = xmin_method
        else:
            xmin_choice = avals.min() if len(avals) > 0 else 1

        fit = powerlaw.Fit(avals, xmin=xmin_choice, discrete=discrete, verbose=False)
        return float(fit.alpha), float(fit.D), float(fit.xmin)
    except Exception:
        return np.nan, np.nan, np.nan


def load_batch_logs(parent_dir: str) -> pd.DataFrame:
    logs = []
    for sub in sorted(os.listdir(parent_dir)):
        sub_path = os.path.join(parent_dir, sub)
        if not os.path.isdir(sub_path):
            continue
        csvs = glob.glob(os.path.join(sub_path, 'sotc_simulation_log_*.csv'))
        for csv in csvs:
            df = pd.read_csv(csv)
            df['run_id'] = sub
            logs.append(df)
    if not logs:
        raise FileNotFoundError('No simulation logs found in ' + parent_dir)
    df_all = pd.concat(logs, ignore_index=True)
    return df_all


def ensure_sotc_column(df: pd.DataFrame) -> pd.DataFrame:
    if 'is_sustained_sotc_period' not in df.columns:
        df['is_sustained_sotc_period'] = False
    out_frames = []
    for run, grp in df.groupby('run_id'):
        tmp = identify_sustained_sotc_periods(grp, G_UPPER_SOTC_BOUND, G_LOWER_SOTC_BOUND, MIN_SOTC_DURATION)
        tmp.loc[tmp['time_step'] <= BURN_IN_STEPS, 'is_sustained_sotc_period'] = False
        out_frames.append(tmp)
    return pd.concat(out_frames, ignore_index=True)


def recompute_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute TD diagnostic columns per run_id."""
    out_frames = []
    for run, grp in df.groupby('run_id'):
        g = grp.copy()
        g['dot_beta_p'] = calculate_derivatives_savgol(g['beta_lever_p'], window_length=11)
        g['dot_fcrit_p'] = calculate_derivatives_savgol(g['f_crit_p'], window_length=11)
        g['speed_p'] = np.sqrt(g['dot_beta_p']**2 + g['dot_fcrit_p']**2)
        corr_vals = [np.nan] * len(g)
        for i in range(30 - 1, len(g)):
            seg_b = g['dot_beta_p'].iloc[i-29:i+1]
            seg_f = g['dot_fcrit_p'].iloc[i-29:i+1]
            if seg_b.isna().any() or seg_f.isna().any():
                continue
            if np.var(seg_b) > 1e-9 and np.var(seg_f) > 1e-9:
                corr_vals[i] = pearsonr(seg_b, seg_f)[0]
            else:
                corr_vals[i] = 0.0
        g['couple_p'] = corr_vals
        g['Q1_both_pos'] = (g['dot_beta_p'] > 0) & (g['dot_fcrit_p'] > 0)
        g['Q2_beta_neg_fcrit_pos'] = (g['dot_beta_p'] < 0) & (g['dot_fcrit_p'] > 0)
        g['Q3_both_neg'] = (g['dot_beta_p'] < 0) & (g['dot_fcrit_p'] < 0)
        g['Q4_beta_pos_fcrit_neg'] = (g['dot_beta_p'] > 0) & (g['dot_fcrit_p'] < 0)
        out_frames.append(g)
    return pd.concat(out_frames, ignore_index=True)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cliff's delta effect size for two samples."""
    n1 = len(x)
    n2 = len(y)
    if n1 == 0 or n2 == 0:
        return np.nan
    u, _ = mannwhitneyu(x, y, alternative="two-sided")
    return 2.0 * u / (n1 * n2) - 1.0


def descr_stats(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "median": float(np.median(arr)),
        "iqr": [float(np.percentile(arr, 25)), float(np.percentile(arr, 75))],
        "mean": float(np.mean(arr)),
        "sd": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
    }



def plot_full_diagnostics(df_run: pd.DataFrame, run_label: str, out_path: str) -> None:
    """Replicate the seven-panel diagnostic plot from the simulation script."""
    fig, axs = plt.subplots(7, 1, figsize=(15, 24), sharex=True)

    axs[0].plot(df_run['time_step'], df_run['avg_delta_P_p'], label='Systemic Strain', linewidth=1.0)
    axs[0].plot(df_run['time_step'], df_run['theta_T_p'], label='Tolerance Sheet', linestyle='--', linewidth=1.0)
    ax0_twin = axs[0].twinx()
    ax0_twin.plot(df_run['time_step'], df_run['g_lever_p_topple_prob'], color='green', linestyle=':', label='g_lever_p')
    axs[0].set_ylabel('Strain / Tolerance')
    lines1, labels1 = axs[0].get_legend_handles_labels()
    lines2, labels2 = ax0_twin.get_legend_handles_labels()
    axs[0].legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    axs[1].plot(df_run['time_step'], df_run['safety_margin_g'], label='G_p', color='darkgreen')
    axs[1].axhline(G_UPPER_SOTC_BOUND, color='red', linestyle=':')
    axs[1].axhline(G_LOWER_SOTC_BOUND, color='red', linestyle=':')
    axs[1].fill_between(
        df_run['time_step'],
        axs[1].get_ylim()[0],
        axs[1].get_ylim()[1],
        where=df_run['is_sustained_sotc_period'],
        color='lightcoral',
        alpha=0.2,
        interpolate=True,
        label='SOTC Period',
    )
    axs[1].set_ylabel('G_p')
    axs[1].legend(loc='upper left')

    axs[2].plot(df_run['time_step'], df_run['f_crit_p'], label='Fcrit_p', color='teal')
    axs[2].set_ylabel('Fcrit_p')
    axs[2].legend(loc='upper left')

    axs[3].plot(df_run['time_step'], df_run['beta_lever_p'], label='Beta Lever', color='blueviolet')
    axs[3].plot(df_run['time_step'], df_run['actual_k_th'], label='Actual k_th', linestyle='--', color='indigo')
    axs[3].set_ylabel('Beta / k_th')
    axs[3].legend(loc='upper left')

    axs[4].plot(df_run['time_step'], df_run['speed_p'], label='Speed Index', color='crimson')
    axs[4].set_ylabel('Speed_p')
    axs[4].legend(loc='upper left')

    axs[5].plot(df_run['time_step'], df_run['couple_p'], label='Couple Index', color='purple')
    axs[5].set_ylabel('Couple_p')
    axs[5].set_ylim(-1.1, 1.1)
    axs[5].legend(loc='upper left')

    axs[6].plot(df_run['time_step'], df_run['avalanche_size'], label='Avalanche Size', color='forestgreen')
    axs[6].set_ylabel('Avalanche Size')
    axs[6].set_xlabel('Time Step')
    axs[6].legend(loc='upper left')

    plt.suptitle(f'Full TD Diagnostics and Sandpile Dynamics ({run_label})', fontsize=18, y=0.99)
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(out_path, dpi=SAVEFIG_DPI)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Analyze batch Experiment 15B results')
    parser.add_argument('parent_results_dir', help='Directory containing run subfolders')
    args = parser.parse_args()

    summary_dir = os.path.join(args.parent_results_dir, '_BatchAnalysis_Summary')
    os.makedirs(summary_dir, exist_ok=True)

    df_all = load_batch_logs(args.parent_results_dir)
    df_all = recompute_diagnostics(df_all)
    df_all = ensure_sotc_column(df_all)

    summary = {}
    summary['general'] = {
        'n_runs': int(df_all['run_id'].nunique()),
        'timesteps_total': int(len(df_all)),
        'avalanches_total': int((df_all['avalanche_size'] > 0).sum()),
        'large_avalanches_total': int(df_all['is_large_avalanche'].sum()),
    }
    # helper columns for later analyses
    df_all['sotc_prev'] = df_all.groupby('run_id')['is_sustained_sotc_period'].shift(1).fillna(False)
    df_all['prop_q4_roll20'] = (
        df_all.groupby('run_id')['Q4_beta_pos_fcrit_neg']
        .rolling(window=20, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # ------------------------------------------------------------------
    # H1: fraction time in SOTC
    # ------------------------------------------------------------------
    fractions = df_all[df_all['time_step'] > BURN_IN_STEPS].groupby('run_id')['is_sustained_sotc_period'].mean()
    summary['H1'] = {
        'fraction_sotc_by_run': {k: float(v) for k, v in fractions.items()},
        'mean': float(fractions.mean()),
        'sd': float(fractions.std()),
        'min': float(fractions.min()),
        'max': float(fractions.max()),
        'steps_in_sotc': int(df_all['is_sustained_sotc_period'].sum()),
        'steps_outside': int((~df_all['is_sustained_sotc_period']).sum()),
    }

    # SOTC segment structure
    seg_counts = {}
    seg_durations = []
    for run, grp in df_all.groupby('run_id'):
        flag = grp['is_sustained_sotc_period'].values
        changes = np.concatenate(([True], flag[1:] != flag[:-1]))
        ids = changes.cumsum()
        durations = [len(g) for i, g in grp.groupby(ids) if g['is_sustained_sotc_period'].iloc[0]]
        seg_counts[run] = len(durations)
        seg_durations.extend(durations)

    summary['sotc_segment_stats'] = {
        'segments_per_run': seg_counts,
        'segments_per_run_mean': float(np.mean(list(seg_counts.values()))),
        'segments_per_run_sd': float(np.std(list(seg_counts.values()), ddof=1)) if len(seg_counts) > 1 else 0.0,
        'duration_mean': float(np.mean(seg_durations)) if seg_durations else np.nan,
        'duration_median': float(np.median(seg_durations)) if seg_durations else np.nan,
        'duration_sd': float(np.std(seg_durations, ddof=1)) if len(seg_durations) > 1 else 0.0,
        'duration_min': float(np.min(seg_durations)) if seg_durations else np.nan,
        'duration_max': float(np.max(seg_durations)) if seg_durations else np.nan,
        'fraction_runs_with_sotc': float(np.mean([c > 0 for c in seg_counts.values()])),
    }

    # Detailed 7-panel diagnostic plot for the first run loaded
    first_run = df_all['run_id'].unique()[0]
    df_first = df_all[df_all['run_id'] == first_run]
    plot_full_diagnostics(
        df_first,
        first_run,
        os.path.join(summary_dir, 'full_diagnostics_example.png'),
    )

    # ------------------------------------------------------------------
    # H2: Diagnostic signatures
    # ------------------------------------------------------------------
    print("Starting H2 analysis")
    sotc_df = df_all[df_all['is_sustained_sotc_period']].dropna(subset=['speed_p','couple_p'])
    nonsotc_df = df_all[(~df_all['is_sustained_sotc_period']) & (df_all['time_step']>BURN_IN_STEPS)].dropna(subset=['speed_p','couple_p'])
    if not sotc_df.empty and not nonsotc_df.empty:
        u_s, p_s = mannwhitneyu(sotc_df['speed_p'], nonsotc_df['speed_p'], alternative='two-sided')
        u_c, p_c = mannwhitneyu(sotc_df['couple_p'], nonsotc_df['couple_p'], alternative='two-sided')
        sotc_counts = np.array([
            sotc_df['Q1_both_pos'].sum(),
            sotc_df['Q2_beta_neg_fcrit_pos'].sum(),
            sotc_df['Q3_both_neg'].sum(),
            sotc_df['Q4_beta_pos_fcrit_neg'].sum(),
        ])
        nonsotc_counts = np.array([
            nonsotc_df['Q1_both_pos'].sum(),
            nonsotc_df['Q2_beta_neg_fcrit_pos'].sum(),
            nonsotc_df['Q3_both_neg'].sum(),
            nonsotc_df['Q4_beta_pos_fcrit_neg'].sum(),
        ])
        chi2, p_chi2, _, _ = chi2_contingency(np.vstack([sotc_counts, nonsotc_counts]))
        sp_sotc = descr_stats(sotc_df['speed_p'])
        sp_non = descr_stats(nonsotc_df['speed_p'])
        cp_sotc = descr_stats(sotc_df['couple_p'])
        cp_non = descr_stats(nonsotc_df['couple_p'])
        summary['H2'] = {
            'speed': {
                'n_sotc': sp_sotc['n'],
                'n_nonsotc': sp_non['n'],
                'median_sotc': sp_sotc.get('median'),
                'iqr_sotc': sp_sotc.get('iqr'),
                'mean_sotc': sp_sotc.get('mean'),
                'sd_sotc': sp_sotc.get('sd'),
                'median_nonsotc': sp_non.get('median'),
                'iqr_nonsotc': sp_non.get('iqr'),
                'mean_nonsotc': sp_non.get('mean'),
                'sd_nonsotc': sp_non.get('sd'),
                'mannwhitney_U': float(u_s),
                'p': float(p_s),
                'cliffs_delta': float(cliffs_delta(sotc_df['speed_p'], nonsotc_df['speed_p'])),
            },
            'couple': {
                'n_sotc': cp_sotc['n'],
                'n_nonsotc': cp_non['n'],
                'median_sotc': cp_sotc.get('median'),
                'iqr_sotc': cp_sotc.get('iqr'),
                'mean_sotc': cp_sotc.get('mean'),
                'sd_sotc': cp_sotc.get('sd'),
                'median_nonsotc': cp_non.get('median'),
                'iqr_nonsotc': cp_non.get('iqr'),
                'mean_nonsotc': cp_non.get('mean'),
                'sd_nonsotc': cp_non.get('sd'),
                'mannwhitney_U': float(u_c),
                'p': float(p_c),
                'cliffs_delta': float(cliffs_delta(sotc_df['couple_p'], nonsotc_df['couple_p'])),
            },
            'quadrant_counts': {
                'SOTC': sotc_counts.astype(int).tolist(),
                'NonSOTC': nonsotc_counts.astype(int).tolist(),
                'chi2': float(chi2),
                'p': float(p_chi2),
            }
        }
        plt.figure(figsize=(12,5))
        plt.hist(sotc_df['speed_p'], bins=30, alpha=0.7, density=True, label='SOTC')
        plt.hist(nonsotc_df['speed_p'], bins=30, alpha=0.7, density=True, label='Non-SOTC')
        plt.xlabel('Speed_p')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(summary_dir, 'speed_distribution.png'), dpi=SAVEFIG_DPI)
        plt.close()
        plt.figure(figsize=(12,5))
        plt.hist(sotc_df['couple_p'], bins=30, alpha=0.7, density=True, label='SOTC', range=(-1,1))
        plt.hist(nonsotc_df['couple_p'], bins=30, alpha=0.7, density=True, label='Non-SOTC', range=(-1,1))
        plt.xlabel('Couple_p')
        plt.ylabel('Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(summary_dir, 'couple_distribution.png'), dpi=SAVEFIG_DPI)
        plt.close()

    # ------------------------------------------------------------------
    # H3: Precursors to large avalanches
    # ------------------------------------------------------------------
    print("Starting H3 analysis")
    large_events = df_all[(df_all['is_large_avalanche']) & (df_all['time_step'] > BURN_IN_STEPS)]
    prop_in_sotc = large_events['sotc_prev'].mean()

    sotc_events = large_events[large_events['sotc_prev']]
    nonsotc_events = large_events[~large_events['sotc_prev']]

    def collect_epochs(event_df):
        ep_g, ep_s, ep_c, ep_q = [], [], [], []
        for _, row in event_df.iterrows():
            run = row['run_id']
            t = row['time_step']
            sub = df_all[(df_all['run_id'] == run) & (df_all['time_step'] >= t - LOOKBACK_WINDOW_H3) & (df_all['time_step'] < t)]
            if len(sub) == LOOKBACK_WINDOW_H3:
                ep_g.append(sub['safety_margin_g'].values)
                ep_s.append(sub['speed_p'].values)
                ep_c.append(sub['couple_p'].values)
                ep_q.append(sub['prop_q4_roll20'].values)
        return ep_g, ep_s, ep_c, ep_q

    se_g_sotc, se_s_sotc, se_c_sotc, se_q_sotc = collect_epochs(sotc_events)
    se_g_non, se_s_non, se_c_non, se_q_non = collect_epochs(nonsotc_events)

    cont_table = np.array([
        [(df_all['sotc_prev'] & df_all['is_large_avalanche']).sum(), (df_all['sotc_prev'] & ~df_all['is_large_avalanche']).sum()],
        [(~df_all['sotc_prev'] & df_all['is_large_avalanche']).sum(), (~df_all['sotc_prev'] & ~df_all['is_large_avalanche']).sum()],
    ])
    chi2_la, p_la, _, _ = chi2_contingency(cont_table)
    risk_ratio = (cont_table[0,0] / cont_table[0].sum()) / (cont_table[1,0] / cont_table[1].sum()) if cont_table[0,0] and cont_table[1,0] else np.nan
    try:
        from statsmodels.stats.contingency_tables import Table2x2
        tbl = Table2x2(cont_table)
        rr_low, rr_high = tbl.riskratio_confint()
    except Exception:
        rr_low = rr_high = np.nan

    sea_data = {}
    for name, sotc_list, non_list in [
        ('g', se_g_sotc, se_g_non),
        ('speed', se_s_sotc, se_s_non),
        ('couple', se_c_sotc, se_c_non),
        ('prop_q4', se_q_sotc, se_q_non),
    ]:
        sotc_arr = np.array(sotc_list) if sotc_list else np.empty((0, LOOKBACK_WINDOW_H3))
        non_arr = np.array(non_list) if non_list else np.empty((0, LOOKBACK_WINDOW_H3))
        sea_data[name] = {
            'sotc_mean': np.nanmean(sotc_arr, axis=0).tolist() if sotc_arr.size else [],
            'non_sotc_mean': np.nanmean(non_arr, axis=0).tolist() if non_arr.size else [],
            'sotc_sd': np.nanstd(sotc_arr, axis=0).tolist() if sotc_arr.size else [],
            'non_sotc_sd': np.nanstd(non_arr, axis=0).tolist() if non_arr.size else [],
        }

    summary['H3'] = {
        'large_aval_from_sotc': int(len(sotc_events)),
        'large_aval_from_non_sotc': int(len(nonsotc_events)),
        'prop_large_from_sotc': float(prop_in_sotc) if not np.isnan(prop_in_sotc) else np.nan,
        'chi2': float(chi2_la),
        'p': float(p_la),
        'risk_ratio': float(risk_ratio),
        'risk_ratio_ci95': [float(rr_low), float(rr_high)],
        'sea_mean_traces': sea_data,
    }

    if se_g_sotc or se_g_non:
        x = np.arange(-LOOKBACK_WINDOW_H3, 0)
        plt.figure(figsize=(12, 10))
        for idx, (sotc_list, non_list, title) in enumerate([
            (se_g_sotc, se_g_non, 'G_p'),
            (se_s_sotc, se_s_non, 'Speed_p'),
            (se_c_sotc, se_c_non, 'Couple_p'),
            (se_q_sotc, se_q_non, 'Prop_Q4'),
        ]):
            plt.subplot(2, 2, idx + 1)
            if sotc_list:
                mean_sotc = np.nanmean(sotc_list, axis=0)
                plt.plot(x, mean_sotc, label='From SOTC', color='tab:red')
            if non_list:
                mean_non = np.nanmean(non_list, axis=0)
                plt.plot(x, mean_non, label='From Non-SOTC', color='tab:blue')
            plt.title(title)
            if title == 'G_p':
                plt.axhline(G_UPPER_SOTC_BOUND, color='r', ls=':')
                plt.axhline(G_LOWER_SOTC_BOUND, color='r', ls=':')
            plt.xlabel('Steps before large avalanche')
            plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(summary_dir, 'sea_comparison.png'), dpi=SAVEFIG_DPI)
        plt.close()

    # ------------------------------------------------------------------
    # H4: Pre-event diagnostics vs avalanche size
    # ------------------------------------------------------------------
    print("Starting H4 analysis")
    pre_stats = []
    for _, row in large_events.iterrows():
        run = row['run_id']
        t = row['time_step']
        aval_size = row['avalanche_size']
        sub = df_all[(df_all['run_id']==run) & (df_all['time_step']>=t-PRE_EVENT_WINDOW_H4) & (df_all['time_step']<t)]
        if sub.empty:
            continue
        pre_stats.append({
            'size': aval_size,
            'mean_speed': sub['speed_p'].mean(),
            'mean_couple': sub['couple_p'].mean(),
            'prop_q4': sub['Q4_beta_pos_fcrit_neg'].mean(),
        })
    if pre_stats:
        df_h4 = pd.DataFrame(pre_stats).dropna()
        if not df_h4.empty:
            summary['H4'] = {}
            plt.figure(figsize=(14,4))
            for idx, (col, lab) in enumerate([
                ('mean_speed', 'Mean pre-event speed'),
                ('mean_couple', 'Mean pre-event couple'),
                ('prop_q4', 'Mean pre-event prop Q4'),
            ]):
                x = df_h4[col]
                y = np.log10(df_h4['size'])
                if len(x) > 1:
                    res = linregress(x, y)
                    r = res.rvalue
                    ci = 1.96 * np.sqrt((1 - r**2) / (len(x) - 2)) if len(x) > 2 else np.nan
                    summary['H4'][col] = {
                        'n_events': int(len(x)),
                        'slope': float(res.slope),
                        'intercept': float(res.intercept),
                        'slope_se': float(res.stderr) if res.stderr else np.nan,
                        'r': float(r),
                        'r_ci95': [float(r - ci), float(r + ci)] if not np.isnan(ci) else [np.nan, np.nan],
                    }
                    plt.subplot(1,3,idx+1)
                    plt.scatter(x, y, alpha=0.6)
                    plt.plot(x, res.intercept + res.slope * x, color='red')
                    plt.title(f'r={r:.2f}')
                    plt.xlabel(lab)
                    if idx == 0:
                        plt.ylabel('log10(size)')
            plt.tight_layout()
            plt.savefig(os.path.join(summary_dir, 'h4_pre_event_correlations.png'), dpi=SAVEFIG_DPI)
            plt.close()

    # Advanced: fit power law exponent within each sustained SOTC segment
    segment_records = []
    for run, grp in df_all.groupby('run_id'):
        grp = grp.reset_index(drop=True)
        period_ids = (grp['is_sustained_sotc_period'] != grp['is_sustained_sotc_period'].shift()).cumsum()
        for pid, seg in grp.groupby(period_ids):
            if not seg['is_sustained_sotc_period'].iloc[0]:
                continue
            avals = seg.loc[seg['avalanche_size'] > 0, 'avalanche_size'].values
            if len(avals) < MIN_AVALS_FOR_ALPHA_FIT:
                continue
            alpha, D, xmin = fit_power_law_robust(avals)
            if np.isnan(alpha):
                continue
            record = {
                'run_id': run,
                'segment_id': pid,
                'alpha': alpha,
                'ks_D': D,
                'xmin': xmin,
                'num_avals': len(avals),
                'segment_mean_speed': seg['speed_p'].mean(),
                'segment_mean_couple': seg['couple_p'].mean(),
                'segment_mean_prop_q4': seg['Q4_beta_pos_fcrit_neg'].mean(),
                'segment_mean_g': seg['safety_margin_g'].mean(),
                'segment_duration': len(seg),
            }
            segment_records.append(record)

    if segment_records:
        df_segments = pd.DataFrame(segment_records)
        df_segments.to_csv(os.path.join(summary_dir, 'alpha_td_values.csv'), index=False)

        corrs = {}
        for col in ['segment_mean_speed', 'segment_mean_couple', 'segment_mean_prop_q4', 'segment_mean_g']:
            if df_segments[col].notna().sum() > 1:
                r, p = pearsonr(df_segments[col], df_segments['alpha'])
                corrs[col] = {'r': float(r), 'p': float(p)}

        if 'H4' not in summary:
            summary['H4'] = {}
        summary['H4']['alpha_correlations'] = corrs
        summary['avalanche_size_distribution'] = {
            'alpha_mean': float(df_segments['alpha'].mean()),
            'alpha_sd': float(df_segments['alpha'].std()),
            'alpha_min': float(df_segments['alpha'].min()),
            'alpha_max': float(df_segments['alpha'].max()),
            'ks_D_mean': float(df_segments['ks_D'].mean()),
            'ks_D_sd': float(df_segments['ks_D'].std()),
            'xmin_mean': float(df_segments['xmin'].mean()),
            'xmin_sd': float(df_segments['xmin'].std()),
            'xmin_min': float(df_segments['xmin'].min()),
            'xmin_max': float(df_segments['xmin'].max()),
            'segment_count': int(len(df_segments)),
        }

        # scatter plots
        plt.figure(figsize=(12,3))
        for idx, col in enumerate(['segment_mean_speed', 'segment_mean_couple', 'segment_mean_prop_q4', 'segment_mean_g']):
            if col in df_segments:
                plt.subplot(1,4,idx+1)
                plt.scatter(df_segments[col], df_segments['alpha'], alpha=0.7)
                if df_segments[col].notna().sum() > 1:
                    coeff = np.polyfit(df_segments[col], df_segments['alpha'], 1)
                    plt.plot(df_segments[col], np.polyval(coeff, df_segments[col]), color='red')
                plt.xlabel(col)
                if idx == 0:
                    plt.ylabel('alpha')
        plt.tight_layout()
        plt.savefig(os.path.join(summary_dir, 'alpha_td_correlations.png'), dpi=SAVEFIG_DPI)
        plt.close()

    # Combined avalanche distribution
    all_aval = df_all['avalanche_size']
    plt.figure(figsize=(8,6))
    bins = np.logspace(np.log10(max(1, all_aval[all_aval>0].min())), np.log10(max(1, all_aval.max())), 50)
    counts, edges = np.histogram(all_aval[all_aval>0], bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    plt.scatter(centers[counts>0], counts[counts>0], edgecolor='k')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Avalanche size')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(os.path.join(summary_dir, 'combined_avalanche_distribution.png'), dpi=SAVEFIG_DPI)
    plt.close()

    alpha_g, D_g, xmin_g = fit_power_law_robust(all_aval.values)

    def extended_stats(arr: np.ndarray) -> dict:
        arr = np.asarray(arr, dtype=float)
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            return {"n": 0}
        q1, median, q3 = np.percentile(arr, [25, 50, 75])
        return {
            "n": int(arr.size),
            "min": float(arr.min()),
            "q1": float(q1),
            "median": float(median),
            "q3": float(q3),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "mean": float(arr.mean()),
            "sd": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        }

    summary['avalanche_size_stats'] = {
        'all': extended_stats(all_aval[all_aval > 0]),
        'large': extended_stats(all_aval[all_aval >= LARGE_AVALANCHE_THRESH]),
        'powerlaw_global': {
            'alpha': float(alpha_g),
            'ks_D': float(D_g),
            'xmin': float(xmin_g),
        },
    }

    # Speed and couple distribution percentiles
    quantiles = [5, 25, 50, 75, 95]
    summary['speed_distribution'] = {
        'overall': [float(np.nanpercentile(df_all['speed_p'], q)) for q in quantiles],
        'sotc': [float(np.nanpercentile(sotc_df['speed_p'], q)) for q in quantiles] if not sotc_df.empty else [],
        'nonsotc': [float(np.nanpercentile(nonsotc_df['speed_p'], q)) for q in quantiles] if not nonsotc_df.empty else [],
    }
    summary['couple_distribution'] = {
        'overall': [float(np.nanpercentile(df_all['couple_p'], q)) for q in quantiles],
        'sotc': [float(np.nanpercentile(sotc_df['couple_p'], q)) for q in quantiles] if not sotc_df.empty else [],
        'nonsotc': [float(np.nanpercentile(nonsotc_df['couple_p'], q)) for q in quantiles] if not nonsotc_df.empty else [],
    }

    # Quadrant dynamics overall fractions and jaccard overlap
    quad_cols = ['Q1_both_pos','Q2_beta_neg_fcrit_pos','Q3_both_neg','Q4_beta_pos_fcrit_neg']
    summary['quadrant_dynamics'] = {
        'fractions': {col: float(df_all[col].mean()) for col in quad_cols},
    }
    jaccards = []
    arr = df_all[quad_cols].astype(bool)
    for run, grp in arr.groupby(df_all['run_id']):
        g = grp.values
        inter = (g[1:] & g[:-1]).sum(axis=1)
        union = (g[1:] | g[:-1]).sum(axis=1)
        jaccards.extend(inter / union)
    summary['quadrant_dynamics']['mean_jaccard'] = float(np.mean(jaccards)) if jaccards else np.nan

    trans_matrix = np.zeros((4,4), dtype=int)
    for run, grp in arr.groupby(df_all['run_id']):
        idx = np.argmax(grp.values, axis=1)
        for i in range(len(idx)-1):
            trans_matrix[idx[i], idx[i+1]] += 1

    summary['quadrant_dynamics']['transition_counts'] = trans_matrix.astype(int).tolist()

    with open(os.path.join(summary_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
