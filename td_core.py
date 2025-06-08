# td_core.py

import os
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from typing import Sequence, Tuple, Set

class SandpileBTW:
    """Bak–Tang–Wiesenfeld sandpile model with adjustable k_th and p_topple."""

    def __init__(self, grid_size: Tuple[int, int] = (50, 50), k_th: int = 4, p_topple: float = 1.0):
        self.grid_size = grid_size
        self.grid = np.zeros(grid_size, dtype=int)
        self.k_th = k_th
        self.p_topple = p_topple
        self.time_step_counter = 0
        self.total_grains_lost = 0
        self.last_event_scar_coords: Set[Tuple[int, int]] = set() # New attribute

    def add_grain(self, pos: Tuple[int, int] = None) -> None:
        if pos is None:
            pos = (np.random.randint(0, self.grid_size[0]), np.random.randint(0, self.grid_size[1]))
        self.grid[pos] += 1
        self.time_step_counter += 1

    def get_unstable_coords(self) -> np.ndarray:
        return np.argwhere(self.grid >= self.k_th)

    def topple_and_relax(self) -> int: # Signature remains -> int
        current_event_avalanche_size = 0
        # --- For scar tracking ---
        # Clear scar coords for this new cascade event at the beginning of the call
        # This assumes topple_and_relax is called once per "event" (grain add + full relaxation)
        _current_cascade_scar_coords: Set[Tuple[int, int]] = set()
        # --- End scar tracking init ---

        unstable_coords = self.get_unstable_coords()
        max_iterations = self.grid_size[0] * self.grid_size[1] * 100
        relaxation_iterations = 0

        while unstable_coords.shape[0] > 0:
            relaxation_iterations += 1
            if relaxation_iterations > max_iterations:
                break

            coords_to_topple_this_sub_step = []
            for r_unstable, c_unstable in unstable_coords:
                if np.random.rand() < self.p_topple:
                    coords_to_topple_this_sub_step.append((r_unstable, c_unstable))
            
            if not coords_to_topple_this_sub_step:
                break

            for r, c in coords_to_topple_this_sub_step:
                if self.grid[r, c] >= self.k_th:
                    current_event_avalanche_size += 1
                    _current_cascade_scar_coords.add((r, c)) # Add to internal set for this cascade

                    grains_to_distribute = self.k_th
                    self.grid[r, c] -= grains_to_distribute
                    
                    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    for i in range(min(grains_to_distribute, len(dirs))):
                        dr, dc = dirs[i]
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < self.grid_size[0] and 0 <= cc < self.grid_size[1]:
                            self.grid[rr, cc] += 1
                        else:
                            self.total_grains_lost += 1
            
            unstable_coords = self.get_unstable_coords()
        
        # --- At the end, update the instance attribute ---
        self.last_event_scar_coords = _current_cascade_scar_coords
        # --- End scar tracking update ---
        
        return current_event_avalanche_size


def calculate_raw_instantaneous_strain_p(grid: np.ndarray, k_th: int) -> float:
    unstable_mask = grid >= k_th
    if np.sum(unstable_mask) == 0:
        return 0.0
    return float(np.sum(grid[unstable_mask] - (k_th - 1)))


def calculate_energetic_slack_p(grid: np.ndarray, k_th: int) -> float:
    stable_mask = grid < k_th
    slack = np.sum(k_th - grid[stable_mask])
    return float(slack)


def calculate_tolerance_sheet_p(
    g_lever_p_val: float,
    beta_lever_p_val: float,
    f_crit_p_val: float,
    w_g: float,
    w_beta: float,
    w_fcrit: float,
    C_p: float = 1.0,
    scaling: float = 1.0,
) -> float:
    g_eff = max(g_lever_p_val, 1e-9)
    beta_eff = max(beta_lever_p_val, 1e-9)
    fcrit_eff = max(f_crit_p_val, 1e-9)
    term_g = g_eff ** w_g if w_g > 0 else 1.0
    term_beta = beta_eff ** w_beta if w_beta > 0 else 1.0
    term_fcrit = fcrit_eff ** w_fcrit if w_fcrit > 0 else 1.0
    return scaling * C_p * term_g * term_beta * term_fcrit


def calculate_derivatives_savgol(timeseries_arr: Sequence[float], window_length: int = 5, polyorder: int = 2) -> np.ndarray:
    if len(timeseries_arr) < window_length:
        return np.full_like(timeseries_arr, np.nan, dtype=float)
    wl = window_length if window_length % 2 != 0 else window_length + 1
    if len(timeseries_arr) < wl:
        return np.full_like(timeseries_arr, np.nan, dtype=float)
    derivatives = np.full_like(timeseries_arr, np.nan, dtype=float)
    half_wl = wl // 2
    padded_ts = np.pad(timeseries_arr, (half_wl, half_wl), mode="edge")
    try:
        deriv_padded = savgol_filter(padded_ts, window_length=wl, polyorder=polyorder, deriv=1, delta=1.0)
        derivatives = deriv_padded[half_wl:-half_wl]
    except ValueError:
        pass
    return derivatives


# --- Simple Generic Plotting Utilities ---

def plot_timeseries(df: pd.DataFrame, columns: Sequence[str], labels: Sequence[str], title: str, filename: str, results_dir: str) -> None:
    plt.figure(figsize=(10, 4))
    for col, lab in zip(columns, labels):
        plt.plot(df['time_step'], df[col], label=lab)
    plt.xlabel('Time Step')
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, filename))
    plt.close()


def plot_histogram(data: Sequence[float], bins: Sequence[float], title: str, xlabel: str, filename: str, results_dir: str, log: bool = False) -> None:
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=bins, edgecolor='black', log=log)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, filename))
    plt.close()


# --- Simple Analysis Helpers ---

def jaccard_index(series_a: pd.Series, series_b: pd.Series) -> float:
    intersection = np.sum(series_a & series_b)
    union = np.sum(series_a | series_b)
    return intersection / union if union > 0 else np.nan

