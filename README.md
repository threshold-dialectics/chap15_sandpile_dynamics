# Simulation Code for Chapter 15 of "Threshold Dialectics"

This repository contains the Python simulation and analysis code for **Experiment 15A** and **Experiment 15B**, as described in Chapter 15 ("*Simulating Threshold Dialectics with Sandpile Dynamics*") of the book *Threshold Dialectics: Understanding Complex Systems and Enabling Active Robustness* by Axel Pond.

This code provides a concrete, executable implementation of the concepts discussed in the book, exploring how the principles of Threshold Dialectics (TD) can be applied to the canonical Bak-Tang-Wiesenfeld (BTW) sandpile model to provide a mechanistic understanding of Self-Organized Criticality (SOC).

**Note:** The code for Experiment 15C is located in a different repository.

## Project Overview

As detailed in Chapter 15 of the book, this work uses the sandpile model as a microcosm to explore and validate TD principles in a non-biological, rule-based system.

*   **Experiment 15A: Core TD Mechanics and Diagnostics**
    This experiment operationalizes core TD concepts within the sandpile model. It demonstrates how TD lever proxies can be defined and how the core diagnostics ($\SpeedIndex_p, \CoupleIndex_p$) can capture the system's dynamic state and anticipate critical events (large avalanches).

*   **Experiment 15B: Self-Organized Tolerance Creep (SOTC)**
    This experiment provides a TD-based mechanistic explanation for the emergence of Self-Organized Criticality. It demonstrates how a simple, FEP-aligned "greedy" adaptation rule can drive the system into a globally critical state, characterized by a minimal safety margin—a phenomenon termed Self-Organized Tolerance Creep (SOTC).

### Key Concepts from Threshold Dialectics (TD)

A brief familiarity with these concepts from the book will aid in understanding the code:

*   **Adaptive Levers:** The core adaptive capacities of a system. The proxies used here are:
    *   **Policy Precision ($\betaLever_p$):** The toppling threshold, $k_{th}$. A higher $k_{th}$ reflects a more "greedy" or precise policy of accumulating stress before acting.
    *   **Energetic Slack ($\FEcrit^p$):** The total capacity of stable cells to absorb more grains before becoming unstable.
    *   **Perception Gain ($\gLever_p$):** The probability that an unstable cell will topple in a given time step.
*   **Tolerance Sheet ($\ThetaT$):** The system's dynamic capacity to withstand stress, calculated from the levers. A "breach" occurs when systemic strain exceeds this capacity.
*   **TD Diagnostics:**
    *   **Speed Index ($\SpeedIndex_p$):** The joint rate of change of the primary levers ($\betaLever_p, \FEcrit^p$), indicating the speed of structural drift.
    *   **Couple Index ($\CoupleIndex_p$):** The correlation between the lever velocities, indicating how their drifts are coordinated.
*   **Self-Organized Tolerance Creep (SOTC):** The TD reinterpretation of SOC, where a system's local, FEP-driven optimizations cause it to self-organize into a globally critical state with a chronically minimal safety margin ($G = \ThetaT - \text{Strain} \approx 0$).

## Repository Contents

This repository contains the following Python scripts:

*   "run_experiment_15A.py": A standalone script to run a single simulation of Experiment 15A. It generates diagnostic plots and a summary JSON file.
*   "run_experiment_15B.py": The core simulation logic for a single run of Experiment 15B. This script is called by the batch runner.
*   "run_experiment_15B_batch.py": The main script to execute the full Experiment 15B. It runs multiple simulations of "run_experiment_15B.py" with different random seeds and then invokes the analysis script.
*   "analyze_experiment_15B_results.py": A script to perform a comprehensive analysis of the batch results from Experiment 15B. It aggregates data, performs statistical tests, and generates summary plots and a JSON report.
*   "td_core.py": A library of core functions and classes (e.g., "SandpileBTW", derivative calculations) used by the experiment scripts.
*   "td_summary.py": A helper script used by "run_experiment_15A.py" to generate its summary JSON output.

## Setup and Installation

To run these simulations, you will need Python 3.8+ and the following packages. It is highly recommended to use a virtual environment.

1.  **Create and activate a virtual environment:**
    """bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use "venv\Scripts\activate"
    """

2.  **Install the required packages:**
    """bash
    pip install numpy pandas scipy matplotlib powerlaw statsmodels
    """

    Alternatively, you can create a "requirements.txt" file with the following content and install from it:
    """
    # requirements.txt
    numpy
    pandas
    scipy
    matplotlib
    powerlaw
    statsmodels
    """
    Then run: "pip install -r requirements.txt"

## Running the Experiments

All scripts should be run from the root of the repository.

### Experiment 15A

This is a single, self-contained simulation. To run it:

"""bash
python run_experiment_15A.py
"""

**Output:**
This will create a directory named "Experiment_15A_results/". Inside, you will find:
*   Several ".png" image files containing diagnostic plots (e.g., "td_diagnostics_timeseries_experiment15A.png").
*   A "summary.json" file with detailed quantitative results from the simulation run.

### Experiment 15B

This experiment is designed to be run as a batch to gather robust statistical data. The main entry point is "run_experiment_15B_batch.py".

To run the full batch experiment and subsequent analysis:

"""bash
python run_experiment_15B_batch.py
"""

**Process:**
1.  The script will first create a parent directory named "Experiment_15B_Batch_Results/".
2.  It will then loop "NUM_SEEDS" times (default is 10), executing "run_experiment_15B.py" for each run.
3.  Each run will generate its own results subdirectory (e.g., "Experiment_15B_Run0_results/", "Experiment_15B_Run1_results/", etc.) containing a detailed "sotc_simulation_log_... .csv" file and plots for that specific run.
4.  After all simulation runs are complete, the script will automatically call "analyze_experiment_15B_results.py".
5.  The analysis script will create a final summary directory named "_BatchAnalysis_Summary/" inside "Experiment_15B_Batch_Results/".

**Final Output:**
The "_BatchAnalysis_Summary/" directory will contain:
*   Aggregated analysis plots (e.g., "speed_distribution.png", "sea_comparison.png").
*   A detailed "summary.json" file containing the results of all statistical analyses across the batch runs.
*   An "alpha_td_values.csv" file with data on power-law fits for SOTC segments.

## Citation

If you use or refer to this code or the concepts from Threshold Dialectics, please cite the accompanying book:

@book{pond2025threshold,
  author    = {Axel Pond},
  title     = {Threshold Dialectics: Understanding Complex Systems and Enabling Active Robustness},
  year      = {2025},
  isbn      = {978-82-693862-2-6},
  publisher = {Amazon Kindle Direct Publishing},
  url       = {https://www.thresholddialectics.com},
  note      = {Code repository: \url{https://github.com/threshold-dialectics}}
}

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.