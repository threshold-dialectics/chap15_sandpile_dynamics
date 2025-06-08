# run_experiment_15B_batch.py
import subprocess
import sys
from pathlib import Path

NUM_SEEDS = 10
BASE_EXPERIMENT_NAME = "Experiment_15B"
PARENT_RESULTS_DIR = "Experiment_15B_Batch_Results"

ROOT = Path(__file__).resolve().parent           # repo root
RUN_SCRIPT = ROOT / "run_experiment_15B.py"         # or ..._final.py if that is the real file
ANALYSE_SCRIPT = ROOT / "analyze_experiment_15B_results.py"

def main() -> None:
    results_dir = ROOT / PARENT_RESULTS_DIR
    results_dir.mkdir(exist_ok=True)

    for run_id in range(NUM_SEEDS):
        subprocess.run(
            [
                sys.executable,
                str(RUN_SCRIPT),
                "--experiment_name", BASE_EXPERIMENT_NAME,   # <— no suffix here
                "--run_id",     str(run_id),
                "--parent_dir", str(results_dir),
            ],
            check=True,
            cwd=ROOT,
        )

    # optional aggregated analysis
    try:
        subprocess.run(
            [sys.executable, str(ANALYSE_SCRIPT), str(results_dir)],
            check=True,
            cwd=ROOT,
        )
    except subprocess.CalledProcessError as exc:
        print(f"Batch analysis failed: {exc}")

if __name__ == "__main__":
    main()
