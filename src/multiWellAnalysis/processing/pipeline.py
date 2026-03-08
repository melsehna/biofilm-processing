import os
import json
import glob
from .batch_runner import batch_run
from .plotting import plotting_main

def Pipeline(
    base_dir="/home/smellick/ImageLibrary/plates",
    replicate_csv="/home/smellick/ImageLibrary/ReplicatePositions.csv",
    config_path="/home/smellick/ImageLibrary/experiment_config.json"
):
    """
    Unified pipeline entry point — runs the full biofilm analysis:
    [1 Detects plates (via config or autodiscovery)
    [2 Runs full analysis and feature extraction via batch_runner
    [3 Runs plotting_main for per-well and per-plate summaries
    """
    print("Starting analysis stage...")

    if os.path.exists(config_path):
        print(f"Using global config: {config_path}")
        with open(config_path, "r") as f:
            config = json.load(f)
        plates = config.get("images_directory", [])
        if not plates:
            plates = sorted([os.path.dirname(p)
                             for p in glob.glob(os.path.join(base_dir, "**/metadata.csv"), recursive=True)])
        print(f"Found {len(plates)} plate(s) via config or autodiscovery.")
    else:
        print("No experiment_config.json found — autodetecting plates...")
        plates = sorted([os.path.dirname(p)
                         for p in glob.glob(os.path.join(base_dir, "**/metadata.csv"), recursive=True)])
        if not plates:
            print(f"No plates found in {base_dir}")
            return


    # Run the main batch analysis once for all plates
    # ---------------------------------------------------
    batch_run(config_path=config_path, replicate_csv=replicate_csv, force=False)

    # Generate all plotting summaries automatically
    # ---------------------------------------------------
    print("\nStarting plotting stage...")
    plotting_main(base_dir)
    print("\nPipeline complete.")

if __name__ == "__main__":
    pipeline()
