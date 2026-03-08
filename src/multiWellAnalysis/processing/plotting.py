import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


def plotting_main(root_dir):
    """
    Extended plotting stage:
    - Scans both Julia ('Numerical data') and Python ('Numerical_data_py') outputs.
    - Creates:
        1️Per-plate summary biomass plots.
        2️Per-well biomass curves if '_biomass.csv' files exist.
    """
    print(f" Scanning for biomass CSVs in {root_dir}")

    plate_patterns = [
        os.path.join(root_dir, "**/Numerical data/*_BF_biomass.csv"),
        os.path.join(root_dir, "**/Numerical_data_py/*_BF_biomass.csv"),
    ]

    well_patterns = [
        os.path.join(root_dir, "**/Processed_images_py/*_biomass.csv"),
        os.path.join(root_dir, "**/Processed_images/*_biomass.csv"),
    ]

    plate_paths, well_paths = [], []
    for pattern in plate_patterns:
        plate_paths.extend(glob.glob(pattern, recursive=True))
    for pattern in well_patterns:
        well_paths.extend(glob.glob(pattern, recursive=True))

    if not plate_paths and not well_paths:
        print(" No biomass CSVs found.")
        return

    # Plate level summary plots
    for csv_path in plate_paths:
        try:
            df = pd.read_csv(csv_path)
            plate_name = os.path.basename(csv_path).replace("_BF_biomass.csv", "")
            out_dir = os.path.dirname(csv_path)
            out_plot = os.path.join(out_dir, f"{plate_name}_summary.pdf")

            # Transpose to match Julia’s column=well format
            df_t = df.T
            nframes = df_t.shape[1]

            plt.figure(figsize=(8, 6))
            for col in df_t.index:
                plt.plot(range(1, nframes + 1), df_t.loc[col], lw=1.2, label=col)
            plt.xlabel("Frame")
            plt.ylabel("Biomass (a.u.)")
            plt.title(f"Plate summary: {plate_name}")
            plt.legend(fontsize=6, loc='best', ncol=2)
            plt.tight_layout()
            plt.savefig(out_plot)
            plt.close()

            print(f" Saved plate summary plot: {out_plot}")
        except Exception as e:
            print(f" Failed to plot {csv_path}: {e}")

    # Per-well biomass curve
    for csv_path in well_paths:
        try:
            df = pd.read_csv(csv_path)
            if "Frame" not in df.columns or "Biomass" not in df.columns:
                continue
            well_name = os.path.basename(csv_path).replace("_biomass.csv", "")
            out_dir = os.path.dirname(csv_path)
            out_plot = os.path.join(out_dir, f"{well_name}_biomass_curve.pdf")

            plt.figure(figsize=(5, 3))
            plt.plot(df["Frame"], df["Biomass"], "-o", lw=1.5, color="teal")
            plt.xlabel("Frame")
            plt.ylabel("Biomass (a.u.)")
            plt.title(well_name)
            plt.tight_layout()
            plt.savefig(out_plot)
            plt.close()

            print(f" Saved per-well biomass plot: {out_plot}")
        except Exception as e:
            print(f" Failed to plot {csv_path}: {e}")
