import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
import os
import pandas as pd

def save_biomass_curve(biomass, outdir, filename):
    """Save biomass curve and CSV."""
    frames = np.arange(1, len(biomass) + 1)
    df = pd.DataFrame({"Frame": frames, "Biomass": biomass})
    csv_path = os.path.join(outdir, f"{filename}_biomass.csv")
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(5, 3))
    plt.plot(frames, biomass, '-o', lw=1.5, color='steelblue')
    plt.xlabel("Frame")
    plt.ylabel("Biofilm Biomass (a.u.)")
    plt.title(f"Biomass over time – {filename}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{filename}_biomass_curve.pdf"))
    plt.close()
    print(f"   Saved biomass curve -> {csv_path}")


def save_peak_panel(raw_peak, processed_peak, peak_mask, overlay, segmentation, outdir, filename, peak_t):
    """Save a 5-panel diagnostic figure for the peak biomass frame."""
    import matplotlib.pyplot as plt
    from skimage import exposure
    import os

    fig, axes = plt.subplots(1, 5, figsize=(16, 4))
    axes[0].imshow(exposure.rescale_intensity(raw_peak, out_range=(0, 1)), cmap='gray')
    axes[0].set_title("Raw image")
    axes[1].imshow(exposure.rescale_intensity(processed_peak, out_range=(0, 1)), cmap='gray')
    axes[1].set_title("Processed")
    axes[2].imshow(peak_mask, cmap='gray')
    axes[2].set_title("Mask")
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay")
    axes[4].imshow(segmentation)
    axes[4].set_title("Segmentation (labeled)")

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    panel_path = os.path.join(outdir, f"{filename}_t{peak_t+1}_panel.pdf")
    plt.savefig(panel_path, dpi=200)
    plt.close(fig)
    print(f"   Saved diagnostic panel -> {panel_path}")
