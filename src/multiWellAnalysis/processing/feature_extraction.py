import numpy as np
import pandas as pd
import mahotas
from skimage.measure import regionprops_table, regionprops, label
from skimage.feature import local_binary_pattern
from skimage import exposure, util
from scipy.stats import entropy as shannon_entropy

# ----------------------------------------------
# Utility: entropy from values
# ----------------------------------------------
def _entropy_from_values(vals, bins=64):
    vals = np.asarray(vals)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    hist, _ = np.histogram(vals, bins=bins, range=(vals.min(), vals.max() + 1e-9))
    p = hist.astype(float)
    p /= p.sum() + 1e-12
    return shannon_entropy(p, base=2)

# ----------------------------------------------
# Fractal dimension (box-counting)
# ----------------------------------------------
def fractal_dimension_boxcount(mask):
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return np.nan
    h, w = mask.shape
    n = 1 << int(np.ceil(np.log2(max(h, w))))
    padded = np.zeros((n, n), dtype=bool)
    padded[:h, :w] = mask
    sizes = 2 ** np.arange(int(np.log2(n)), 0, -1)
    counts = []
    for s in sizes:
        view = util.view_as_blocks(padded, (s, s))
        nonempty = (view.reshape(view.shape[0], view.shape[1], -1).any(axis=2)).sum()
        counts.append(nonempty)
    sizes = np.array(sizes, dtype=float)
    counts = np.array(counts, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        coeffs = np.polyfit(np.log(1.0 / sizes), np.log(counts + 1e-9), 1)
    return coeffs[0] if np.isfinite(coeffs[0]) else np.nan

# ----------------------------------------------
# Haralick + Zernike feature extraction (mahotas)
# ----------------------------------------------
def compute_haralick(image, mask):
    img = exposure.rescale_intensity(image, out_range=(0, 255)).astype(np.uint8)
    img_masked = np.where(mask, img, 0)
    H = mahotas.features.haralick(img_masked, ignore_zeros=True, return_mean=True)
    names = [
        "haralick_asm", "haralick_contrast", "haralick_correlation", "haralick_variance",
        "haralick_idm", "haralick_sum_avg", "haralick_sum_var", "haralick_sum_entropy",
        "haralick_entropy", "haralick_diff_var", "haralick_diff_entropy",
        "haralick_meas_corr1", "haralick_meas_corr2"
    ]
    return dict(zip(names, H))

def compute_zernike(image, mask, radius=64, max_order=8):
    """Compute Zernike moments using mahotas."""
    from skimage.transform import resize
    patch = image.astype(float)
    patch = exposure.rescale_intensity(patch, out_range=(0, 1))
    # crop mask to bounding box
    ys, xs = np.where(mask)
    if ys.size == 0:
        return {f"zernike_{i}": np.nan for i in range(1, max_order + 1)}
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    patch = patch[y0:y1, x0:x1]
    mask = mask[y0:y1, x0:x1]
    patch = resize(patch, (radius * 2, radius * 2), anti_aliasing=True)
    mask = resize(mask.astype(float), (radius * 2, radius * 2), anti_aliasing=False)
    patch[mask < 0.5] = 0
    zm = mahotas.features.zernike_moments(patch, radius=radius, degree=max_order)
    return {f"zernike_{i}": float(v) for i, v in enumerate(zm[:max_order], 1)}

# ----------------------------------------------
# Peak frame finder
# ----------------------------------------------
def compute_peak_frame(biomass_curve):
    return int(np.nanargmax(biomass_curve))

# ----------------------------------------------
# Region feature extraction
# ----------------------------------------------
def extract_region_features(image, mask, label_prefix="biofilm"):
    labeled = label(mask)
    props = [
        'label', 'area', 'eccentricity', 'perimeter',
        'major_axis_length', 'minor_axis_length', 'mean_intensity',
        'solidity', 'bbox_area', 'extent', 'orientation',
        'feret_diameter_max', 'centroid', 'convex_area'
    ]
    df = pd.DataFrame(regionprops_table(labeled, intensity_image=image, properties=props))
    if df.empty:
        return df

    df.rename(columns={'centroid-0':'centroid_y','centroid-1':'centroid_x'}, inplace=True)
    df['circularity'] = (4 * np.pi * df['area']) / (df['perimeter'] ** 2 + 1e-8)
    df['aspect_ratio'] = df['major_axis_length'] / (df['minor_axis_length'] + 1e-8)
    df['compactness'] = (df['perimeter'] ** 2) / (4 * np.pi * df['area'] + 1e-8)
    df['convexity'] = df['area'] / (df['convex_area'] + 1e-8)
    df['label_type'] = label_prefix

    regs = regionprops(labeled, intensity_image=image)
    variances, stds, mins, maxs, entrs = [], [], [], [], []
    for rp in regs:
        vals = rp.intensity_image[rp.image]
        variances.append(np.var(vals))
        stds.append(np.std(vals))
        mins.append(np.min(vals))
        maxs.append(np.max(vals))
        entrs.append(_entropy_from_values(vals))
    df['intensity_var'] = variances
    df['intensity_std'] = stds
    df['intensity_min'] = mins
    df['intensity_max'] = maxs
    df['intensity_entropy'] = entrs

    img_rescaled = exposure.rescale_intensity(image, in_range='image', out_range=(0, 1))
    lbp = local_binary_pattern(img_rescaled, P=8, R=1, method='default')
    lbp_vals = lbp[mask]
    hist, _ = np.histogram(lbp_vals, bins=10, range=(0, 255), density=True)
    lbp_entropy = -(hist[hist > 0] * np.log2(hist[hist > 0])).sum()
    lbp_var = np.var(lbp_vals)
    for i, v in enumerate(hist):
        df[f'lbp_{i}'] = v
    df['lbp_var'] = lbp_var
    df['lbp_entropy'] = lbp_entropy

    H = compute_haralick(image, mask)
    for k, v in H.items():
        df[k] = v

    Z = compute_zernike(image, mask)
    for k, v in Z.items():
        df[k] = v

    df['fractal_dim'] = fractal_dimension_boxcount(mask)
    return df

# ----------------------------------------------
# Background feature extraction
# ----------------------------------------------
def extract_background_features(image, mask):
    inv_mask = ~mask
    return extract_region_features(image, inv_mask, label_prefix="background")

# ----------------------------------------------
# Summary feature aggregation
# ----------------------------------------------
def summarize_features(region_df):
    if region_df.empty:
        return pd.DataFrame()
    num_cols = region_df.select_dtypes(include=[np.number]).columns
    means = region_df[num_cols].mean().add_prefix('mean_')
    stds = region_df[num_cols].std().add_prefix('std_')
    return pd.concat([means, stds]).to_frame().T
