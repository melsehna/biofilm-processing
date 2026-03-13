# segmentation.py — OpenCV-accelerated version
# Changes from original:
#   - scipy.ndimage.binary_fill_holes -> cv2.floodFill from border
#   - skimage.morphology.remove_small_objects -> cv2.connectedComponentsWithStats + area filter
#   - skimage.measure.label -> cv2.connectedComponents

import numpy as np
import cv2


def compute_mask_inplace(stack, masks, fixedThresh):
    masks[:] = stack > float(fixedThresh)
    return masks


# vectorized version — pure numpy, no change needed
def dust_correct_inplace(masks):
    firstOn = masks[..., 0]                    # t = 0
    everOff = np.any(~masks[..., 1:], axis=2)  # over time
    kill = firstOn & everOff
    masks[kill, :] = False                     # kill across all time
    return masks


def _binary_fill_holes(mask):
    """Fill holes using cv2.floodFill from border (equivalent to binary_fill_holes)."""
    mask_u8 = mask.astype(np.uint8) * 255

    # Flood fill from top-left corner on the inverted image
    inv = cv2.bitwise_not(mask_u8)
    h, w = inv.shape
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(inv, flood_mask, (0, 0), 255)

    # Invert back: pixels that were NOT reached by flood = holes -> fill them
    filled = mask_u8 | cv2.bitwise_not(inv)
    return filled > 0


def _remove_small_objects(mask, min_size):
    """Remove connected components smaller than min_size pixels."""
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_u8, connectivity=8
    )

    # Vectorized: build a lookup table of which labels to keep
    keep = np.zeros(num_labels, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_size
    return keep[labels]


def segmentColonies(
    rawImg,
    mask,
    minColonyArea_px=200,
    connectivity=2
):
    mask = mask.astype(bool)
    maskedRaw = rawImg * mask

    binary = _binary_fill_holes(mask)
    binary = _remove_small_objects(binary, minColonyArea_px)

    # cv2 connected components (connectivity 8 = skimage connectivity 2)
    cv_connectivity = 8 if connectivity == 2 else 4
    mask_u8 = binary.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(mask_u8, connectivity=cv_connectivity)

    # Build regionprops-compatible list of dicts for downstream use
    props = []
    for lbl in range(1, num_labels):
        region_mask = labels == lbl
        area = int(np.sum(region_mask))
        intensity_vals = maskedRaw[region_mask]
        props.append({
            'label': lbl,
            'area': area,
            'mean_intensity': float(np.mean(intensity_vals)) if area > 0 else 0.0,
            'bbox': cv2.boundingRect(region_mask.astype(np.uint8)),
        })

    return labels, props
