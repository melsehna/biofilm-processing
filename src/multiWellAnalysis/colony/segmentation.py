import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, disk, binary_closing

# def segmentColonies(
#     rawImg,
#     mask,
#     minColonyArea_px=200,
#     closingRadius_px=1,
#     connectivity=2
# ):
#     mask = mask.astype(bool)
#     maskedRaw = rawImg * mask

#     binary = binary_fill_holes(mask)
#     # binary = binary_closing(binary, disk(closingRadius_px))
#     binary = binary_closing(binary, disk(2))
#     binary = remove_small_objects(binary, min_size=minColonyArea_px)

#     labels = label(binary, connectivity=connectivity)
#     props = regionprops(labels, intensity_image=maskedRaw)

#     return labels, props

def segmentColonies(
    rawImg,
    mask,
    minColonyArea_px=200,
    connectivity=2
):
    mask = mask.astype(bool)
    maskedRaw = rawImg * mask

    binary = binary_fill_holes(mask)
    binary = remove_small_objects(binary, min_size=minColonyArea_px)

    labels = label(binary, connectivity=connectivity)
    props = regionprops(labels, intensity_image=maskedRaw)

    return labels, props
