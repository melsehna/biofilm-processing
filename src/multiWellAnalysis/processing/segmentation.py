# multiWellAnalysis/segmentation.py
import numpy as np
import cv2

def compute_mask_inplace(stack, masks, fixedThresh):
    masks[:] = stack > float(fixedThresh)
    return masks

# vectorized version
def dust_correct_inplace(masks):
    firstOn = masks[..., 0]                  # t = 0
    everOff = np.any(~masks[..., 1:], axis=2)  # over time
    kill = firstOn & everOff
    masks[kill, :] = False                  # kill across all time
    return masks
