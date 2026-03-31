"""Overlay video generation for biofilm timelapse data."""

import os
import shutil
import tempfile
import numpy as np
import cv2


def writeOverlayVideo(
    displayStack,
    masks,
    outPath,
    fps=2,
    label=None,
    alpha=0.35,
    overlayColor=(255, 255, 0),
):
    """Write an overlay MP4 from a display stack and mask stack.

    Parameters
    ----------
    displayStack : ndarray, shape (H, W, T), float32 in [0, 1]
    masks : ndarray, shape (H, W, T), bool
    outPath : str
    fps : int
    label : str or None
    alpha : float
    overlayColor : tuple of 3 ints
        BGR color for the mask overlay.
    """
    h, w, nFrames = displayStack.shape
    bgWeight = np.float32(1.0 - alpha)

    # batch convert entire stack to uint8 BGR frames: (T, H, W, 3)
    grayAll = np.clip(displayStack * 255, 0, 255).astype(np.uint8)  # (H, W, T)
    frames = np.stack([grayAll, grayAll, grayAll], axis=-1)  # (H, W, T, 3)
    frames = np.moveaxis(frames, 2, 0)  # (T, H, W, 3)

    # batch apply mask overlay with vectorized numpy
    colorArr = np.array(overlayColor, dtype=np.float32) * alpha
    masksHW = masks[:h, :w, :]
    masksT = np.moveaxis(masksHW, 2, 0)  # (T, H, W)
    for t in range(nFrames):
        m = masksT[t]
        if m.any():
            region = frames[t][m].astype(np.float32)
            frames[t][m] = (region * bgWeight + colorArr).astype(np.uint8)

    # pre-render label once and composite (avoids per-frame putText)
    if label:
        labelOverlay = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(labelOverlay, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        labelMask = labelOverlay.any(axis=-1)
        for t in range(nFrames):
            frames[t][labelMask] = labelOverlay[labelMask]

    # write to local temp file to avoid corrupt files on network mounts
    tmpFd, tmpPath = tempfile.mkstemp(suffix='.mp4')
    os.close(tmpFd)

    video = None
    for codec in ['avc1', 'mp4v']:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video = cv2.VideoWriter(tmpPath, fourcc, fps, (w, h))
        if video.isOpened():
            break
        video.release()
        video = None
    if video is None:
        os.remove(tmpPath)
        return

    for t in range(nFrames):
        video.write(frames[t])

    video.release()

    try:
        shutil.move(tmpPath, outPath)
    except Exception:
        shutil.copy2(tmpPath, outPath)
        os.remove(tmpPath)
