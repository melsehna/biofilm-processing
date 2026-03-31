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
        Normalized grayscale frames for visualization.
    masks : ndarray, shape (H, W, T), bool
        Binary mask per frame.
    outPath : str
        Output .mp4 path.
    fps : int
        Frames per second.
    label : str or None
        Text label to burn into each frame.
    alpha : float
        Overlay blending weight (0 = no overlay, 1 = solid color).
    overlayColor : tuple of 3 ints
        BGR color for the mask overlay. Default is cyan (255, 255, 0) in BGR.
    """
    h, w, nFrames = displayStack.shape
    colorBgr = np.array(overlayColor, dtype=np.float32)
    bgWeight = 1.0 - alpha

    # write to a local temp file first, then move to outPath
    # cv2.VideoWriter can produce corrupt files on network/SMB mounts
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
        gray = displayStack[:, :, t]
        grayU8 = np.clip(gray * 255, 0, 255).astype(np.uint8)
        frame = cv2.cvtColor(grayU8, cv2.COLOR_GRAY2BGR)

        maskT = masks[:h, :w, t]
        if maskT.any():
            region = frame[maskT].astype(np.float32)
            frame[maskT] = (region * bgWeight + colorBgr * alpha).astype(np.uint8)

        if label:
            cv2.putText(
                frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2, cv2.LINE_AA,
            )

        video.write(frame)

    video.release()

    # move completed file to final destination
    try:
        shutil.move(tmpPath, outPath)
    except Exception:
        shutil.copy2(tmpPath, outPath)
        os.remove(tmpPath)
