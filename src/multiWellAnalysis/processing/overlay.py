"""Overlay video generation for biofilm timelapse data.

Core function: write_overlay_video() takes a display stack and mask stack
and writes an MP4 with cyan mask overlay and optional text label.
"""

import numpy as np
import cv2


def write_overlay_video(
    display_stack,
    masks,
    out_path,
    fps=2,
    label=None,
    alpha=0.35,
    overlay_color=(255, 255, 0),
):
    """Write an overlay MP4 from a display stack and mask stack.

    Parameters
    ----------
    display_stack : ndarray, shape (H, W, T), float32 in [0, 1]
        Normalized grayscale frames for visualization.
    masks : ndarray, shape (H, W, T), bool
        Binary mask per frame.
    out_path : str
        Output .mp4 path.
    fps : int
        Frames per second.
    label : str or None
        Text label to burn into each frame (e.g. "mutant  plate-well").
    alpha : float
        Overlay blending weight (0 = no overlay, 1 = solid color).
    overlay_color : tuple of 3 ints
        BGR color for the mask overlay. Default is cyan (255, 255, 0) in BGR.
    """
    h, w, nframes = display_stack.shape
    color_bgr = np.array(overlay_color, dtype=np.float32)
    bg_weight = 1.0 - alpha

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for t in range(nframes):
        gray = display_stack[:, :, t]
        gray_u8 = np.clip(gray * 255, 0, 255).astype(np.uint8)
        frame = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)

        mask_t = masks[:h, :w, t]
        if mask_t.any():
            region = frame[mask_t].astype(np.float32)
            frame[mask_t] = (region * bg_weight + color_bgr * alpha).astype(np.uint8)

        if label:
            cv2.putText(
                frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255), 2, cv2.LINE_AA,
            )

        video.write(frame)

    video.release()
