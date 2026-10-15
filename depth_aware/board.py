"""AprilGrid (6 x 6 AprilTag 36h11) detections as a ground-truth planar target.

The refrax-dataset sequences (gopro10_linear, gopro11_linear) film a flat Kalibr-style
AprilGrid in a tank. A perfect refraction correction turns the image of a plane into
a pinhole image of a plane, so the board corners in the corrected image must be
related to the board coordinates by a homography. The RMS homography transfer
residual of the corrected corners is therefore a direct, scale-free measure of how
well an image was corrected; it needs no reconstruction and no ground-truth depth.

Absolute depth is observable from the same residual, because the flat-port
distortion depends on Z / d0: the plane (normal, distance) that minimises it is the
board pose in metres, and the tag size it implies must be the same in every image.
"""
import glob
import os

import cv2
import numpy as np
from scipy.optimize import least_squares, minimize_scalar

GRID = 6
# Corner offsets of a tag in its own frame (OpenCV order: TL, TR, BR, BL).
CORNER_OFFS = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])


def make_detector():
    ar = cv2.aruco
    p = ar.DetectorParameters()
    p.adaptiveThreshWinSizeMin, p.adaptiveThreshWinSizeMax, p.adaptiveThreshWinSizeStep = 5, 63, 6
    p.cornerRefinementMethod = ar.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = 5
    p.errorCorrectionRate = 0.8
    p.polygonalApproxAccuracyRate = 0.05
    return ar.ArucoDetector(ar.getPredefinedDictionary(ar.DICT_APRILTAG_36h11), p)


def detect_dir(rgb_dir, cache, step=1):
    """{name: (ids (N,), corners (N, 4, 2) raw px)}; cached in an npz."""
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["det"].item()
    det = make_detector()
    out = {}
    for f in sorted(glob.glob(f"{rgb_dir}/*.png"))[::step]:
        g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        c, ids, _ = det.detectMarkers(g)
        if ids is None:
            continue
        ids = ids.ravel()
        keep = ids < GRID * GRID
        out[os.path.splitext(os.path.basename(f))[0]] = (ids[keep], np.array(c)[keep, 0])
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez(cache, det=np.array(out, dtype=object))
    return out


def board_coords(ids, rho, orient=0):
    """Board coordinates (units of the tag size) of the 4 corners of each tag.

    rho is the gap between tags relative to the tag size; orient (0..7) selects how
    the tag id maps onto the grid axes (swap and flips), which a homography absorbs
    except for the tag-corner offsets.
    """
    a, b = ids // GRID, ids % GRID
    if orient & 1:
        a, b = b, a
    centers = np.stack([a, b], 1).astype(float) * (1 + rho)
    offs = CORNER_OFFS.copy()
    if orient & 2:
        offs[:, 0] *= -1
    if orient & 4:
        offs[:, 1] *= -1
    return (centers[:, None, :] + offs[None]).reshape(-1, 2)


def homography_residual(board_xy, img_xy):
    """Per-point transfer residual (px) of the best DLT homography board -> image."""
    H, _ = cv2.findHomography(board_xy, img_xy, 0)
    if H is None:
        return np.full(len(img_xy), np.nan)
    p = cv2.perspectiveTransform(board_xy.reshape(-1, 1, 2), H).reshape(-1, 2)
    return np.linalg.norm(p - img_xy, axis=1)


def inlier_tags(ids, board_xy, img_xy, thresh=4.0, min_tags=8):
    """Boolean mask over tags whose 4 corners all fit a RANSAC homography.

    Removes misdetections (wrong corner order, false tags). Decided once per image,
    on a reference correction, and then used for every method compared.
    """
    if len(ids) < min_tags:
        return None
    H, inl = cv2.findHomography(board_xy, img_xy, cv2.RANSAC, thresh, maxIters=5000,
                                confidence=0.999)
    if H is None:
        return None
    ok = inl.ravel().reshape(-1, 4).all(1)
    p = cv2.perspectiveTransform(board_xy.reshape(-1, 1, 2), H).reshape(-1, 2)
    # second pass: refit on inliers and keep tags with all corners < thresh
    H, _ = cv2.findHomography(board_xy.reshape(-1, 4, 2)[ok].reshape(-1, 2),
                              img_xy.reshape(-1, 4, 2)[ok].reshape(-1, 2), 0)
    p = cv2.perspectiveTransform(board_xy.reshape(-1, 1, 2), H).reshape(-1, 2)
    ok = (np.linalg.norm(p - img_xy, axis=1).reshape(-1, 4) < thresh).all(1)
    return ok if ok.sum() >= min_tags else None


def rms(r):
    return float(np.sqrt(np.nanmean(np.asarray(r) ** 2)))


def plane_from_params(p):
    """Plane n . X = d from (tilt_x, tilt_y, d); n points along +Z for zero tilt."""
    n = np.array([np.tan(p[0]), np.tan(p[1]), 1.0])
    n /= np.linalg.norm(n)
    return n, p[2]


def fit_plane(C, raw_pts, board_xy, z_init):
    """Refractive plane fit: the camera-frame plane minimising the homography residual.

    Returns (n, d, rms_px, world points (N, 3)).
    """
    ideal = C.raw_to_ideal(raw_pts)

    def resid(p):
        n, d = plane_from_params(p)
        out = C.project_world(C.world_points_on_plane(ideal, n, d))
        return homography_residual(board_xy, out)

    best = None
    for z in z_init:
        r = least_squares(resid, [0.0, 0.0, z], x_scale=[0.1, 0.1, 0.1 * z],
                          bounds=([-1.2, -1.2, 0.05], [1.2, 1.2, 20.0]), method="trf")
        if best is None or r.cost < best.cost:
            best = r
    n, d = plane_from_params(best.x)
    Pw = C.world_points_on_plane(ideal, n, d)
    return n, d, rms(best.fun), Pw


def best_fronto_depth(C, raw_pts, board_xy, bounds=(0.1, 5.0)):
    """Single depth (fronto-parallel) per image minimising the homography residual."""
    f = lambda z: rms(homography_residual(board_xy, C.correct_points(raw_pts, z)))
    r = minimize_scalar(f, bounds=bounds, method="bounded", options={"xatol": 1e-4})
    return float(r.x), float(r.fun)
