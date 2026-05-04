import cv2
import numpy as np
import logging
import os
import time
from typing import Optional, Tuple, List
from config import AppConfig

logger = logging.getLogger(__name__)

class CalibrationData:
    def __init__(self):
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.reprojection_error: float = float("inf")
        self.image_size: Optional[Tuple[int, int]] = None

    @property
    def is_calibrated(self) -> bool:
        return self.camera_matrix is not None

    @property
    def fx(self) -> Optional[float]:
        return float(self.camera_matrix[0, 0]) if self.is_calibrated else None

    @property
    def fy(self) -> Optional[float]:
        return float(self.camera_matrix[1, 1]) if self.is_calibrated else None

    @property
    def cx(self) -> Optional[float]:
        return float(self.camera_matrix[0, 2]) if self.is_calibrated else None

    @property
    def cy(self) -> Optional[float]:
        return float(self.camera_matrix[1, 2]) if self.is_calibrated else None

    def is_safe_for_undistort(self, max_reproj_px: float) -> bool:
        return self.is_calibrated and self.reprojection_error <= max_reproj_px

    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        if not self.is_calibrated:
            return frame
        return cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)

    def save(self, path: str):
        np.savez(
            path,
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            reprojection_error=np.array([self.reprojection_error]),
            image_size=np.array(self.image_size) if self.image_size else np.array([0, 0]),
        )
        logger.info(f"Calibration saved to {path} (reprojection error: {self.reprojection_error:.4f}px)")

    def load(self, path: str, reject_above_px: float = 1.5) -> bool:
        if not os.path.isfile(path):
            logger.info(f"No calibration file at {path} - running uncalibrated")
            return False
        try:
            data = np.load(path)
            mtx = data["camera_matrix"]
            dist = data["dist_coeffs"]
            err = float(data["reprojection_error"][0])
            imsz = data["image_size"]
            w, h = int(imsz[0]), int(imsz[1])

            fx = float(mtx[0, 0]); fy = float(mtx[1, 1])
            cx = float(mtx[0, 2]); cy = float(mtx[1, 2])
            ref_w = max(640, w)
            ref_h = max(480, h)
            fx_ok = (ref_w * 0.35) <= fx <= (ref_w * 3.5)
            fy_ok = (ref_h * 0.4) <= fy <= (ref_h * 4.5)
            cx_ok = (ref_w * 0.20) <= cx <= (ref_w * 0.80)
            cy_ok = (ref_h * 0.20) <= cy <= (ref_h * 0.80)
            d_flat = np.array(dist).flatten()
            dist_ok = bool(np.all(np.isfinite(d_flat))) and float(np.max(np.abs(d_flat))) <= 1.5

            if err > reject_above_px or not (fx_ok and fy_ok and cx_ok and cy_ok and dist_ok):
                logger.warning(
                    f"Calibration in {path} REJECTED "
                    f"(err={err:.3f}px fx={fx:.0f} fy={fy:.0f} cx={cx:.0f} cy={cy:.0f} dist_max={np.max(np.abs(d_flat)):.3f}). "
                    f"Running uncalibrated."
                )
                return False

            self.camera_matrix = mtx
            self.dist_coeffs = dist
            self.reprojection_error = err
            self.image_size = (w, h)
            logger.info(
                f"Calibration loaded (fx={fx:.1f}, fy={fy:.1f}, err={err:.4f}px)"
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to load calibration: {e}")
            return False

    def print_summary(self):
        if not self.is_calibrated:
            print("Calibration: NOT CALIBRATED")
            return
        print("=" * 60)
        print(" CAMERA CALIBRATION SUMMARY")
        print("=" * 60)
        print(f"  Image size:         {self.image_size}")
        print(f"  Focal length fx:    {self.fx:.2f} px")
        print(f"  Focal length fy:    {self.fy:.2f} px")
        print(f"  Principal point cx: {self.cx:.2f} px")
        print(f"  Principal point cy: {self.cy:.2f} px")
        if self.dist_coeffs is not None:
            d = np.array(self.dist_coeffs).flatten()
            keys = ["k1", "k2", "p1", "p2", "k3"]
            for i, k in enumerate(keys[: len(d)]):
                print(f"  Distortion {k}:       {d[i]:+.6f}")
        print(f"  Reprojection error: {self.reprojection_error:.4f} px")
        print("=" * 60)

def _angle_signature(corners: np.ndarray):
    pts = corners.reshape(-1, 2)
    cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    spread = float(np.std(pts[:, 0]) + np.std(pts[:, 1]))
    dx = float(np.std(pts[:, 0]))
    dy = float(np.std(pts[:, 1]))
    aspect = dx / max(1e-6, dy)
    return (cx, cy, aspect, spread)

def _angle_is_distinct(sig, history, frame_w, frame_h) -> bool:
    if not history:
        return True
    cx, cy, aspect, spread = sig
    for hcx, hcy, hasp, hsp in history:
        dx = abs(cx - hcx) / max(1.0, frame_w)
        dy = abs(cy - hcy) / max(1.0, frame_h)
        da = abs(aspect - hasp)
        ds = abs(spread - hsp) / max(1.0, frame_w)
        if dx < 0.10 and dy < 0.10 and da < 0.15 and ds < 0.05:
            return False
    return True

def _solve_calibration(obj_points, img_points, image_size):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)
    per_frame_errors: List[float] = []
    for i in range(len(obj_points)):
        projected, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i], mtx, dist)
        err = cv2.norm(img_points[i], projected, cv2.NORM_L2) / len(projected)
        per_frame_errors.append(float(err))
    mean_err = float(np.mean(per_frame_errors)) if per_frame_errors else float("inf")
    return mean_err, mtx, dist, rvecs, tvecs, per_frame_errors

def _draw_overlay(display, text_lines, found, primary_color):
    h, w = display.shape[:2]
    cv2.rectangle(display, (0, 0), (w, 90), (15, 15, 15), -1)
    cv2.rectangle(display, (0, h - 32), (w, h), (15, 15, 15), -1)
    for i, line in enumerate(text_lines):
        cv2.putText(
            display, line, (16, 28 + i * 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            primary_color if i == 0 else (210, 210, 210),
            2 if i == 0 else 1, cv2.LINE_AA,
        )
    cv2.putText(
        display, "C / SPACE = capture   R = reset   Q = finish/abort",
        (16, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA,
    )

def run_calibration_workflow(
    cap: cv2.VideoCapture,
    config: AppConfig,
    speech=None,
    window_name: str = "ManeNavSystem",
) -> Optional[CalibrationData]:
    cols, rows = config.checkerboard_size
    square_size = config.square_size_mm / 1000.0
    needed = max(config.min_calibration_frames, 1)
    max_attempts = 2

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    if speech is not None:
        try:
            speech.say(
                "Hold the chessboard in front of the camera. "
                "Press C to capture from at least four different angles. "
                "Press Q when done."
            )
        except Exception:
            pass

    best_calib: Optional[CalibrationData] = None

    for attempt in range(1, max_attempts + 1):
        obj_points: List[np.ndarray] = []
        img_points: List[np.ndarray] = []
        angle_history: List = []
        last_capture_ts = 0.0
        cooldown_s = 0.5
        attempt_aborted = False

        while len(obj_points) < needed:
            ret, frame = cap.read()
            if not ret:
                attempt_aborted = True
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(
                gray, (cols, rows),
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            display = frame.copy()
            corners_sub = None

            if found:
                corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(display, (cols, rows), corners_sub, found)
                sig = _angle_signature(corners_sub)
                fresh_angle = _angle_is_distinct(sig, angle_history, display.shape[1], display.shape[0])
                color = (0, 220, 80) if fresh_angle else (0, 180, 220)
                lines = [
                    f"FOUND  ({len(obj_points)}/{needed})  attempt {attempt}/{max_attempts}",
                    "Distinct angle - press C to capture" if fresh_angle
                    else "Move to a NEW angle (too similar to a captured one)",
                ]
            else:
                color = (0, 150, 255)
                lines = [
                    f"Searching {cols}x{rows} board...  ({len(obj_points)}/{needed})",
                    "Tilt / rotate the board so corners are clearly visible",
                ]

            _draw_overlay(display, lines, found, color)
            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                logger.info(f"Calibration aborted by user at {len(obj_points)} frames")
                attempt_aborted = True
                break
            if key == ord("r"):
                obj_points.clear()
                img_points.clear()
                angle_history.clear()
                logger.info("Calibration reset by user")
                continue

            now = time.time()
            if (
                (key == ord("c") or key == ord(" "))
                and found and corners_sub is not None
                and (now - last_capture_ts) > cooldown_s
            ):
                sig = _angle_signature(corners_sub)
                if not _angle_is_distinct(sig, angle_history, display.shape[1], display.shape[0]):
                    logger.info("Skipping capture - pose too similar")
                    continue
                obj_points.append(objp.copy())
                img_points.append(corners_sub)
                angle_history.append(sig)
                last_capture_ts = now
                logger.info(f"Captured calibration frame {len(obj_points)}/{needed}")
                if speech is not None:
                    try:
                        speech.say(f"Captured frame {len(obj_points)} of {needed}.")
                    except Exception:
                        pass

                if len(obj_points) >= max(8, needed // 2):
                    h_, w_ = frame.shape[:2]
                    try:
                        mean_err, _, _, _, _, per_errs = _solve_calibration(obj_points, img_points, (w_, h_))
                        if mean_err > config.max_calibration_reprojection_px * 2.0:
                            worst = int(np.argmax(per_errs))
                            obj_points.pop(worst)
                            img_points.pop(worst)
                            angle_history.pop(worst)
                            logger.info(f"Dropped frame #{worst} (err={per_errs[worst]:.3f}px)")
                    except cv2.error:
                        pass

        if attempt_aborted:
            break

        if len(obj_points) < max(10, config.min_calibration_frames // 2):
            logger.warning("Too few calibration frames - aborting attempt")
            if speech is not None:
                try:
                    speech.say("Calibration aborted. Not enough frames.")
                except Exception:
                    pass
            break

        h, w = frame.shape[:2]
        try:
            mean_error, mtx, dist, _, _, per_errs = _solve_calibration(obj_points, img_points, (w, h))
        except cv2.error as e:
            logger.error(f"Calibration solver failed: {e}")
            break

        if mean_error > config.max_calibration_reprojection_px and len(obj_points) >= 12:
            errs = np.array(per_errs)
            cutoff = float(np.percentile(errs, 75))
            keep = errs <= cutoff
            kept = sum(keep)
            if kept >= 8:
                obj_points = [op for op, k in zip(obj_points, keep) if k]
                img_points = [ip for ip, k in zip(img_points, keep) if k]
                logger.info(f"Trimmed worst-quartile frames (kept {kept}). Re-solving...")
                try:
                    mean_error, mtx, dist, _, _, per_errs = _solve_calibration(obj_points, img_points, (w, h))
                except cv2.error as e:
                    logger.warning(f"Resolve after trim failed: {e}")

        calib = CalibrationData()
        calib.camera_matrix = mtx
        calib.dist_coeffs = dist
        calib.reprojection_error = mean_error
        calib.image_size = (w, h)
        calib.print_summary()
        logger.info(f"Calibration attempt {attempt} complete. mean_err={mean_error:.4f}px frames={len(obj_points)}")

        if best_calib is None or mean_error < best_calib.reprojection_error:
            best_calib = calib

        if mean_error <= config.max_calibration_reprojection_px:
            if speech is not None:
                try:
                    speech.say("Calibration complete. Measurement accuracy improved.")
                except Exception:
                    pass
            break

        logger.warning(
            f"Calibration mean error {mean_error:.3f}px > threshold {config.max_calibration_reprojection_px:.3f}px"
        )
        if attempt < max_attempts:
            if speech is not None:
                try:
                    speech.say("Calibration error too high. Let us try again with new angles.")
                except Exception:
                    pass
            logger.info("Auto-repeating calibration with fresh frames...")
        else:
            if speech is not None:
                try:
                    speech.say(
                        f"Calibration finished but quality is low. "
                        f"Reprojection error {best_calib.reprojection_error:.2f} pixels."
                    )
                except Exception:
                    pass

    return best_calib
