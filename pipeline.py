"""Camera capture pipeline for SENTINEL tremor analysis.

This module keeps the hand detection approach from ``tremor_detector.py``:
MediaPipe tracks full 21-point hands, then the recorder converts those
landmarks into the ``hand_data`` shape consumed by
``tremor_analysis.analyze_tremor``. It supports both classic
``mp.solutions.hands`` installs and newer MediaPipe Tasks installs.

OAK-D Lite support is built around DepthAI socket detection so the app does not
depend on hard-coded CAM_A/CAM_B/CAM_C assumptions when the device reports a
different board-socket enum.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Iterator

import numpy as np

cv2 = None
mp = None
dai = None


LANDMARK_INDEXES = {
    "wrist": 0,
    "thumb_tip": 4,
    "index_tip": 8,
    "middle_tip": 12,
    "ring_tip": 16,
    "pinky_tip": 20,
    "index_mcp": 5,
    "middle_mcp": 9,
    "pinky_mcp": 17,
}

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),           # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # index
    (5, 9), (9, 10), (10, 11), (11, 12),      # middle
    (9, 13), (13, 14), (14, 15), (15, 16),    # ring
    (13, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (0, 17),                                  # palm
)
 


DEFAULT_HAND_LANDMARKER_MODEL_CANDIDATES = (
    Path("hand_landmarker.task"),
    Path("models/hand_landmarker.task"),
)


@dataclass
class CameraFrame:
    color: np.ndarray
    depth: np.ndarray | None = None
    timestamp: float | None = None
    stream_fresh: dict[str, bool] | None = None


def load_runtime_dependencies() -> None:
    global cv2, mp
    if cv2 is not None and mp is not None:
        return
    try:
        import cv2 as cv2_module
        import mediapipe as mp_module
    except ImportError as exc:
        raise RuntimeError(
            f"Missing runtime dependency: {exc.name}. Install opencv-python and mediapipe."
        ) from exc

    cv2 = cv2_module
    mp = mp_module


def load_depthai_dependency() -> None:
    global dai
    if dai is not None:
        return
    try:
        import depthai as dai_module
    except ImportError as exc:
        raise RuntimeError("DepthAI is not installed. Install the depthai package.") from exc
    dai = dai_module


def _socket(*names: str) -> Any:
    load_depthai_dependency()
    for name in names:
        if hasattr(dai.CameraBoardSocket, name):
            return getattr(dai.CameraBoardSocket, name)
    joined = ", ".join(names)
    raise RuntimeError(f"DepthAI does not expose any of these camera sockets: {joined}")


def _socket_sort_key(socket: Any) -> int:
    socket_name = getattr(socket, "name", str(socket))
    order = {
        "CAM_A": 0,
        "RGB": 0,
        "CAM_B": 1,
        "LEFT": 1,
        "CAM_C": 2,
        "RIGHT": 2,
    }
    if socket_name in order:
        return order[socket_name]
    try:
        return int(socket)
    except (TypeError, ValueError):
        return 99


def _sensor_type_names(feature: Any) -> set[str]:
    return {getattr(sensor_type, "name", str(sensor_type)) for sensor_type in feature.supportedTypes}


def detect_oak_camera_sockets(device: Any) -> tuple[Any, Any, Any]:
    """Return ``(rgb, left, right)`` sockets for OAK-D/OAK-D Lite devices."""
    fallback = (_socket("CAM_A", "RGB"), _socket("CAM_B", "LEFT"), _socket("CAM_C", "RIGHT"))
    try:
        features = device.getConnectedCameraFeatures()
    except Exception:
        return fallback

    color_sockets = []
    mono_sockets = []
    for feature in features:
        type_names = _sensor_type_names(feature)
        if "COLOR" in type_names:
            color_sockets.append(feature.socket)
        if "MONO" in type_names:
            mono_sockets.append(feature.socket)

    if not color_sockets and not mono_sockets:
        return fallback
    if not color_sockets:
        raise RuntimeError("No OAK color camera detected.")
    if len(mono_sockets) < 2:
        detected = ", ".join(
            f"{getattr(feature.socket, 'name', feature.socket)}:{feature.name}:{sorted(_sensor_type_names(feature))}"
            for feature in features
        )
        raise RuntimeError(
            "OAK stereo depth requires two mono cameras, but fewer than two were detected. "
            f"Detected cameras: {detected or 'none'}"
        )

    mono_sockets = sorted(mono_sockets, key=_socket_sort_key)
    return color_sockets[0], mono_sockets[0], mono_sockets[1]


def detect_oak_color_socket(device: Any) -> Any:
    """Return the OAK color socket without requiring stereo mono cameras."""
    fallback = _socket("CAM_A", "RGB")
    try:
        features = device.getConnectedCameraFeatures()
    except Exception:
        return fallback

    color_sockets = [
        feature.socket
        for feature in features
        if "COLOR" in _sensor_type_names(feature)
    ]
    if color_sockets:
        return color_sockets[0]
    if not features:
        return fallback
    detected = ", ".join(
        f"{getattr(feature.socket, 'name', feature.socket)}:{feature.name}:{sorted(_sensor_type_names(feature))}"
        for feature in features
    )
    raise RuntimeError(f"No OAK color camera detected. Detected cameras: {detected}")


def _mono_resolution() -> Any:
    return getattr(
        dai.MonoCameraProperties.SensorResolution,
        "THE_480_P",
        dai.MonoCameraProperties.SensorResolution.THE_400_P,
    )


def _configure_xlink_out(node: Any, stream_name: str) -> None:
    node.setStreamName(stream_name)
    node.input.setBlocking(False)
    node.input.setQueueSize(1)


def create_oak_d_lite_pipeline(
    *,
    fps: int = 30,
    rgb_socket: Any | None = None,
    left_socket: Any | None = None,
    right_socket: Any | None = None,
    color_size: tuple[int, int] = (1280, 720),
    include_depth: bool = True,
) -> Any:
    """Build a DepthAI pipeline for OAK-D Lite RGB plus aligned stereo depth."""
    load_depthai_dependency()
    rgb_socket = _socket("CAM_A", "RGB") if rgb_socket is None else rgb_socket

    pipeline = dai.Pipeline()
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    rgb_out = pipeline.create(dai.node.XLinkOut)
    _configure_xlink_out(rgb_out, "rgb")

    cam_rgb.setBoardSocket(rgb_socket)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setFps(fps)
    cam_rgb.setIspScale(2, 3)
    cam_rgb.setPreviewSize(*color_size)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.isp.link(rgb_out.input)

    if include_depth:
        left_socket = _socket("CAM_B", "LEFT") if left_socket is None else left_socket
        right_socket = _socket("CAM_C", "RIGHT") if right_socket is None else right_socket
        mono_left = pipeline.create(dai.node.MonoCamera)
        mono_right = pipeline.create(dai.node.MonoCamera)
        stereo = pipeline.create(dai.node.StereoDepth)
        depth_out = pipeline.create(dai.node.XLinkOut)
        _configure_xlink_out(depth_out, "depth")

        mono_left.setResolution(_mono_resolution())
        mono_right.setResolution(_mono_resolution())
        mono_left.setBoardSocket(left_socket)
        mono_right.setBoardSocket(right_socket)
        mono_left.setFps(fps)
        mono_right.setFps(fps)

        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setLeftRightCheck(True)
        # Subpixel mode + HIGH_DENSITY + LR-check overruns the OAK-D Lite's
        # Myriad X memory budget; the pipeline fails to boot and the device
        # ends up stuck BOOTED, throwing X_LINK_INSUFFICIENT_PERMISSIONS on
        # the next launch. Subpixel buys finer depth precision, which we
        # don't need for landmark-scale tremor analysis.
        stereo.setDepthAlign(rgb_socket)

        mono_left.out.link(stereo.left)
        mono_right.out.link(stereo.right)
        stereo.depth.link(depth_out.input)

    return pipeline


class OpenCvFrameSource:
    def __init__(self, camera_index: int = 0, fps: int = 30) -> None:
        load_runtime_dependencies()
        self.capture = cv2.VideoCapture(camera_index)
        # 640x480 is the native resolution most laptop webcams hit at full FPS.
        # 1280x720 forces many cameras down to 15fps, and larger frames make
        # MediaPipe significantly slower — hand detection accuracy doesn't
        # meaningfully improve above 640x480 for tremor analysis.
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FPS, fps)

    def is_opened(self) -> bool:
        return bool(self.capture.isOpened())

    def read(self) -> tuple[bool, CameraFrame | None]:
        ok, frame = self.capture.read()
        if not ok:
            return False, None
        return True, CameraFrame(color=frame, timestamp=time.perf_counter())

    def release(self) -> None:
        self.capture.release()


class OakDFrameSource:
    def __init__(self, fps: int = 30, include_depth: bool = True) -> None:
        load_depthai_dependency()
        self.include_depth = include_depth
        self.device = dai.Device()
        left_socket = None
        right_socket = None
        if include_depth:
            try:
                rgb_socket, left_socket, right_socket = detect_oak_camera_sockets(self.device)
            except RuntimeError as exc:
                print(f"{exc} Falling back to OAK RGB-only capture.")
                self.include_depth = False
                include_depth = False
                rgb_socket = detect_oak_color_socket(self.device)
        else:
            rgb_socket = detect_oak_color_socket(self.device)
        self.rgb_socket = rgb_socket
        pipeline = create_oak_d_lite_pipeline(
            fps=fps,
            rgb_socket=rgb_socket,
            left_socket=left_socket,
            right_socket=right_socket,
            include_depth=include_depth,
        )
        self.device.startPipeline(pipeline)
        self.rgb_queue = self.device.getOutputQueue("rgb", maxSize=1, blocking=False)
        self.depth_queue = (
            self.device.getOutputQueue("depth", maxSize=1, blocking=False)
            if include_depth
            else None
        )
        self._last_depth_frame: np.ndarray | None = None
        self._intrinsics_by_size: dict[tuple[int, int], np.ndarray | None] = {}

    def is_opened(self) -> bool:
        return True

    @staticmethod
    def _latest_packet(queue: Any, *, block: bool = False) -> Any | None:
        packet = queue.get() if block else queue.tryGet()
        if packet is None:
            return None
        while True:
            next_packet = queue.tryGet()
            if next_packet is None:
                return packet
            packet = next_packet

    def read(self) -> tuple[bool, CameraFrame | None]:
        rgb_packet = self._latest_packet(self.rgb_queue, block=True)
        if rgb_packet is None:
            return False, None
        depth_packet = (
            self._latest_packet(self.depth_queue)
            if self.depth_queue is not None
            else None
        )
        if depth_packet is not None:
            self._last_depth_frame = depth_packet.getFrame()
        return True, CameraFrame(
            color=rgb_packet.getCvFrame(),
            depth=self._last_depth_frame,
            timestamp=time.perf_counter(),
            stream_fresh={"rgb": True, "depth": depth_packet is not None},
        )

    def get_rgb_intrinsics(self, width: int, height: int) -> np.ndarray | None:
        key = (width, height)
        if key in self._intrinsics_by_size:
            return self._intrinsics_by_size[key]
        try:
            intrinsics = self.device.readCalibration().getCameraIntrinsics(
                self.rgb_socket,
                width,
                height,
            )
            matrix = np.asarray(intrinsics, dtype=np.float64)
        except Exception:
            matrix = None
        self._intrinsics_by_size[key] = matrix
        return matrix

    def release(self) -> None:
        self.device.close()


def open_frame_source(source: str, *, camera_index: int = 0, fps: int = 30) -> Any:
    if source == "oak":
        return OakDFrameSource(fps=fps, include_depth=True)
    if source == "oak-rgb":
        return OakDFrameSource(fps=fps, include_depth=False)
    if source == "webcam":
        return OpenCvFrameSource(camera_index=camera_index, fps=fps)
    raise ValueError(f"Unsupported camera source: {source}")


class SolutionsHandsDetector:
    def __init__(
        self,
        *,
        max_num_hands: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        model_complexity: int = 0,
    ) -> None:
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_complexity = model_complexity
        self._hands = None

    def __enter__(self):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_num_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        return self

    def process(self, rgb_frame: np.ndarray) -> Any:
        if self._hands is None:
            raise RuntimeError("MediaPipe Hands detector is not initialized.")
        return self._hands.process(rgb_frame)

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._hands is not None:
            self._hands.close()


class TasksHandDetector:
    def __init__(
        self,
        *,
        max_num_hands: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        model_path: str | Path | None,
    ) -> None:
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.model_path = _resolve_hand_landmarker_model(model_path)
        self._landmarker = None

    def __enter__(self):
        if self.model_path is None:
            candidates = ", ".join(str(path) for path in DEFAULT_HAND_LANDMARKER_MODEL_CANDIDATES)
            raise RuntimeError(
                "This MediaPipe install does not expose mp.solutions, so pipeline.py "
                "must use MediaPipe Tasks. Provide a hand landmarker model with "
                f"--model PATH. Checked: {candidates}"
            )

        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe Tasks is not available, and mp.solutions is missing."
            ) from exc

        base_options = mp_python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=self.max_num_hands,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_tracking_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        return self

    def process(self, rgb_frame: np.ndarray) -> Any:
        if self._landmarker is None:
            raise RuntimeError("MediaPipe Tasks hand detector is not initialized.")
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        return self._landmarker.detect(image)

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._landmarker is not None:
            self._landmarker.close()


def _resolve_hand_landmarker_model(model_path: str | Path | None) -> Path | None:
    if model_path is not None:
        resolved = Path(model_path).expanduser()
        if not resolved.exists():
            raise RuntimeError(f"Hand landmarker model was not found: {resolved}")
        return resolved

    for candidate in DEFAULT_HAND_LANDMARKER_MODEL_CANDIDATES:
        resolved = candidate.expanduser()
        if resolved.exists():
            return resolved
    return None


def create_hand_detector(
    *,
    max_num_hands: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    model_path: str | Path | None = None,
    model_complexity: int = 0,
) -> SolutionsHandsDetector | TasksHandDetector:
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
        return SolutionsHandsDetector(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=model_complexity,
        )
    return TasksHandDetector(
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
        model_path=model_path,
    )


def _handedness_label_and_score(handedness: Any) -> tuple[str | None, float]:
    if hasattr(handedness, "classification"):
        if not handedness.classification:
            return None, 0.0
        head = handedness.classification[0]
        label = (
            getattr(head, "label", None)
            or getattr(head, "category_name", None)
            or getattr(head, "display_name", None)
        )
        return label.lower() if label else None, float(head.score)
    if not handedness:
        return None, 0.0
    head = handedness[0]
    label = (
        getattr(head, "label", None)
        or getattr(head, "category_name", None)
        or getattr(head, "display_name", None)
    )
    return label.lower() if label else None, float(head.score)


def hand_label_in_patient_anatomy(raw_label: str | None, mirrored: bool) -> str | None:
    """Convert MediaPipe's image-space handedness to the patient's anatomical side."""
    if raw_label is None:
        return None
    if mirrored:
        return raw_label
    return "right" if raw_label == "left" else "left"


def select_hands(results: Any, *, mirrored: bool, hand_filter: str) -> list[tuple[str, Any]]:
    landmark_groups = getattr(results, "multi_hand_landmarks", None)
    handedness_groups = getattr(results, "multi_handedness", None)
    if landmark_groups is None:
        landmark_groups = getattr(results, "hand_landmarks", None)
        handedness_groups = getattr(results, "handedness", None)
    if not landmark_groups:
        return []

    labelled: list[tuple[str, float, Any]] = []
    for idx, hand_landmarks in enumerate(landmark_groups):
        handedness = (
            handedness_groups[idx]
            if handedness_groups is not None and idx < len(handedness_groups)
            else None
        )
        raw_label, score = _handedness_label_and_score(handedness)
        label = hand_label_in_patient_anatomy(raw_label, mirrored) or "hand"
        labelled.append((label, score, hand_landmarks))

    if hand_filter == "auto":
        labelled.sort(key=lambda entry: entry[1], reverse=True)
        label, _, landmarks = labelled[0]
        return [(label, landmarks)]

    if hand_filter in ("left", "right"):
        matches = [entry for entry in labelled if entry[0] == hand_filter]
        matches.sort(key=lambda entry: entry[1], reverse=True)
        return [(label, landmarks) for label, _, landmarks in matches[:1]]

    by_label: dict[str, tuple[float, Any]] = {}
    for label, score, landmarks in labelled:
        if label not in by_label or score > by_label[label][0]:
            by_label[label] = (score, landmarks)
    ordered = sorted(by_label.items(), key=lambda item: (item[0] != "left", item[0] != "right", item[0]))
    return [(label, landmarks) for label, (_, landmarks) in ordered]


def sample_depth_mm(
    depth_frame: np.ndarray | None,
    normalized_x: float,
    normalized_y: float,
    *,
    roi_radius: int = 3,
    min_depth_mm: float = 200.0,
    max_depth_mm: float = 5000.0,
) -> float | None:
    if depth_frame is None:
        return None

    height, width = depth_frame.shape[:2]
    px = int(np.clip(normalized_x * width, 0, width - 1))
    py = int(np.clip(normalized_y * height, 0, height - 1))
    x0 = max(0, px - roi_radius)
    x1 = min(width, px + roi_radius + 1)
    y0 = max(0, py - roi_radius)
    y1 = min(height, py + roi_radius + 1)
    patch = depth_frame[y0:y1, x0:x1].astype(np.float64)
    valid = patch[(patch >= min_depth_mm) & (patch <= max_depth_mm)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def _project_pixel_to_mm(
    pixel_x: float,
    pixel_y: float,
    depth_mm: float,
    intrinsics: np.ndarray | None,
) -> tuple[float, float, float]:
    if intrinsics is None or intrinsics.shape != (3, 3):
        return pixel_x, pixel_y, depth_mm
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    if min(abs(fx), abs(fy)) < 1e-6:
        return pixel_x, pixel_y, depth_mm
    x_mm = (pixel_x - cx) * depth_mm / fx
    y_mm = (pixel_y - cy) * depth_mm / fy
    return x_mm, y_mm, depth_mm


def hand_landmarks_to_xyz(
    hand_landmarks: Any,
    color_shape: tuple[int, ...],
    *,
    depth_frame: np.ndarray | None = None,
    intrinsics: np.ndarray | None = None,
) -> tuple[np.ndarray, float, str]:
    """Convert MediaPipe landmarks into a ``(21, 3)`` hand array."""
    height, width = color_shape[:2]
    landmarks = hand_landmarks.landmark if hasattr(hand_landmarks, "landmark") else hand_landmarks
    depths = [
        sample_depth_mm(depth_frame, float(landmark.x), float(landmark.y))
        for landmark in landmarks
    ]
    valid_depths = [depth for depth in depths if depth is not None]
    default_depth = float(np.median(valid_depths)) if valid_depths else None

    output = np.zeros((len(landmarks), 3), dtype=np.float64)
    depth_hits = 0
    for idx, landmark in enumerate(landmarks):
        pixel_x = float(landmark.x) * width
        pixel_y = float(landmark.y) * height
        depth_mm = depths[idx] if depths[idx] is not None else default_depth
        if depth_mm is None:
            # Store normalized coords (0-1) so downstream amplitude is in a
            # consistent unit. analyze_tremor converts to mm via hand-size calibration.
            output[idx] = (float(landmark.x), float(landmark.y), float(landmark.z))
            # Store normalized coords (0-1) so downstream amplitude is in a
            # consistent unit. analyze_tremor converts to mm via hand-size calibration.
            output[idx] = (float(landmark.x), float(landmark.y), float(landmark.z))
            continue
        if depths[idx] is not None:
            depth_hits += 1
        output[idx] = _project_pixel_to_mm(pixel_x, pixel_y, depth_mm, intrinsics)

    if default_depth is None:
        return output, 0.5, "image_px"

    quality = float(np.clip(0.5 + 0.5 * (depth_hits / max(len(landmarks), 1)), 0.0, 1.0))
    return output, quality, "mm"

def draw_hand_overlay(
    frame: np.ndarray,
    selected_hands: list[tuple[str, Any]],
    *,
    color_right: tuple[int, int, int] = (37, 99, 235),   # blue
    color_left: tuple[int, int, int] = (217, 119, 6),    # amber
) -> np.ndarray:
    """Draw landmark points and skeleton edges on a BGR frame in place.
 
    Returns the same frame for convenience.
    """
    height, width = frame.shape[:2]
    for label, hand_landmarks in selected_hands:
        landmarks = (
            hand_landmarks.landmark
            if hasattr(hand_landmarks, "landmark")
            else hand_landmarks
        )
        color = color_right if label == "right" else color_left
        points = [
            (int(lm.x * width), int(lm.y * height)) for lm in landmarks
        ]
        for a, b in HAND_CONNECTIONS:
            if a < len(points) and b < len(points):
                cv2.line(frame, points[a], points[b], color, 2, cv2.LINE_AA)
        for px, py in points:
            cv2.circle(frame, (px, py), 4, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def capture_hand_data_streaming(
    *,
    duration_seconds: float = 30.0,
    source: str = "oak",
    hand: str = "both",
    camera_index: int = 0,
    fps: int = 30,
    mirror: bool | None = None,
    hand_landmarker_model: str | Path | None = None,
    min_detection_confidence: float | None = None,
    min_tracking_confidence: float = 0.5,
    draw_overlay: bool = True,
) -> Iterator[tuple[np.ndarray, float, dict[str, Any] | None]]:
    """Streaming version of capture_hand_data.
 
    Yields (preview_frame_bgr, elapsed_seconds, final_hand_data_or_None) on
    every iteration. ``final_hand_data_or_None`` is None for every frame
    except the last, where it carries the same dict shape as
    ``capture_hand_data``'s return value. The caller can stop early by
    closing the generator (``gen.close()``).
    """
    load_runtime_dependencies()
    if hand not in {"auto", "left", "right", "both"}:
        raise ValueError("hand must be one of: auto, left, right, both")

    # Laptop webcams are lower quality than OAK-D; 0.7 is too strict and causes
    # frequent detection drops which starve the FFT of frames. 0.5 keeps most
    # real hands while still rejecting background false positives.
    if min_detection_confidence is None:
        min_detection_confidence = 0.5 if source == "webcam" else 0.7

    frame_source = open_frame_source(source, camera_index=camera_index, fps=fps)
    if not frame_source.is_opened():
        raise RuntimeError(f"Could not open camera source: {source}")
 
    mirror_frames = source == "webcam" if mirror is None else mirror
    max_num_hands = 1 if hand in {"auto", "left", "right"} else 2
    samples: dict[str, list[np.ndarray]] = {"right": [], "left": []}
    timestamps: dict[str, list[float]] = {"right": [], "left": []}
    qualities: dict[str, list[float]] = {"right": [], "left": []}
    observed_units: set[str] = set()
 
    start = time.perf_counter()
    try:
        with create_hand_detector(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_path=hand_landmarker_model,
            # OAK-D streams high-quality frames where the heavier model pays off.
            # Webcam runs model_complexity=0 (lightweight) to keep frame rate smooth.
            model_complexity=0 if source == "webcam" else 1,
        ) as hands:
            while True:
                elapsed = time.perf_counter() - start
                if elapsed >= duration_seconds:
                    break
 
                ok, camera_frame = frame_source.read()
                if not ok or camera_frame is None:
                    break
                frame = camera_frame.color
                depth_frame = camera_frame.depth
                if mirror_frames:
                    frame = cv2.flip(frame, 1)
                    if depth_frame is not None:
                        depth_frame = cv2.flip(depth_frame, 1)
 
                intrinsics = None
                if hasattr(frame_source, "get_rgb_intrinsics"):
                    height, width = frame.shape[:2]
                    intrinsics = frame_source.get_rgb_intrinsics(width, height)
 
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = hands.process(rgb)
                rgb.flags.writeable = True
 
                selected = select_hands(results, mirrored=mirror_frames, hand_filter=hand)
                sample_elapsed = (
                    camera_frame.timestamp - start
                    if camera_frame.timestamp is not None
                    else elapsed
                )
                for label, landmarks in selected:
                    side = label if label in {"right", "left"} else "right"
                    xyz, quality, units = hand_landmarks_to_xyz(
                        landmarks,
                        frame.shape,
                        depth_frame=depth_frame,
                        intrinsics=intrinsics,
                    )
                    if xyz.shape != (21, 3):
                        continue
                    samples[side].append(xyz)
                    timestamps[side].append(float(sample_elapsed))
                    qualities[side].append(quality)
                    observed_units.add(units)
 
                preview = frame.copy() if draw_overlay else frame
                if draw_overlay:
                    draw_hand_overlay(preview, selected)
 
                yield preview, elapsed, None
    finally:
        frame_source.release()
 
    right_ts = np.asarray(timestamps["right"], dtype=np.float64)
    left_ts = np.asarray(timestamps["left"], dtype=np.float64)
    primary_ts = right_ts if right_ts.size >= left_ts.size else left_ts
    sample_rate = _infer_sample_rate(primary_ts, fps)
    units = "mm" if observed_units == {"mm"} else ("image_px" if observed_units else "unknown")
 
    final = {
        "right": np.stack(samples["right"]) if samples["right"] else _empty_hand_array(),
        "left": np.stack(samples["left"]) if samples["left"] else _empty_hand_array(),
        "right_timestamps": right_ts,
        "left_timestamps": left_ts,
        "right_quality": np.asarray(qualities["right"], dtype=np.float64),
        "left_quality": np.asarray(qualities["left"], dtype=np.float64),
        "timestamps": primary_ts,
        "sample_rate": sample_rate,
        "metadata": {
            "source": source,
            "hand": hand,
            "units": units,
            "right_samples": len(samples["right"]),
            "left_samples": len(samples["left"]),
        },
    }
    # Final frame slot: yield a dummy black frame paired with the data so the
    # UI can detect completion via the third tuple element.
    yield np.zeros((1, 1, 3), dtype=np.uint8), duration_seconds, final





def _empty_hand_array() -> np.ndarray:
    return np.empty((0, 21, 3), dtype=np.float64)


def _infer_sample_rate(timestamps: np.ndarray, fallback: int) -> float:
    if timestamps.size < 2:
        return float(fallback)
    diffs = np.diff(timestamps)
    valid = diffs[diffs > 1e-6]
    if valid.size == 0:
        return float(fallback)
    return float(1.0 / np.median(valid))


def capture_hand_data(
    *,
    duration_seconds: float = 30.0,
    source: str = "oak",
    hand: str = "both",
    camera_index: int = 0,
    fps: int = 30,
    mirror: bool | None = None,
    hand_landmarker_model: str | Path | None = None,
    min_detection_confidence: float = 0.7,
    min_tracking_confidence: float = 0.7,
) -> dict[str, Any]:
    """Capture live hand landmarks for ``tremor_analysis.analyze_tremor``.

    ``source="oak"`` uses RGB plus aligned stereo depth from an OAK-D Lite.
    ``source="oak-rgb"`` and ``source="webcam"`` still track hands, but their
    amplitudes are image-space estimates because no depth frame is available.
    """
    load_runtime_dependencies()
    if hand not in {"auto", "left", "right", "both"}:
        raise ValueError("hand must be one of: auto, left, right, both")

    frame_source = open_frame_source(source, camera_index=camera_index, fps=fps)
    if not frame_source.is_opened():
        raise RuntimeError(f"Could not open camera source: {source}")

    mirror_frames = source == "webcam" if mirror is None else mirror
    max_num_hands = 1 if hand in {"auto", "left", "right"} else 2
    samples: dict[str, list[np.ndarray]] = {"right": [], "left": []}
    timestamps: dict[str, list[float]] = {"right": [], "left": []}
    qualities: dict[str, list[float]] = {"right": [], "left": []}
    observed_units: set[str] = set()

    start = time.perf_counter()
    try:
        with create_hand_detector(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_path=hand_landmarker_model,
            model_complexity=0 if source == "webcam" else 1,
        ) as hands:
            while (time.perf_counter() - start) < duration_seconds:
                ok, camera_frame = frame_source.read()
                if not ok or camera_frame is None:
                    break
                frame = camera_frame.color
                depth_frame = camera_frame.depth
                if mirror_frames:
                    frame = cv2.flip(frame, 1)
                    if depth_frame is not None:
                        depth_frame = cv2.flip(depth_frame, 1)

                intrinsics = None
                if hasattr(frame_source, "get_rgb_intrinsics"):
                    height, width = frame.shape[:2]
                    intrinsics = frame_source.get_rgb_intrinsics(width, height)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = hands.process(rgb)
                rgb.flags.writeable = True

                selected = select_hands(results, mirrored=mirror_frames, hand_filter=hand)
                elapsed = (
                    camera_frame.timestamp - start
                    if camera_frame.timestamp is not None
                    else time.perf_counter() - start
                )
                for label, landmarks in selected:
                    side = label if label in {"right", "left"} else "right"
                    xyz, quality, units = hand_landmarks_to_xyz(
                        landmarks,
                        frame.shape,
                        depth_frame=depth_frame,
                        intrinsics=intrinsics,
                    )
                    if xyz.shape != (21, 3):
                        continue
                    samples[side].append(xyz)
                    timestamps[side].append(float(elapsed))
                    qualities[side].append(quality)
                    observed_units.add(units)
    finally:
        frame_source.release()

    right_ts = np.asarray(timestamps["right"], dtype=np.float64)
    left_ts = np.asarray(timestamps["left"], dtype=np.float64)
    primary_ts = right_ts if right_ts.size >= left_ts.size else left_ts
    sample_rate = _infer_sample_rate(primary_ts, fps)
    units = "mm" if observed_units == {"mm"} else ("image_px" if observed_units else "unknown")

    return {
        "right": np.stack(samples["right"]) if samples["right"] else _empty_hand_array(),
        "left": np.stack(samples["left"]) if samples["left"] else _empty_hand_array(),
        "right_timestamps": right_ts,
        "left_timestamps": left_ts,
        "right_quality": np.asarray(qualities["right"], dtype=np.float64),
        "left_quality": np.asarray(qualities["left"], dtype=np.float64),
        "timestamps": primary_ts,
        "sample_rate": sample_rate,
        "metadata": {
            "source": source,
            "hand": hand,
            "units": units,
            "right_samples": len(samples["right"]),
            "left_samples": len(samples["left"]),
        },
    }


def write_hand_data_csv(hand_data: dict[str, Any], output_path: str | Path, landmark_index: int = 8) -> None:
    """Write a compact CSV for inspection or archival."""
    output_path = Path(output_path)
    rows: list[dict[str, Any]] = []
    for side in ("right", "left"):
        hand_array = np.asarray(hand_data.get(side, _empty_hand_array()))
        hand_timestamps = np.asarray(hand_data.get(f"{side}_timestamps", []), dtype=np.float64)
        for idx, landmarks in enumerate(hand_array):
            timestamp = hand_timestamps[idx] if idx < hand_timestamps.size else idx
            point = landmarks[landmark_index]
            rows.append(
                {
                    "hand": side,
                    "timestamp": round(float(timestamp), 4),
                    "landmark": landmark_index,
                    "x": round(float(point[0]), 4),
                    "y": round(float(point[1]), 4),
                    "z": round(float(point[2]), 4),
                }
            )

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["hand", "timestamp", "landmark", "x", "y", "z"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture OAK-D Lite hand landmarks for tremor analysis.")
    parser.add_argument("--source", choices=("oak", "oak-rgb", "webcam"), default="oak")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index for --source webcam.")
    parser.add_argument("--duration", type=float, default=30.0, help="Recording duration in seconds.")
    parser.add_argument("--fps", type=int, default=30, help="Requested camera FPS.")
    parser.add_argument("--hand", choices=("auto", "left", "right", "both"), default="both")
    parser.add_argument("--output", default="hand_xyz.csv", help="CSV output path.")
    parser.add_argument(
        "--model",
        default=None,
        help="Path to hand_landmarker.task when using MediaPipe Tasks.",
    )
    parser.add_argument("--analyze", action="store_true", help="Print tremor_analysis features after capture.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(f"Recording {args.source} hand landmarks for {args.duration:g} seconds...")
    hand_data = capture_hand_data(
        duration_seconds=args.duration,
        source=args.source,
        hand=args.hand,
        camera_index=args.camera,
        fps=args.fps,
        hand_landmarker_model=args.model,
    )
    write_hand_data_csv(hand_data, args.output)
    metadata = hand_data["metadata"]
    print(
        f"Saved {metadata['right_samples']} right-hand and {metadata['left_samples']} "
        f"left-hand samples to {args.output}."
    )

    if args.analyze:
        from dataclasses import asdict

        from tremor_analysis import analyze_tremor

        features = analyze_tremor(hand_data)
        print(json.dumps(asdict(features), indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# ── Facial capture (hypomimia mode) ───────────────────────────────────────────

DEFAULT_FACE_LANDMARKER_MODEL_CANDIDATES = (
    Path("face_landmarker.task"),
    Path("models/face_landmarker.task"),
)

FACE_LANDMARKER_DOWNLOAD_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

FACE_OVAL_IDXS = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
    288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
    150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]
FACE_KEY_IDXS = [61, 291, 117, 346]   # mouth corners + cheeks


def _resolve_face_landmarker_model() -> Path | None:
    """Return path to face_landmarker.task, downloading it if needed."""
    for candidate in DEFAULT_FACE_LANDMARKER_MODEL_CANDIDATES:
        resolved = candidate.expanduser()
        if resolved.exists():
            return resolved

    # Not found locally — try to download next to hand_landmarker.task
    try:
        import urllib.request
        dest = Path("face_landmarker.task")
        print(f"[pipeline] Downloading face landmarker model to {dest} ...")
        urllib.request.urlretrieve(FACE_LANDMARKER_DOWNLOAD_URL, dest)
        if dest.exists():
            print("[pipeline] Download complete.")
            return dest
    except Exception as exc:
        print(f"[pipeline] Could not download face model: {exc}")

    return None


class _SolutionsFaceMesh:
    """Thin wrapper around mp.solutions.face_mesh for use in capture loop."""

    def __init__(self) -> None:
        self._mesh = None

    def __enter__(self):
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return self

    def landmarks_from_rgb(self, rgb: np.ndarray) -> np.ndarray | None:
        """Return (N_lm, 3) float32 array or None if no face detected."""
        rgb.flags.writeable = False
        results = self._mesh.process(rgb)
        rgb.flags.writeable = True
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        return None

    def __exit__(self, *_) -> None:
        if self._mesh is not None:
            self._mesh.close()


class _TasksFaceLandmarker:
    """Thin wrapper around mediapipe.tasks FaceLandmarker."""

    def __init__(self, model_path: Path) -> None:
        self._model_path = model_path
        self._landmarker = None

    def __enter__(self):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        base = mp_python.BaseOptions(model_asset_path=str(self._model_path))
        opts = vision.FaceLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(opts)
        return self

    def landmarks_from_rgb(self, rgb: np.ndarray) -> np.ndarray | None:
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(image)
        if result.face_landmarks:
            lm = result.face_landmarks[0]
            return np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        return None

    def __exit__(self, *_) -> None:
        if self._landmarker is not None:
            self._landmarker.close()


def _create_face_detector():
    """Return whichever face detector the installed MediaPipe supports."""
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
        return _SolutionsFaceMesh()

    model_path = _resolve_face_landmarker_model()
    if model_path is None:
        raise RuntimeError(
            "This MediaPipe install does not have mp.solutions (Tasks-only build). "
            "A face_landmarker.task model file is required but could not be found or "
            "downloaded. Place face_landmarker.task next to hand_landmarker.task and retry. "
            f"Download URL: {FACE_LANDMARKER_DOWNLOAD_URL}"
        )
    return _TasksFaceLandmarker(model_path)


def capture_face_data_streaming(
    *,
    duration_seconds: float = 10.0,
    source: str = "webcam",
    camera_index: int = 0,
    fps: int = 30,
    mirror: bool = True,
    draw_overlay: bool = True,
) -> "Iterator[tuple[np.ndarray, float, np.ndarray | None]]":
    """Streaming face-landmark capture for hypomimia analysis.

    Yields (preview_frame_bgr, elapsed_seconds, face_landmarks_or_None) on
    every camera frame.  ``face_landmarks_or_None`` is None on every frame
    except the last, where it carries the full (N_frames, 478, 3) array of
    all detected landmark frames collected during the session.

    Supports both classic mp.solutions.face_mesh and the newer MediaPipe
    Tasks FaceLandmarker — whichever the installed version provides.

    Args:
        duration_seconds: How long to record.
        source:           "webcam" | "oak" | "oak-rgb"
        camera_index:     OpenCV camera index for webcam source.
        fps:              Target capture frame rate.
        mirror:           Flip frame horizontally (natural for selfie view).
        draw_overlay:     Draw face oval and key landmarks on the preview frame.
    """
    load_runtime_dependencies()

    frame_source = open_frame_source(source, camera_index=camera_index, fps=fps)
    if not frame_source.is_opened():
        raise RuntimeError(f"Could not open camera source: {source!r}")

    all_frames: list[np.ndarray] = []
    start = time.perf_counter()

    with _create_face_detector() as face_detector:
        try:
            while True:
                elapsed = time.perf_counter() - start
                if elapsed >= duration_seconds:
                    break

                ok, cam_frame = frame_source.read()
                if not ok or cam_frame is None:
                    break

                frame = cam_frame.color
                if mirror:
                    frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                coords = face_detector.landmarks_from_rgb(rgb)

                preview = frame.copy()

                if coords is not None:
                    all_frames.append(coords)

                    if draw_overlay:
                        h, w = frame.shape[:2]
                        pts = [
                            (int(coords[i, 0] * w), int(coords[i, 1] * h))
                            for i in FACE_OVAL_IDXS
                            if i < len(coords)
                        ]
                        for j in range(len(pts)):
                            cv2.line(preview, pts[j], pts[(j + 1) % len(pts)],
                                     (30, 144, 255), 1, cv2.LINE_AA)
                        for idx in FACE_KEY_IDXS:
                            if idx < len(coords):
                                px = int(coords[idx, 0] * w)
                                py = int(coords[idx, 1] * h)
                                cv2.circle(preview, (px, py), 4, (0, 255, 160), -1, cv2.LINE_AA)
                else:
                    cv2.putText(
                        preview, "No face detected — centre your face", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 60, 220), 2, cv2.LINE_AA,
                    )

                yield preview, elapsed, None

        finally:
            if hasattr(frame_source, "release"):
                frame_source.release()

    # Final yield — carries the complete landmark array
    landmarks_array = (
        np.stack(all_frames, axis=0)
        if all_frames
        else np.empty((0, 478, 3), dtype=np.float32)
    )
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    yield blank, duration_seconds, landmarks_array