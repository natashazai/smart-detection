
import depthai as dai
import mediapipe as mp
import numpy as np
import csv
import time
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

base_options = mp_python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
   base_options=base_options,
   num_hands=1,
   min_hand_detection_confidence=0.7
)
hands = vision.HandLandmarker.create_from_options(options)

output_file = "hand_xyz.csv"
record_seconds = 30

pipeline = dai.Pipeline()

cam_rgb = pipeline.create(dai.node.ColorCamera)
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)
cam_rgb.setFps(30)

mono_left = pipeline.create(dai.node.MonoCamera)
mono_right = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)

stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

xout_rgb = pipeline.create(dai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_depth = pipeline.create(dai.node.XLinkOut)
xout_depth.setStreamName("depth")
stereo.depth.link(xout_depth.input)

print("Recording for 30 seconds...")
records = []

with dai.Device(pipeline) as device:
   rgb_queue = device.getOutputQueue("rgb", maxSize=4, blocking=False)
   depth_queue = device.getOutputQueue("depth", maxSize=4, blocking=False)
   start_time = time.time()
   while True:
       elapsed = time.time() - start_time
       if elapsed > record_seconds:
           break
       rgb_frame = rgb_queue.get().getCvFrame()
       depth_frame = depth_queue.get().getFrame()
       mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame[:, :, ::-1])
       result = hands.detect(mp_image)
       if result.hand_landmarks:
           lm = result.hand_landmarks[0][0]
           h, w = rgb_frame.shape[:2]
           px = int(lm.x * w)
           py = int(lm.y * h)
           px = np.clip(px, 0, depth_frame.shape[1] - 1)
           py = np.clip(py, 0, depth_frame.shape[0] - 1)
           depth_mm = depth_frame[py, px]
           records.append({"timestamp": round(elapsed, 4), "x_px": px, "y_px": py, "z_mm": int(depth_mm)})

with open(output_file, "w", newline="") as f:
   writer = csv.DictWriter(f, fieldnames=["timestamp", "x_px", "y_px", "z_mm"])
   writer.writeheader()
   writer.writerows(records)

print(f"Done. {len(records)} frames saved to {output_file}")
EOF