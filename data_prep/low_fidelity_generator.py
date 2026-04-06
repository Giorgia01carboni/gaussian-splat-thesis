"""
Experiment: Low-Fidelity Sensor Simulation

Objective: Evaluate 3D Gaussian Splatting (3DGS) densification robustness against photometric degradation.
Rationale: 3DGS utilizes explicit photometric matching via color gradients. Small-sensor consumer 
hardware introduces zero-mean Gaussian noise (high ISO) and block compression artifacts. This script 
isolates these variables by synthesizing standard sensor degradation (\sigma=20, JPEG quality=65) 
on the complete spatial geometry (194 views). This isolates the photometric variable to test if the 
algorithm spawns micro-Gaussians to explain high-frequency noise (overfitting) or successfully 
learns the underlying object geometry.
"""

import os
import cv2
import numpy as np
import shutil

BASE_DIR = "/home/xrdev/Desktop/GPU-tests/gaussian-splatting/GsplatTest/gsplat/examples/datasets/data/360_v2"
IN_DIR = os.path.join(BASE_DIR, "bicycle")
OUT_DIR = os.path.join(BASE_DIR, "bicycle_low_fidelity")

IN_IMAGES = os.path.join(IN_DIR, "images")
IN_SPARSE = os.path.join(IN_DIR, "sparse")

OUT_IMAGES = os.path.join(OUT_DIR, "images")
OUT_SPARSE = os.path.join(OUT_DIR, "sparse")

GAUSSIAN_SIGMA = 20.0
JPEG_QUALITY = 65

def apply_sensor_degradation(image_path: str, output_path: str):
    img = cv2.imread(image_path)
    if img is None:
        return

    img_float = img.astype(np.float32)
    noise = np.random.normal(0, GAUSSIAN_SIGMA, img_float.shape)
    noisy_img = np.clip(img_float + noise, 0, 255).astype(np.uint8)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
    cv2.imwrite(output_path, noisy_img, encode_param)

def main():
    os.makedirs(OUT_IMAGES, exist_ok=True)
    
    if os.path.exists(OUT_SPARSE):
        shutil.rmtree(OUT_SPARSE)
    shutil.copytree(IN_SPARSE, OUT_SPARSE)

    image_files = [f for f in os.listdir(IN_IMAGES) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for filename in image_files:
        in_path = os.path.join(IN_IMAGES, filename)
        out_path = os.path.join(OUT_IMAGES, filename)
        apply_sensor_degradation(in_path, out_path)

if __name__ == "__main__":
    main()