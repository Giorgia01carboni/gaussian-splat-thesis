import os
import shutil
import numpy as np
from scipy.spatial.transform import Rotation
from read_write_model import read_model, write_model

# --- CONFIGURATION ---
ORIGINAL_DATA_DIR = "/gsplat/examples/datasets/data/360_v2/bicycle"
NEW_DATA_DIR = "/gsplat/examples/datasets/data/360_v2/bicycle_reduced"

NUM_TRAIN_VIEWS = 24 

GLOBAL_TEST_VIEWS = [
    "_DSC8679.JPG", 
    "_DSC8687.JPG",
    "_DSC8693.JPG",
    "_DSC8701.JPG",
    "_DSC8715.JPG"
]

# --- DIRECTORY SETUP ---
ORIG_IMAGES_DIR = os.path.join(ORIGINAL_DATA_DIR, "images")
ORIG_SPARSE_DIR = os.path.join(ORIGINAL_DATA_DIR, "sparse/0")

NEW_IMAGES_DIR = os.path.join(NEW_DATA_DIR, "images")
NEW_SPARSE_DIR = os.path.join(NEW_DATA_DIR, "sparse/0")

os.makedirs(NEW_IMAGES_DIR, exist_ok=True)
os.makedirs(NEW_SPARSE_DIR, exist_ok=True)

# --- FARTHEST POINT SAMPLING LOGIC ---
def compute_camera_centers(images_dict):
    centers = {}
    for image_id, img_data in images_dict.items():
        qvec = img_data.qvec
        scipy_quat = [qvec[1], qvec[2], qvec[3], qvec[0]]
        R = Rotation.from_quat(scipy_quat).as_matrix()
        t = img_data.tvec
        C = -np.dot(R.T, t)
        centers[image_id] = C
    return centers

def compute_fps_on_cameras(images_dict, num_views):
    centers_dict = compute_camera_centers(images_dict)
    image_ids = list(centers_dict.keys())
    centers = np.array([centers_dict[i] for i in image_ids])
    
    if num_views >= len(image_ids):
        return [images_dict[i].name for i in image_ids]

    selected_indices = []
    centroid = np.mean(centers, axis=0)
    first_idx = int(np.argmin(np.linalg.norm(centers - centroid, axis=1)))
    selected_indices.append(first_idx)
    
    distances = np.linalg.norm(centers - centers[first_idx], axis=1)
    
    for _ in range(1, num_views):
        farthest_idx = int(np.argmax(distances))
        selected_indices.append(farthest_idx)
        dist_to_new = np.linalg.norm(centers - centers[farthest_idx], axis=1)
        distances = np.minimum(distances, dist_to_new)

    return [images_dict[image_ids[idx]].name for idx in selected_indices]

def main():
    print("Loading original COLMAP model...")
    cameras, images, points3D = read_model(path=ORIG_SPARSE_DIR, ext=".bin")
    
    train_pool_images = {
        img_id: img_data for img_id, img_data in images.items() 
        if img_data.name not in GLOBAL_TEST_VIEWS
    }
    
    print(f"Running FPS to select {NUM_TRAIN_VIEWS} training views...")
    selected_train_names = compute_fps_on_cameras(train_pool_images, NUM_TRAIN_VIEWS)
    
    allowed_names = set(selected_train_names + GLOBAL_TEST_VIEWS)
    
    filtered_images = {
        img_id: img_data for img_id, img_data in images.items() 
        if img_data.name in allowed_names
    }
    
    print(f"Total images in new dataset: {len(filtered_images)} ({NUM_TRAIN_VIEWS} train + {len(GLOBAL_TEST_VIEWS)} test)")

    print(f"Copying images to {NEW_IMAGES_DIR}...")
    for img_name in allowed_names:
        src_path = os.path.join(ORIG_IMAGES_DIR, img_name)
        dst_path = os.path.join(NEW_IMAGES_DIR, img_name)
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"WARNING: File {src_path} not found.")

    print(f"Writing updated COLMAP model to {NEW_SPARSE_DIR}...")
    write_model(cameras, filtered_images, points3D, path=NEW_SPARSE_DIR, ext=".bin")
    print("Process complete.")

if __name__ == "__main__":
    main()
