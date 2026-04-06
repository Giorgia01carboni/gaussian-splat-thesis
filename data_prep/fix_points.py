import numpy as np
from read_write_model import read_model, write_model

# Target the reduced directory directly
SPARSE_DIR = "/home/xrdev/Desktop/GPU-tests/gaussian-splatting/GsplatTest/gsplat/examples/datasets/data/360_v2/bicycle_reduced/sparse/0"

def fix_points3d():
    print("Loading reduced COLMAP model...")
    cameras, images, points3D = read_model(path=SPARSE_DIR, ext=".bin")

    valid_image_ids = set(images.keys())
    filtered_points3D = {}

    print("Filtering invalid tracks from points3D.bin...")
    for pt_id, pt in points3D.items():
        # Create a boolean mask of track observations that exist in the valid set
        mask = np.isin(pt.image_ids, list(valid_image_ids))
        
        # If the 3D point is still observed by at least 1 valid image, retain it
        if mask.any():
            # Reconstruct the namedtuple with the filtered arrays
            new_pt = pt._replace(
                image_ids=pt.image_ids[mask], 
                point2D_idxs=pt.point2D_idxs[mask]
            )
            filtered_points3D[pt_id] = new_pt

    print(f"Original 3D points: {len(points3D)}")
    print(f"Filtered 3D points: {len(filtered_points3D)}")
    
    # Overwrite the broken points3D.bin with the patched one
    write_model(cameras, images, filtered_points3D, path=SPARSE_DIR, ext=".bin")
    print("Fix complete. The dataset is now strictly manifold.")

if __name__ == "__main__":
    fix_points3d()