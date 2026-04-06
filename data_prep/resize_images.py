import os
import cv2
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
DATA_DIR = Path("") 
FACTOR = 2  
# ---------------------

def resize_images(data_dir, factor):
    src_dir = data_dir / "images"
    
    dst_dir = data_dir / f"images_{factor}" 

    if not src_dir.exists():
        print(f"Error: Source directory {src_dir} does not exist.")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Resizing images from {src_dir} to {dst_dir} by factor {factor}...")
    
    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    files = [f for f in os.listdir(src_dir) if Path(f).suffix.lower() in image_extensions]
    
    for fname in tqdm(files):
        src_path = src_dir / fname
        
        # GSPLAT SPECIFIC CHANGE:
        dst_fname = Path(fname).with_suffix('.png') 
        dst_path = dst_dir / dst_fname
        
        img = cv2.imread(str(src_path))
        if img is None:
            continue
            
        h, w = img.shape[:2]
        new_size = (w // factor, h // factor)
        
        # INTER_AREA is correct for downscaling
        resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        
        cv2.imwrite(str(dst_path), resized_img)
    
    print(f"Done. Created {dst_dir.name} at {dst_dir}")

if __name__ == "__main__":
    resize_images(DATA_DIR, FACTOR)
