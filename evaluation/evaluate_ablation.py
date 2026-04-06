"""
Script: evaluate_ablation.py
Usage: python evaluate_ablation.py

Objective: Quantitatively evaluates 3D Gaussian Splatting (3DGS) robustness across multiple degradation profiles (Spatial Upsampling, View Decimation, and Low-Fidelity Sensor Noise).
Methodology: Locks the evaluation coordinate space and test set to 5 immutable ground-truth views from the Baseline dataset. Rasterizes each trained model from these exact SE(3) poses and computes PSNR, SSIM, and LPIPS to mathematically isolate the algorithm's response to specific input degradations.
Parameters (Hardcoded within script):
- BASELINE_DATA_DIR: Path to reference geometry and ground-truth images.
- RESULTS_BASE_DIR: Root path to model checkpoints and output directories.
- CHECKPOINTS: Dictionary mapping experiment names to their respective trained `.pt` files.
- GLOBAL_TEST_VIEWS: Array of the 5 static filenames strictly held out for validation.
"""

import os
import json
import torch
import numpy as np
import torchvision
from tqdm import tqdm
from typing import Dict, Any

from datasets.colmap import Parser, Dataset
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from simple_trainer import Runner, Config

BASELINE_DATA_DIR = "/gsplat/examples/datasets/data/360_v2/bicycle"
RESULTS_BASE_DIR = "/gsplat/examples/results"

CHECKPOINTS = {
    "Baseline": os.path.join(RESULTS_BASE_DIR, "bicycle/ckpts/ckpt_29999_rank0.pt"),
    "Spatial_Upsample": os.path.join(RESULTS_BASE_DIR, "bicycle_spatial_upsample/ckpts/ckpt_29999_rank0.pt"),
    "Decimated": os.path.join(RESULTS_BASE_DIR, "bicycle_reduced/ckpts/ckpt_29999_rank0.pt"),
    "Low_Fidelity": os.path.join(RESULTS_BASE_DIR, "bicycle_low_fidelity/ckpts/ckpt_29999_rank0.pt")
}

OUTPUT_JSON = os.path.join(RESULTS_BASE_DIR, "eval_renders", "ablation_metrics.json")

GLOBAL_TEST_VIEWS = [
    "_DSC8679.JPG", 
    "_DSC8687.JPG",
    "_DSC8693.JPG",
    "_DSC8701.JPG",
    "_DSC8715.JPG"
]

@torch.no_grad()
def evaluate_checkpoint(ckpt_path: str, dataset: Dataset, device: str = "cuda:0", save_gt: bool = False) -> Dict[str, Any]:
    if not os.path.exists(ckpt_path):
        print(f"WARNING: Checkpoint not found -> {ckpt_path}")
        return {}

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)
    
    splats = ckpt["splats"] if "splats" in ckpt else ckpt["model"]
    
    cfg = Config(data_dir=BASELINE_DATA_DIR, data_factor=1, eval_steps=[], save_steps=[])
    runner = Runner(local_rank=0, world_rank=0, world_size=1, cfg=cfg)
    
    runner.splats = torch.nn.ParameterDict({
        k: torch.nn.Parameter(v.to(device)) for k, v in splats.items()
    })
    runner.splats.eval()
    
    model_id = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))
    render_out_dir = os.path.join(RESULTS_BASE_DIR, "eval_renders", model_id)
    os.makedirs(render_out_dir, exist_ok=True)
    
    if save_gt:
        gt_out_dir = os.path.join(RESULTS_BASE_DIR, "eval_renders", "ground_truth")
        os.makedirs(gt_out_dir, exist_ok=True)

    image_metrics_log = {}
    psnr_vals, ssim_vals, lpips_vals = [], [], []

    for i, data in enumerate(tqdm(dataset, desc=f"Evaluating {model_id}")):
        camtoworlds = data["camtoworld"].unsqueeze(0).to(device)
        Ks = data["K"].unsqueeze(0).to(device)
        
        gt_image = data["image"].unsqueeze(0).to(device) / 255.0
        width, height = gt_image.shape[2], gt_image.shape[1]
        
        renders, _, _ = runner.rasterize_splats(
            camtoworlds=camtoworlds, Ks=Ks, width=width, height=height
        )
        
        renders = renders.permute(0, 3, 1, 2).clamp(0.0, 1.0)
        gt_image = gt_image.permute(0, 3, 1, 2).clamp(0.0, 1.0)
        
        frame_psnr = psnr_metric(renders, gt_image).item()
        frame_ssim = ssim_metric(renders, gt_image).item()
        frame_lpips = lpips_metric(renders, gt_image).item()
        
        psnr_vals.append(frame_psnr)
        ssim_vals.append(frame_ssim)
        lpips_vals.append(frame_lpips)
        
        actual_img_index = dataset.indices[i] if hasattr(dataset, 'indices') else i
        frame_filename = dataset.parser.image_names[actual_img_index]
        
        torchvision.utils.save_image(renders, os.path.join(render_out_dir, frame_filename))
        if save_gt:
            torchvision.utils.save_image(gt_image, os.path.join(gt_out_dir, frame_filename))
        
        image_metrics_log[frame_filename] = {
            "PSNR": frame_psnr,
            "SSIM": frame_ssim,
            "LPIPS": frame_lpips
        }

    return {
        "mean_metrics": {
            "PSNR": float(np.mean(psnr_vals)),
            "SSIM": float(np.mean(ssim_vals)),
            "LPIPS": float(np.mean(lpips_vals))
        },
        "per_frame_metrics": image_metrics_log
    }

def main():
    # Force parser to ignore interval logic and explicitly map the 5 target test views
    parser = Parser(data_dir=BASELINE_DATA_DIR, factor=1, normalize=True, test_every=1)
    
    test_indices = []
    for i, name in enumerate(parser.image_names):
        if name in GLOBAL_TEST_VIEWS:
            test_indices.append(i)
            
    if len(test_indices) != len(GLOBAL_TEST_VIEWS):
        print(f"CRITICAL WARNING: Found {len(test_indices)} of {len(GLOBAL_TEST_VIEWS)} test images. Verify filenames in baseline directory.")
        
    parser.eval_indices = test_indices
    parser.train_indices = [i for i in range(len(parser.image_names)) if i not in test_indices]
    
    val_dataset = Dataset(parser, split="test", load_depths=False)
    
    results = {}
    first_model = True
    
    for model_name, ckpt_path in CHECKPOINTS.items():
        metrics = evaluate_checkpoint(ckpt_path, val_dataset, save_gt=first_model)
        if metrics:
            results[model_name] = metrics
        first_model = False

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Evaluation complete. Results saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
