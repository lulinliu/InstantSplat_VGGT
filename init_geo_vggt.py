import numpy as np
import pathlib
import os
import re
import argparse
import sys
import torch
from pathlib import Path


# Add path to ensure vggt is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vggt.vggt_to_colmap import main as vggt_main
from vggt.vggt_to_colmap import load_model, process_images, extrinsic_to_colmap_format
from vggt.vggt_to_colmap import filter_and_prepare_points, write_colmap_cameras_txt
from vggt.vggt_to_colmap import write_colmap_images_txt, write_colmap_points3D_txt
from vggt.vggt_to_colmap import write_colmap_confidence_npy

from utils.sfm_utils import (save_intrinsics, save_extrinsic, save_points3D, save_time, save_images_and_masks,
                             init_filestructure, get_sorted_image_files, split_train_test, load_images, compute_co_vis_masks)
from utils.camera_utils import generate_interpolated_path

def load_points3D(txt):
    xyz, rgb, err = [], [], []
    with open(txt, 'r') as f:
        for ln in f:
            if ln.startswith('#'): continue
            v = re.split(r'\s+', ln.strip())
            xyz.append([float(v[1]), float(v[2]), float(v[3])])
            rgb.append([int(v[4])/255, int(v[5])/255, int(v[6])/255])
            err.append(float(v[7]))
    return np.asarray(xyz, np.float32), np.asarray(rgb, np.float32), np.asarray(err, np.float32)

def generate_colmap_and_confidence(image_dir, source_path, n_views, conf_threshold=50.0, 
                                  mask_sky=True, mask_black_bg=True, mask_white_bg=False,
                                  stride=1, prediction_mode="Depthmap and Camera Branch", infer_video=False):
    """
    Generate COLMAP data using VGGT and compute confidence
    
    Args:
        image_dir: Directory containing input images
        source_path: Source path from run_infer.sh
        n_views: Number of views being processed
        conf_threshold: Confidence threshold for VGGT point filtering
        mask_sky: Whether to filter out sky points
        mask_black_bg: Whether to filter out black background
        mask_white_bg: Whether to filter out white background
        stride: Stride for point sampling (higher = fewer points)
        prediction_mode: Which prediction branch to use ("Depthmap and Camera Branch" or "Pointmap Branch")
        infer_video: If True, only sparse/0 is populated. If False, both sparse/0 and sparse/1 are populated.
    """
    print(f"Processing images from {image_dir}")
    
    # Define output directories based on SOURCE_PATH and N_VIEWS
    colmap_output_dir = source_path
    # output_dir_0 = os.path.join(source_path, f"sparse_{n_views}/0")
    save_path, sparse_0_path, sparse_1_path = init_filestructure(Path(source_path), n_views)
    
    print(f"COLMAP output will be saved to {colmap_output_dir}")
    print(f"Confidence will be saved to {sparse_0_path}")
    
    # Ensure output directories exist
    os.makedirs(colmap_output_dir, exist_ok=True)
    os.makedirs(sparse_0_path, exist_ok=True)
    
    # Only create sparse/1 directory if infer_video is False
    if not infer_video:
        sparse_1_path = os.path.join(source_path, f"sparse_{n_views}/1")
        os.makedirs(sparse_1_path, exist_ok=True)
        print(f"Created directory: {sparse_1_path}")
    
    # Run VGGT to generate COLMAP data directly
    print("Step 1: Running VGGT to generate COLMAP data...")
    
    # Initialize model
    model, device = load_model()
    
    # Process images
    predictions, image_names = process_images(image_dir, model, device)
    
    # Convert to COLMAP format
    print("Converting camera parameters to COLMAP format...")
    quaternions, translations = extrinsic_to_colmap_format(predictions["extrinsic"])
    
    # Filter and prepare points
    print(f"Filtering points with confidence threshold {conf_threshold}%...")
    points3D, image_points2D, final_conf_values = filter_and_prepare_points(
        predictions, 
        conf_threshold, 
        mask_sky=mask_sky, 
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        stride=stride,
        prediction_mode=prediction_mode
    )
    
    # Get image dimensions
    height, width = predictions["depth"].shape[1:3]
    
    # Write COLMAP files
    print(f"Writing COLMAP files to {sparse_0_path}...")
    write_colmap_cameras_txt(
        os.path.join(sparse_0_path, "cameras.txt"), 
        predictions["intrinsic"], width, height)
    write_colmap_images_txt(
        os.path.join(sparse_0_path, "images.txt"), 
        quaternions, translations, image_points2D, image_names)
    write_colmap_points3D_txt(
        os.path.join(sparse_0_path, "points3D.txt"), 
        points3D)
    
    # Save the confidence values directly from VGGT
    confidence_path = os.path.join(sparse_0_path, "confidence_dsp.npy")
    write_colmap_confidence_npy(confidence_path, final_conf_values)
    
    print("COLMAP files successfully written to sparse/0")
    
    # Only create placeholder in sparse/1 if infer_video is False
    if not infer_video:
        sparse_1_path = os.path.join(source_path, f"sparse_{n_views}/1")
        os.makedirs(sparse_1_path, exist_ok=True)
        
        # Also write COLMAP files to sparse/1 when infer_video is False
        print(f"Writing COLMAP files to {sparse_1_path}...")
        write_colmap_cameras_txt(
            os.path.join(sparse_1_path, "cameras.txt"), 
            predictions["intrinsic"], width, height)
        write_colmap_images_txt(
            os.path.join(sparse_1_path, "images.txt"), 
            quaternions, translations, image_points2D, image_names)
        write_colmap_points3D_txt(
            os.path.join(sparse_1_path, "points3D.txt"), 
            points3D)
        
        # Also save confidence to sparse/1
        confidence_path_1 = os.path.join(sparse_1_path, "confidence_dsp.npy")
        write_colmap_confidence_npy(confidence_path_1, final_conf_values)
        
        print("COLMAP files successfully written to sparse/1")
    
    return confidence_path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", help="Scene directory (MODEL_PATH from run_infer.sh)")
    ap.add_argument("--run_vggt", action="store_true", help="Run VGGT to COLMAP conversion")
    ap.add_argument("--image_dir", type=str, help="Directory containing input images")
    ap.add_argument("--n_views", type=int, default=3, help="Number of views being processed")
    ap.add_argument("--conf_threshold", type=float, default=50.0, help="Confidence threshold (0-100%)")
    ap.add_argument("--mask_sky", action="store_true", help="Filter sky points")
    ap.add_argument("--mask_black_bg", action="store_true", help="Filter black background points")
    ap.add_argument("--mask_white_bg", action="store_true", help="Filter white background points")
    ap.add_argument("--stride", type=int, default=1, help="Stride for point sampling (higher = fewer points)")
    ap.add_argument("--prediction_mode", type=str, default="Depthmap and Camera Branch",
                    choices=["Depthmap and Camera Branch", "Pointmap Branch"],
                    help="Which prediction branch to use")
    ap.add_argument("--infer_video", action="store_true", help="If set, only sparse/0 is populated")
    args = ap.parse_args()
    
    if args.run_vggt:
        if not args.image_dir:
            print("Error: --image_dir must be specified when using --run_vggt")
            sys.exit(1)
        
        source_path = args.scene
        
        generate_colmap_and_confidence(
            args.image_dir,
            source_path,
            args.n_views,
            args.conf_threshold,
            args.mask_sky,
            args.mask_black_bg,
            args.mask_white_bg,
            args.stride,
            args.prediction_mode,
            args.infer_video
        )
    else:
        # Legacy functionality - just generate confidence from existing points3D.txt
        sparse = pathlib.Path(args.scene)
        xyz, rgb, reproj = load_points3D(sparse/"points3D.txt")

        # set confidence to 0.3
        conf = np.full(reproj.shape, 0.3, dtype=np.float32)
        
        # Define directories based on n_views
        sparse_0_path = sparse/f"sparse_{args.n_views}/0"
        sparse_0_path.mkdir(parents=True, exist_ok=True)
        
        # Save confidence to sparse_0_path
        np.save(sparse_0_path/"confidence_dsp.npy", conf)
        
        # Only create placeholder in sparse/1 if infer_video is False
        if not args.infer_video:
            sparse_1_path = sparse/f"sparse_{args.n_views}/1"
            sparse_1_path.mkdir(parents=True, exist_ok=True)
            with open(sparse_1_path/".placeholder", 'w') as f:
                f.write("Placeholder file to ensure directory exists")
        
        print("Generated confidence_dsp.npy")