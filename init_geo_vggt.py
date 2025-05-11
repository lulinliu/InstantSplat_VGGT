import numpy as np
import pathlib
import os
import re
import argparse
import sys
import torch

# Add path to ensure vggt is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vggt.vggt_to_colmap import main as vggt_main
from vggt.vggt_to_colmap import load_model, process_images, extrinsic_to_colmap_format
from vggt.vggt_to_colmap import filter_and_prepare_points, write_colmap_cameras_txt
from vggt.vggt_to_colmap import write_colmap_images_txt, write_colmap_points3D_txt
from vggt.vggt_to_colmap import write_colmap_confidence_npy

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
                                  stride=1, prediction_mode="Depthmap and Camera Branch"):
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
    """
    print(f"Processing images from {image_dir}")
    
    # Define output directories based on SOURCE_PATH and N_VIEWS
    colmap_output_dir = source_path
    output_dir_0 = os.path.join(source_path, f"sparse_{n_views}/0")
    output_dir_1 = os.path.join(source_path, f"sparse_{n_views}/1")
    
    print(f"COLMAP output will be saved to {colmap_output_dir}")
    print(f"Confidence will be saved to {output_dir_0}")
    
    # Ensure output directories exist
    os.makedirs(colmap_output_dir, exist_ok=True)
    os.makedirs(output_dir_0, exist_ok=True)
    os.makedirs(output_dir_1, exist_ok=True)
    
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
    print(f"Writing COLMAP files to {output_dir_0}...")
    write_colmap_cameras_txt(
        os.path.join(output_dir_0, "cameras.txt"), 
        predictions["intrinsic"], width, height)
    write_colmap_images_txt(
        os.path.join(output_dir_0, "images.txt"), 
        quaternions, translations, image_points2D, image_names)
    write_colmap_points3D_txt(
        os.path.join(output_dir_0, "points3D.txt"), 
        points3D)
    
    # Save the confidence values directly from VGGT
    confidence_path = os.path.join(output_dir_0, "confidence_dsp.npy")
    write_colmap_confidence_npy(confidence_path, final_conf_values)
    
    print("COLMAP files successfully written")
    
    # Also create empty file in output_dir_1 to ensure the directory is populated
    with open(os.path.join(output_dir_1, ".placeholder"), 'w') as f:
        f.write("Placeholder file to ensure directory exists")
    
    print(f"Created directory: {output_dir_1}")
    
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
            args.prediction_mode
        )
    else:
        # Legacy functionality - just generate confidence from existing points3D.txt
        sparse = pathlib.Path(args.scene)
        xyz, rgb, reproj = load_points3D(sparse/"points3D.txt")

        # set confidence to 0.3
        conf = np.full(reproj.shape, 0.3, dtype=np.float32)
        
        # Define directories based on n_views
        output_dir_0 = sparse/f"sparse_{args.n_views}/0"
        output_dir_0.mkdir(parents=True, exist_ok=True)
        
        output_dir_1 = sparse/f"sparse_{args.n_views}/1"
        output_dir_1.mkdir(parents=True, exist_ok=True)
        
        # Save confidence to output_dir_0
        np.save(output_dir_0/"confidence_dsp.npy", conf)
        
        # Create placeholder in output_dir_1
        with open(output_dir_1/".placeholder", 'w') as f:
            f.write("Placeholder file to ensure directory exists")
        
        print("Generated confidence_dsp.npy")