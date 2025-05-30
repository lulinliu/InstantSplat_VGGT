import numpy as np
import pathlib
import os
import re
import argparse
import sys
import torch
from pathlib import Path
from PIL import Image
import glob


# Add path to ensure vggt is importable
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vggt.vggt_to_colmap import main as vggt_main
from vggt.vggt_to_colmap import load_model, process_images, extrinsic_to_colmap_format
from vggt.vggt_to_colmap import filter_and_prepare_points, write_colmap_cameras_txt
from vggt.vggt_to_colmap import write_colmap_images_txt, write_colmap_points3D_txt
from vggt.vggt_to_colmap import write_colmap_confidence_npy

# Import metadata loading function - may need to adjust import path based on actual code structure
from vggt.vggt.utils.load_fn import load_and_preprocess_images

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

def get_image_dimensions(all_predictions, image_dir):
    # Get VGGT processed image dimensions
    if len(all_predictions["depth"][0].shape) > 2:
        vggt_h, vggt_w = all_predictions["depth"][0].shape[:2]  
    else:
        vggt_h, vggt_w = all_predictions["depth"][0].shape
    
    # Get original image dimensions
    image_files = glob.glob(os.path.join(image_dir, "*"))
    image_files = sorted([f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if image_files:
        sample_img = Image.open(image_files[0])
        orig_w, orig_h = sample_img.size
        print(f"Original image size: {orig_w}x{orig_h}, VGGT size: {vggt_w}x{vggt_h}")
    else:
        print("Warning: Could not determine original image size, using VGGT size")
        orig_w, orig_h = vggt_w, vggt_h
    
    return vggt_w, vggt_h, orig_w, orig_h

def recover_real_scale(predictions, vggt_w, vggt_h, orig_w, orig_h):
    # Compute scene center and scale from 3D points
    all_points = []
    for i in range(len(predictions["extrinsic"])):
        depth = predictions["depth"][i]
        xyz_points = predictions.get("xyz_points", None)
        
        if xyz_points is not None:
            pts = xyz_points[i].reshape(-1, 3)
        else:
            if len(depth.shape) > 2:
                depth = depth.reshape(depth.shape[0], depth.shape[1])
            
            h, w = depth.shape
            y, x = np.mgrid[:h, :w]
            x = x.reshape(-1)
            y = y.reshape(-1)
            z = depth.reshape(-1)
            
            intrinsics = predictions["intrinsic"][i]
            fx = intrinsics[0, 0]
            fy = intrinsics[1, 1]
            cx = intrinsics[0, 2]
            cy = intrinsics[1, 2]
            
            x_world = (x - cx) * z / fx
            y_world = (y - cy) * z / fy
            pts = np.stack([x_world, y_world, z], axis=1)
        
        # Transform to world coordinates
        R = predictions["extrinsic"][i][:3, :3]
        t = predictions["extrinsic"][i][:3, 3]
        pts_world = (R @ pts.T).T + t
        all_points.append(pts_world)
    
    all_points = np.concatenate(all_points, axis=0)
    mins = np.min(all_points, axis=0)
    maxs = np.max(all_points, axis=0)
    print(f"[VGGT] point cloud coord ranges:\n"
          f"  x: {mins[0]:.4f} ~ {maxs[0]:.4f}\n"
          f"  y: {mins[1]:.4f} ~ {maxs[1]:.4f}\n"
          f"  z: {mins[2]:.4f} ~ {maxs[2]:.4f}")

    scene_center = np.mean(all_points, axis=0)
    dists = np.linalg.norm(all_points - scene_center, axis=1)
    scene_scale = np.max(dists)
    
    print(f"Computed scene scale: {scene_scale}, center: {scene_center}")
    
    # Recover real-world scale of camera translations
    extrinsics_metric = []
    for ext in predictions["extrinsic"]:
        R = ext[:3, :3]
        t_norm = ext[:3, 3]
        
        # Convert normalized translation to metric scale
        t_metric = t_norm * scene_scale + scene_center
        
        # Create metric extrinsic matrix
        ext_metric = np.eye(4)
        ext_metric[:3, :3] = R
        ext_metric[:3, 3] = t_metric
        extrinsics_metric.append(ext_metric)
    
    # Convert FOV to pixel focal lengths WITH scaling
    intrinsics_metric = []
    for i, intr in enumerate(predictions["intrinsic"]):
        # Extract FOV from intrinsics (using VGGT dimensions)
        fx, fy = intr[0, 0], intr[1, 1]
        fov_x = 2 * np.arctan(vggt_w / (2 * fx))
        fov_y = 2 * np.arctan(vggt_h / (2 * fy))
        
        # Convert FOV to pixel focal lengths for original image size
        f_px = orig_w / (2 * np.tan(fov_x / 2))
        f_py = orig_h / (2 * np.tan(fov_y / 2))
        c_x, c_y = orig_w/2, orig_h/2
        
        # Create metric intrinsic matrix for original image size
        K = np.array([
            [f_px,   0,   c_x],
            [  0,  f_py,  c_y],
            [  0,    0,     1]
        ])
        intrinsics_metric.append(K)
        
        print(f"Camera {i+1}: VGGT f=({fx:.1f},{fy:.1f}) -> Original f=({f_px:.1f},{f_py:.1f})")
    
    return extrinsics_metric, intrinsics_metric, scene_center, scene_scale

def scale_test_intrinsics(test_predictions, vggt_w, vggt_h, orig_w, orig_h):
    test_intrinsics_metric = []
    for i, intr in enumerate(test_predictions["intrinsic"]):
        fx, fy = intr[0, 0], intr[1, 1]
        fov_x = 2 * np.arctan(vggt_w / (2 * fx))
        fov_y = 2 * np.arctan(vggt_h / (2 * fy))
        
        # Using original image size
        f_px = orig_w / (2 * np.tan(fov_x / 2))
        f_py = orig_h / (2 * np.tan(fov_y / 2))
        c_x, c_y = orig_w/2, orig_h/2
        
        # Create metric intrinsic matrix
        K = np.array([
            [f_px,   0,   c_x],
            [  0,  f_py,  c_y],
            [  0,    0,     1]
        ])
        test_intrinsics_metric.append(K)
    
    return test_intrinsics_metric

def generate_colmap_and_confidence(image_dir, source_path, n_views, conf_threshold, 
                                  mask_sky=True, mask_black_bg=True, mask_white_bg=False,
                                  stride=1, prediction_mode="Depthmap and Camera Branch", infer_video=False,
                                  llffhold=8, recover_scale=True):

    print(f"Processing images from {image_dir}")
    
    # Define output directories based on SOURCE_PATH and N_VIEWS
    save_path, sparse_0_path, sparse_1_path = init_filestructure(Path(source_path), n_views)
    
    print(f"COLMAP output will be saved to {source_path}")
    print(f"Primary reconstruction will be saved to {sparse_0_path}")
    
    # Get sorted image files
    image_files, image_suffix = get_sorted_image_files(Path(image_dir))
    
    # Split images into train and test sets if not infer_video
    if infer_video:
        train_img_files = image_files
        test_img_files = []
    else:
        train_img_files, test_img_files = split_train_test(image_files, llffhold, n_views, verbose=True)
        print(f"Training images: {len(train_img_files)}, Testing images: {len(test_img_files)}")
    
    # Step 1: Run VGGT to generate COLMAP data
    print("Step 1: Running VGGT to generate COLMAP data...")
    
    # Initialize model
    model, device = load_model()
    
    all_img_files = train_img_files + test_img_files
    print(f"Processing {len(all_img_files)} images (train + test)...")
    all_predictions, all_image_names = process_images(all_img_files, model, device)
    
    vggt_w, vggt_h, orig_w, orig_h = get_image_dimensions(all_predictions, image_dir)
    
    n_train = len(train_img_files)
    n_test = len(test_img_files)
    
    print(f"Separating predictions: {n_train} training, {n_test} testing")
    
    train_image_names = all_image_names[:n_train]
    test_image_names = all_image_names[n_train:] if n_test > 0 else []
    
    # Separate predictions
    train_predictions = {key: all_predictions[key][:n_train] for key in all_predictions.keys()}
    
    if n_test > 0:
        test_predictions = {key: all_predictions[key][n_train:] for key in all_predictions.keys()}
    
    # Recover real-world scale if requested
    if recover_scale:
        print("Recovering real-world scale...")
        extrinsics_metric, intrinsics_metric, scene_center, scene_scale = recover_real_scale(
            train_predictions, vggt_w, vggt_h, orig_w, orig_h)
        
        # Update train predictions with metric data
        train_predictions["extrinsic"] = extrinsics_metric
        train_predictions["intrinsic"] = intrinsics_metric
        
        # Update test predictions with metric data
        if n_test > 0:
            # Scale test extrinsics
            test_extrinsics_metric = []
            for ext in test_predictions["extrinsic"]:
                R = ext[:3, :3]
                t_norm = ext[:3, 3]
                t_metric = t_norm * scene_scale + scene_center
                ext_metric = np.eye(4)
                ext_metric[:3, :3] = R
                ext_metric[:3, 3] = t_metric
                test_extrinsics_metric.append(ext_metric)
            
            # Scale test intrinsics
            test_intrinsics_metric = scale_test_intrinsics(
                test_predictions, vggt_w, vggt_h, orig_w, orig_h)
            
            test_predictions["extrinsic"] = test_extrinsics_metric
            test_predictions["intrinsic"] = test_intrinsics_metric
        
        print(f"Scale recovery complete. Scene scale: {scene_scale}, center: {scene_center}")
    
    # Convert to COLMAP format
    print("Converting camera parameters to COLMAP format...")
    quaternions, translations = extrinsic_to_colmap_format(train_predictions["extrinsic"])
    
    # Filter and prepare points
    print(f"Filtering points with confidence threshold {conf_threshold}%...")
    points3D, image_points2D, final_conf_values = filter_and_prepare_points(
        train_predictions, 
        conf_threshold, 
        mask_sky=mask_sky, 
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        stride=stride,
        prediction_mode=prediction_mode
    )
    
    # Write COLMAP files for training images to sparse_0_path
    print(f"Writing COLMAP files to {sparse_0_path}...")
    write_colmap_cameras_txt(
        os.path.join(sparse_0_path, "cameras.txt"), 
        train_predictions["intrinsic"], orig_w, orig_h)
    write_colmap_images_txt(
        os.path.join(sparse_0_path, "images.txt"), 
        quaternions, translations, image_points2D, train_image_names)
    write_colmap_points3D_txt(
        os.path.join(sparse_0_path, "points3D.txt"), 
        points3D)
    
    # Save the confidence values
    confidence_path = os.path.join(sparse_0_path, "confidence_dsp.npy")
    write_colmap_confidence_npy(confidence_path, final_conf_values)
    
    # Save scene scale and center for reference
    if recover_scale:
        scale_info = {
            "scale": scene_scale,
            "center": scene_center
        }
        np.save(os.path.join(sparse_0_path, "scene_scale_info.npy"), scale_info)
    
    print("COLMAP files successfully written to sparse_0_path")
    
    # Handle test images if not infer_video
    if not infer_video and len(test_img_files) > 0:
        print(f"Processing test views for sparse_1_path...")
        
        test_quaternions, test_translations = extrinsic_to_colmap_format(test_predictions["extrinsic"])
        
        # Create test image points2D (empty for test images)
        test_image_points2D = [[] for _ in range(len(test_img_files))]
        
        # Write COLMAP files for test images to sparse_1_path
        print(f"Writing test views COLMAP files to {sparse_1_path}...")
        os.makedirs(sparse_1_path, exist_ok=True)
        
        print(f"Using {n_test} real camera intrinsics for test images...")
        write_colmap_cameras_txt(
            os.path.join(sparse_1_path, "cameras.txt"), 
            test_predictions["intrinsic"], orig_w, orig_h)
        
        write_colmap_images_txt(
            os.path.join(sparse_1_path, "images.txt"), 
            test_quaternions, test_translations, test_image_points2D, test_image_names)
        
        # Copy points3D from training
        write_colmap_points3D_txt(
            os.path.join(sparse_1_path, "points3D.txt"), 
            points3D)
        
        # Also save confidence to sparse_1_path
        confidence_path_1 = os.path.join(sparse_1_path, "confidence_dsp.npy")
        write_colmap_confidence_npy(confidence_path_1, final_conf_values)
        
        # Also save scene scale info to sparse_1_path
        if recover_scale:
            np.save(os.path.join(sparse_1_path, "scene_scale_info.npy"), scale_info)
        
        print("Test view COLMAP files successfully written to sparse_1_path")
        print(f"Used real intrinsics for {n_test} test cameras (computed by VGGT)")
    
    return confidence_path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", help="Scene directory (MODEL_PATH from run_infer.sh)")
    ap.add_argument("--run_vggt", action="store_true", help="Run VGGT to COLMAP conversion")
    ap.add_argument("--image_dir", type=str, help="Directory containing input images")
    ap.add_argument("--n_views", type=int, default=3, help="Number of views being processed")
    ap.add_argument("--conf_threshold", type=float, default=30.0, help="Confidence threshold (0-100%)")
    ap.add_argument("--mask_sky", action="store_true", help="Filter sky points")
    ap.add_argument("--mask_black_bg", action="store_true", help="Filter black background points")
    ap.add_argument("--mask_white_bg", action="store_true", help="Filter white background points")
    ap.add_argument("--stride", type=int, default=1, help="Stride for point sampling (higher = fewer points)")
    ap.add_argument("--prediction_mode", type=str, default="Depthmap and Camera Branch",
                    choices=["Depthmap and Camera Branch", "Pointmap Branch"],
                    help="Which prediction branch to use")
    ap.add_argument("--infer_video", action="store_true", help="If set, only sparse/0 is populated")
    ap.add_argument("--llffhold", type=int, default=8, help="Hold frequency for train/test splitting")
    ap.add_argument("--recover_scale", action="store_true", help="Recover real-world scale from normalized outputs")
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
            args.infer_video,
            args.llffhold,
            args.recover_scale
        )
    else:
        # Legacy functionality - just generate confidence from existing points3D.txt
        sparse = pathlib.Path(args.scene)
        xyz, rgb, reproj = load_points3D(sparse/"points3D.txt")

        conf = np.full(reproj.shape, 0.1, dtype=np.float32)
        
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