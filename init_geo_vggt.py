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

def recover_real_scale(predictions, image_dir):
    # Try to get scene center and scale from metadata
    try:
        # Since return_metadata is not supported, we're going to skip this approach
        # and directly compute scale from 3D points
        raise Exception("Skipping metadata approach, using 3D points directl because return function is not applicable")
    except Exception as e:
        print(f"Could not load metadata from images: {e}")
        print("Computing scene center and scale from 3D points...")
        
        # Compute scene center and scale from 3D points
        all_points = []
        for i in range(len(predictions["extrinsic"])):
            depth = predictions["depth"][i]
            xyz_points = predictions.get("xyz_points", None)
            
            if xyz_points is not None:
                pts = xyz_points[i].reshape(-1, 3)
            else:
                # If xyz_points not available, use depth to compute 3D points
                # Fix: Handle depth with more than 2 dimensions
                if len(depth.shape) > 2:
                    # If depth has shape like (H, W, 1), reshape it to (H, W)
                    depth = depth.reshape(depth.shape[0], depth.shape[1])
                
                h, w = depth.shape
                y, x = np.mgrid[:h, :w]
                x = x.reshape(-1)
                y = y.reshape(-1)
                z = depth.reshape(-1)
                
                # Use estimated intrinsics to get 3D points
                intrinsics = predictions["intrinsic"][i]
                fx = intrinsics[0, 0]
                fy = intrinsics[1, 1]
                cx = intrinsics[0, 2]
                cy = intrinsics[1, 2]
                
                # Project to 3D
                x_world = (x - cx) * z / fx
                y_world = (y - cy) * z / fy
                pts = np.stack([x_world, y_world, z], axis=1)
            
            # Transform to world coordinates
            R = predictions["extrinsic"][i][:3, :3]
            t = predictions["extrinsic"][i][:3, 3]
            pts_world = (R @ pts.T).T + t
            all_points.append(pts_world)
        
        all_points = np.concatenate(all_points, axis=0)
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
    
    # Convert FOV to pixel focal lengths
    intrinsics_metric = []
    
    # Get image dimensions from predictions
    if len(predictions["depth"][0].shape) > 2:
        h, w = predictions["depth"][0].shape[:2]  # Handle depth with shape (H, W, 1)
    else:
        h, w = predictions["depth"][0].shape
    
    for i, intr in enumerate(predictions["intrinsic"]):
        # Extract FOV from intrinsics if available, otherwise use a default
        # Note: This may need adjustment based on how FOV is stored in your VGGT model
        if "fov" in predictions:
            fov_x, fov_y = predictions["fov"][i]
        else:
            # Estimate FOV from intrinsics
            fx, fy = intr[0, 0], intr[1, 1]
            fov_x = 2 * np.arctan(w / (2 * fx))
            fov_y = 2 * np.arctan(h / (2 * fy))
        
        # Convert FOV to pixel focal lengths
        f_px = w / (2 * np.tan(fov_x / 2))
        f_py = h / (2 * np.tan(fov_y / 2))
        c_x, c_y = w/2, h/2
        
        # Create metric intrinsic matrix
        K = np.array([
            [f_px,   0,   c_x],
            [  0,  f_py,  c_y],
            [  0,    0,     1]
        ])
        intrinsics_metric.append(K)
    
    return extrinsics_metric, intrinsics_metric, scene_center, scene_scale

def generate_colmap_and_confidence(image_dir, source_path, n_views, conf_threshold=50.0, 
                                  mask_sky=True, mask_black_bg=True, mask_white_bg=False,
                                  stride=1, prediction_mode="Depthmap and Camera Branch", infer_video=False,
                                  llffhold=8, recover_scale=True):
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
        llffhold: Hold frequency for train/test splitting
        recover_scale: Whether to recover real-world scale
    """
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
        # test_img_files = []
    else:
        train_img_files, test_img_files = split_train_test(image_files, llffhold, n_views, verbose=True)
        print(f"Training images: {len(train_img_files)}, Testing images: {len(test_img_files)}")
    
    # Step 1: Run VGGT to generate COLMAP data
    print("Step 1: Running VGGT to generate COLMAP data...")
    
    # Initialize model
    model, device = load_model()
    
    # Process training images only
    print(f"Processing {len(train_img_files)} training images...")
    predictions, image_names = process_images(train_img_files, model, device)
    
    # Recover real-world scale if requested
    if recover_scale:
        print("Recovering real-world scale...")
        extrinsics_metric, intrinsics_metric, scene_center, scene_scale = recover_real_scale(predictions, image_dir)
        
        # Update predictions with metric data
        original_extrinsics = predictions["extrinsic"]  # Save for reference
        predictions["extrinsic"] = extrinsics_metric
        predictions["intrinsic"] = intrinsics_metric
        
        print(f"Scale recovery complete. Scene scale: {scene_scale}, center: {scene_center}")
    
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
    
    # Write COLMAP files for training images to sparse_0_path
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
        
        # Extract camera extrinsics from predictions
        extrinsics_w2c = np.array([np.linalg.inv(ext) for ext in predictions["extrinsic"]])
        
        # Generate interpolated camera poses for test images
        n_train = len(train_img_files)
        n_test = len(test_img_files)
        
        print(f"Interpolating {n_test} test poses from {n_train} training poses...")
        
        if n_train < n_test:
            # Need to interpolate more poses than we have training views
            n_interp = (n_test // (n_train-1)) + 1
            all_inter_pose = []
            
            for i in range(n_train-1):
                # Extract the rotation and translation components
                pose1 = np.eye(4)
                pose1[:3, :3] = extrinsics_w2c[i][:3, :3]
                pose1[:3, 3] = extrinsics_w2c[i][:3, 3]
                
                pose2 = np.eye(4)
                pose2[:3, :3] = extrinsics_w2c[i+1][:3, :3]
                pose2[:3, 3] = extrinsics_w2c[i+1][:3, 3]
                
                poses = np.stack([pose1[:3, :], pose2[:3, :]], axis=0)
                tmp_inter_pose = generate_interpolated_path(poses=poses, n_interp=n_interp)
                all_inter_pose.append(tmp_inter_pose)
            
            all_inter_pose = np.concatenate(all_inter_pose, axis=0)
            last_pose = np.eye(4)
            last_pose[:3, :3] = extrinsics_w2c[-1][:3, :3]
            last_pose[:3, 3] = extrinsics_w2c[-1][:3, 3]
            all_inter_pose = np.concatenate([all_inter_pose, last_pose[:3, :].reshape(1, 3, 4)], axis=0)
            
            # Sample the poses at regular intervals to match the number of test images
            indices = np.linspace(0, all_inter_pose.shape[0] - 1, n_test, dtype=int)
            sampled_poses = all_inter_pose[indices]
            pose_test_init = []
            
            for p in sampled_poses:
                tmp_view = np.eye(4)
                tmp_view[:3, :3] = p[:3, :3]
                tmp_view[:3, 3] = p[:3, 3]
                pose_test_init.append(tmp_view)
            
            pose_test_init = np.stack(pose_test_init, 0)
        else:
            # We have enough training poses, so just sample from them
            indices = np.linspace(0, extrinsics_w2c.shape[0] - 1, n_test, dtype=int)
            pose_test_init = extrinsics_w2c[indices]
        
        # Generate test image quaternions and translations
        test_quaternions = []
        test_translations = []
        
        for pose in pose_test_init:
            rot = pose[:3, :3]
            trans = pose[:3, 3]
            
            # Convert rotation matrix to quaternion (simplified for this example)
            # In a real implementation, use a proper rotation to quaternion conversion
            from scipy.spatial.transform import Rotation
            quat = Rotation.from_matrix(rot).as_quat()
            quat = np.array([quat[3], quat[0], quat[1], quat[2]])  # wxyz order for COLMAP
            
            test_quaternions.append(quat)
            test_translations.append(trans)
        
        test_quaternions = np.array(test_quaternions)
        test_translations = np.array(test_translations)
        
        # Create test image points2D (empty for test images)
        test_image_points2D = [[] for _ in range(len(test_img_files))]
        test_image_names = [os.path.basename(img_path) for img_path in test_img_files]
        
        # Write COLMAP files for test images to sparse_1_path
        print(f"Writing test views COLMAP files to {sparse_1_path}...")
        os.makedirs(sparse_1_path, exist_ok=True)
        
        write_colmap_cameras_txt(
            os.path.join(sparse_1_path, "cameras.txt"), 
            predictions["intrinsic"], width, height)  # Use same intrinsics
        
        write_colmap_images_txt(
            os.path.join(sparse_1_path, "images.txt"), 
            test_quaternions, test_translations, test_image_points2D, test_image_names)
        
        # Create an empty points3D.txt file or copy from training
        write_colmap_points3D_txt(
            os.path.join(sparse_1_path, "points3D.txt"), 
            points3D)  # Use same points
        
        # Also save confidence to sparse_1_path
        confidence_path_1 = os.path.join(sparse_1_path, "confidence_dsp.npy")
        write_colmap_confidence_npy(confidence_path_1, final_conf_values)
        
        # Also save scene scale info to sparse_1_path
        if recover_scale:
            np.save(os.path.join(sparse_1_path, "scene_scale_info.npy"), scale_info)
        
        print("Test view COLMAP files successfully written to sparse_1_path")
    
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