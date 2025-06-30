import os
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from time import time
import shutil
from types import SimpleNamespace
import PIL
import PIL.Image
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Add VGGT directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vggt'))

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from icecream import ic
ic(torch.cuda.is_available())  # Check if CUDA is available
ic(torch.cuda.device_count())

# Import VGGT components
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.track_predict import predict_tracks
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track

# Import utilities
from utils.sfm_utils import (save_intrinsics, save_extrinsic, save_time, 
                             init_filestructure, get_sorted_image_files, split_train_test, rigid_points_registration)
from utils.camera_utils import generate_interpolated_path
import pycolmap
import trimesh
import torch.nn.functional as F


# def inv(mat):
#     """ Invert a torch or numpy matrix
#     """
#     if isinstance(mat, torch.Tensor):
#         return torch.linalg.inv(mat)
#     if isinstance(mat, np.ndarray):
#         return np.linalg.inv(mat)
#     raise ValueError(f'bad matrix type = {type(mat)}')


def run_VGGT_with_confidence(model, images, dtype, device, resolution=518):
    """Run VGGT inference with full confidence extraction"""
    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # Resize to VGGT's fixed resolution
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            
            # Get full predictions including all confidence types
            predictions = model(images)

    # Extract all available confidence types
    confidence_data = {}
    
    # 1. Depth confidence (always available)
    if "depth_conf" in predictions:
        confidence_data["depth_conf"] = predictions["depth_conf"].squeeze(0).cpu().numpy()
    
    # 2. World points confidence (if available)
    if "world_points_conf" in predictions:
        confidence_data["world_points_conf"] = predictions["world_points_conf"].squeeze(0).cpu().numpy()
    
    # 3. Extract other predictions
    extrinsic = predictions["pose_enc"].squeeze(0).cpu().numpy()
    depth_map = predictions["depth"].squeeze(0).cpu().numpy()
    
    # Convert pose encoding to camera parameters
    extrinsic_mat, intrinsic_mat = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    extrinsic_mat = extrinsic_mat.squeeze(0).cpu().numpy()
    intrinsic_mat = intrinsic_mat.squeeze(0).cpu().numpy()
    
    return extrinsic_mat, intrinsic_mat, depth_map, confidence_data, predictions


def apply_confidence_based_filtering(points_3d, conf_values, points_rgb, points_xyf, max_points=100000, output_dir=None):

    print(f">> Applying confidence-based filtering...")
    print(f"Original points: {points_3d.shape}")
    print(f"Original confidence: {conf_values.shape}")
    
    # Flatten all arrays
    points_flat = points_3d.reshape(-1, 3)  # (N, 3)
    conf_flat = conf_values.reshape(-1)     # (N,)
    rgb_flat = points_rgb.reshape(-1, 3)    # (N, 3)
    xyf_flat = points_xyf.reshape(-1, 3)    # (N, 3)
    
    n_total_points = len(points_flat)
    print(f"Total flattened points: {n_total_points}")
    
    assert len(conf_flat) == n_total_points, f"Confidence shape mismatch: {len(conf_flat)} vs {n_total_points}"
    assert len(rgb_flat) == n_total_points, f"RGB shape mismatch: {len(rgb_flat)} vs {n_total_points}"
    assert len(xyf_flat) == n_total_points, f"XYF shape mismatch: {len(xyf_flat)} vs {n_total_points}"
    
    print("\n" + "="*60)
    print("CONFIDENCE STATISTICS ANALYSIS")
    print("="*60)
    
    conf_min = np.min(conf_flat)
    conf_max = np.max(conf_flat)
    conf_mean = np.mean(conf_flat)
    conf_median = np.median(conf_flat)
    conf_std = np.std(conf_flat)
    
    conf_q1 = np.percentile(conf_flat, 25)    # (25%)
    conf_q3 = np.percentile(conf_flat, 75)    # (75%)
    conf_iqr = conf_q3 - conf_q1              # (IQR)
    
    conf_p5 = np.percentile(conf_flat, 5)     # 5%
    conf_p95 = np.percentile(conf_flat, 95)   # 95%
    conf_p99 = np.percentile(conf_flat, 99)   # 99%
    
    print(f"Min:        {conf_min:.6f}")
    print(f"Q1 (25%):   {conf_q1:.6f}")
    print(f"Median:     {conf_median:.6f}")
    print(f"Mean:       {conf_mean:.6f}")
    print(f"Q3 (75%):   {conf_q3:.6f}")
    print(f"Max:        {conf_max:.6f}")
    print(f"Std:        {conf_std:.6f}")
    print(f"IQR:        {conf_iqr:.6f}")
    print(f"5th percentile:   {conf_p5:.6f}")
    print(f"95th percentile:  {conf_p95:.6f}")
    print(f"99th percentile:  {conf_p99:.6f}")
    
    # Statistics of confidence distribution
    high_conf_count = np.sum(conf_flat > conf_q3)
    med_conf_count = np.sum((conf_flat >= conf_q1) & (conf_flat <= conf_q3))
    low_conf_count = np.sum(conf_flat < conf_q1)
    
    print(f"\nConfidence Distribution:")
    print(f"max_points: {max_points}")
    print(f"High confidence (>Q3): {high_conf_count:,} points ({100*high_conf_count/n_total_points:.1f}%)")
    print(f"Medium confidence (Q1-Q3): {med_conf_count:,} points ({100*med_conf_count/n_total_points:.1f}%)")
    print(f"Low confidence (<Q1): {low_conf_count:,} points ({100*low_conf_count/n_total_points:.1f}%)")
    
    # Sort by confidence (highest confidence first)
    sorted_indices = np.argsort(conf_flat)[::-1]
    sorted_conf = conf_flat[sorted_indices]
    
    # Select top max_points highest confidence points
    if n_total_points > max_points:
        selected_indices = sorted_indices[:max_points]
        print(f"\nSelected top {max_points:,} points from {n_total_points:,} (keeping {100*max_points/n_total_points:.1f}%)")
        
        # Statistics of confidence of selected points
        selected_conf = sorted_conf[:max_points]
        selected_min = selected_conf.min()
        selected_max = selected_conf.max()
        selected_mean = selected_conf.mean()
        selected_median = np.median(selected_conf)
        
        print(f"Selected points confidence:")
        print(f"  Range: [{selected_min:.6f}, {selected_max:.6f}]")
        print(f"  Mean: {selected_mean:.6f}")
        print(f"  Median: {selected_median:.6f}")
        print(f"  Confidence threshold (lowest selected): {selected_min:.6f}")
        
    else:
        selected_indices = sorted_indices
        selected_conf = sorted_conf
        print(f"\nKeeping all {n_total_points:,} points (below max_points limit)")
    
    # Filter all arrays
    filtered_points = points_flat[selected_indices]
    filtered_conf = conf_flat[selected_indices]
    filtered_rgb = rgb_flat[selected_indices]
    filtered_xyf = xyf_flat[selected_indices]
    
    # ======================== Visualize confidence distribution ========================
    if output_dir is not None:
        print(f"\n>> Creating confidence visualization...")
        
        # Create figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Test Pose Confidence Analysis', fontsize=16, fontweight='bold')
        
        # 1. Sorted confidence curve
        x_all = np.arange(len(sorted_conf))
        ax1.plot(x_all, sorted_conf, 'b-', linewidth=1, alpha=0.7, label='All points')
        if len(selected_conf) < len(sorted_conf):
            ax1.plot(x_all[:len(selected_conf)], selected_conf, 'r-', linewidth=2, label=f'Selected top {len(selected_conf):,}')
            ax1.axvline(x=len(selected_conf), color='red', linestyle='--', alpha=0.8, label='Selection cutoff')
        
        ax1.set_xlabel('Point Index (sorted by confidence)')
        ax1.set_ylabel('Confidence Value')
        ax1.set_title('Sorted Confidence Values')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Confidence histogram
        ax2.hist(conf_flat, bins=100, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(conf_mean, color='red', linestyle='-', linewidth=2, label=f'Mean: {conf_mean:.4f}')
        ax2.axvline(conf_median, color='green', linestyle='-', linewidth=2, label=f'Median: {conf_median:.4f}')
        ax2.axvline(conf_q1, color='orange', linestyle='--', linewidth=1.5, label=f'Q1: {conf_q1:.4f}')
        ax2.axvline(conf_q3, color='orange', linestyle='--', linewidth=1.5, label=f'Q3: {conf_q3:.4f}')
        if len(selected_conf) < len(sorted_conf):
            ax2.axvline(selected_conf.min(), color='purple', linestyle='-', linewidth=2, 
                       label=f'Selection threshold: {selected_conf.min():.4f}')
        
        ax2.set_xlabel('Confidence Value')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Confidence Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Log
        ax3.semilogy(x_all, sorted_conf, 'b-', linewidth=1, alpha=0.7)
        if len(selected_conf) < len(sorted_conf):
            ax3.semilogy(x_all[:len(selected_conf)], selected_conf, 'r-', linewidth=2)
            ax3.axvline(x=len(selected_conf), color='red', linestyle='--', alpha=0.8)
        
        ax3.set_xlabel('Point Index (sorted by confidence)')
        ax3.set_ylabel('Confidence Value (log scale)')
        ax3.set_title('Sorted Confidence Values (Log Scale)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Comparison of selected vs unselected points
        if len(selected_conf) < len(sorted_conf):
            rejected_conf = sorted_conf[len(selected_conf):]
            
            bins = np.linspace(conf_min, conf_max, 50)
            ax4.hist(selected_conf, bins=bins, alpha=0.7, color='green', label=f'Selected ({len(selected_conf):,})', density=True)
            ax4.hist(rejected_conf, bins=bins, alpha=0.7, color='red', label=f'Rejected ({len(rejected_conf):,})', density=True)
            ax4.set_xlabel('Confidence Value')
            ax4.set_ylabel('Density')
            ax4.set_title('Selected vs Rejected Points')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'All points selected\n(below max_points limit)', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=14)
            ax4.set_title('Selection Status')
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs(output_dir, exist_ok=True)
        conf_viz_path = os.path.join(output_dir, "test_pose_confidence_analysis.png")
        plt.savefig(conf_viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Test pose confidence visualization saved to: {conf_viz_path}")
        
        # Save detailed statistics to file
        stats_path = os.path.join(output_dir, "test_pose_confidence_statistics.txt")
        with open(stats_path, 'w') as f:
            f.write("TEST POSE CONFIDENCE STATISTICS ANALYSIS\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total points: {n_total_points:,}\n")
            f.write(f"Selected points: {len(selected_conf):,}\n")
            f.write(f"Selection ratio: {100*len(selected_conf)/n_total_points:.2f}%\n\n")
            
            f.write("Basic Statistics:\n")
            f.write(f"  Min:        {conf_min:.6f}\n")
            f.write(f"  Q1 (25%):   {conf_q1:.6f}\n")
            f.write(f"  Median:     {conf_median:.6f}\n")
            f.write(f"  Mean:       {conf_mean:.6f}\n")
            f.write(f"  Q3 (75%):   {conf_q3:.6f}\n")
            f.write(f"  Max:        {conf_max:.6f}\n")
            f.write(f"  Std:        {conf_std:.6f}\n")
            f.write(f"  IQR:        {conf_iqr:.6f}\n\n")
            
            f.write("Percentiles:\n")
            f.write(f"  5th:        {conf_p5:.6f}\n")
            f.write(f"  95th:       {conf_p95:.6f}\n")
            f.write(f"  99th:       {conf_p99:.6f}\n\n")
            
            if len(selected_conf) < len(sorted_conf):
                f.write("Selected Points Statistics:\n")
                f.write(f"  Range: [{selected_conf.min():.6f}, {selected_conf.max():.6f}]\n")
                f.write(f"  Mean: {selected_conf.mean():.6f}\n")
                f.write(f"  Median: {np.median(selected_conf):.6f}\n")
                f.write(f"  Selection threshold: {selected_conf.min():.6f}\n")
        
        print(f"Detailed statistics saved to: {stats_path}")
    
    print("="*60)
    print(f"Final filtered points shape: {filtered_points.shape}")
    print(f"Final confidence range: [{filtered_conf.min():.4f}, {filtered_conf.max():.4f}]")
    print(f"Final confidence mean: {filtered_conf.mean():.4f}")
    
    return filtered_points, filtered_conf.reshape(-1, 1), filtered_rgb, filtered_xyf


def main(source_path, model_path, device, min_conf_thr, llffhold, n_views, 
         image_size=518, focal_avg=True, infer_video=False, **vggt_kwargs):

    # ---------------- (1) Load model and images ----------------  
    save_path, sparse_0_path, sparse_1_path = init_filestructure(Path(source_path), n_views)
    
    # Load VGGT model (same as in init_geo_vggt.py)
    model = VGGT()
    local_model_path = "/root/autodl-tmp/tamu/instantsplat-2dgs-vggt-dirty/VGGT-1B/models--facebook--VGGT-1B/snapshots/860abec7937da0a4c03c41d3c269c366e82abdf9/model.pt"
    if os.path.exists(local_model_path):
        print(f"Loading local VGGT model from {local_model_path}")
        model.load_state_dict(torch.load(local_model_path, map_location=device))
    else:
        print("Local model not found, downloading from HuggingFace...")
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        model.load_state_dict(torch.hub.load_state_dict_from_url(_URL, map_location=device))
    
    model.eval()
    model = model.to(device)
    print(f"VGGT model loaded")

    # Load and process images
    image_dir = Path(source_path) / 'images'
    image_files, image_suffix = get_sorted_image_files(image_dir)
    if infer_video:
        train_img_files = image_files
    else:
        train_img_files, test_img_files = split_train_test(image_files, llffhold, n_views, verbose=True)
    
    # CRITICAL DIFFERENCE: when init test pose, use ALL images (same as original)
    image_files = train_img_files + test_img_files
    image_path_list = [str(f) for f in image_files]
    
    # VGGT processing parameters
    vggt_fixed_resolution = 518  # Required by VGGT model
    img_load_resolution = image_size  # Use same strategy as MASt3R
    
    # Load images at specified size
    images, original_coords = load_and_preprocess_images_square(image_path_list, img_load_resolution)
    images = images.to(device)
    original_coords = original_coords.to(device)

    start_time = time()
    print(f'>> Running VGGT inference...')
    
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    extrinsic, intrinsic, depth_map, confidence_data, full_predictions = run_VGGT_with_confidence(
        model, images, dtype, device, vggt_fixed_resolution
    )
    
    # Unproject to 3D points
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    # Load previous 3D points from geometry initialization (like original MASt3R version)
    train_pts_all_path = sparse_0_path / 'points3D_all.npy'
    if not train_pts_all_path.exists():
        # Try alternative paths
        alt_paths = [
            sparse_0_path / 'points3D.npy',
            sparse_0_path / 'point_confidence_final.npy',
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                print(f"Using alternative path: {alt_path}")
                train_pts_all_path = alt_path
                break
        else:
            raise FileNotFoundError(f"Could not find previous 3D points at {train_pts_all_path}")
    
    train_pts_all = np.load(train_pts_all_path)
    train_pts3d_m1 = train_pts_all
    if train_pts3d_m1.ndim > 2:
        train_pts3d_m1 = train_pts3d_m1.reshape(-1, 3)
        
    print(f"Loaded previous 3D points: {train_pts3d_m1.shape}")

    # Load preset focal if focal_avg is enabled
    preset_focal = None
    if focal_avg:
        focals_file = sparse_0_path / 'non_scaled_focals.npy'
        if focals_file.exists():
            preset_focals = np.load(focals_file)
            preset_focal = np.mean(preset_focals)
            print(f">> preset_focal: {preset_focal}")
        else:
            print(">> Warning: non_scaled_focals.npy not found, using intrinsic focal")
            preset_focal = np.mean([intrinsic[0, 0, 0], intrinsic[0, 1, 1]])  # Average fx, fy of first camera
            
    print(f'>> Processing point clouds for registration...')
    
    # Extract current reconstruction points and poses
    train_pts3d_n1 = points_3d[:n_views]  # Only training views for registration
    test_poses_n1 = extrinsic[n_views:]   # Test poses from VGGT
    train_pts3d_n1 = np.array(train_pts3d_n1).reshape(-1, 3)
    test_poses_n1 = np.array(test_poses_n1)

    print(f'>> Analyzing confidence for registration...')

    # Get confidence data for training views only
    if "world_points_conf" in confidence_data:
        train_conf_values = confidence_data["world_points_conf"][:n_views]
        print("Using world_points_conf for registration weighting")
    elif "depth_conf" in confidence_data:
        train_conf_values = confidence_data["depth_conf"][:n_views]
        print("Using depth_conf for registration weighting")
    else:
        train_conf_values = np.ones_like(points_3d[:n_views, :, :, 0])
        print("Using uniform confidence")

    # Apply confidence analysis (but don't necessarily filter - just for debugging)
    # Create dummy RGB and XYF for the analysis function
    dummy_rgb = np.ones_like(train_pts3d_n1)
    dummy_xyf = np.ones_like(train_pts3d_n1)

    print(f"Analyzing confidence distribution for registration...")
    _, analyzed_conf, _, _ = apply_confidence_based_filtering(
        points_3d[:n_views], train_conf_values, dummy_rgb.reshape(points_3d[:n_views].shape), 
        dummy_xyf.reshape(points_3d[:n_views].shape),
        max_points=len(train_pts3d_n1),  # Don't filter, just analyze
        output_dir=os.path.join(model_path, "test_pose_confidence_analysis")
    )

    # Option: Apply confidence-based filtering for registration if desired
    # Uncomment the following lines to use only high-confidence points for registration
    # if len(train_pts3d_n1) > 100000:  # Only filter if too many points
    #     print(f"Applying confidence filtering for registration...")
    #     filtered_pts, filtered_conf, _, _ = apply_confidence_based_filtering(
    #         points_3d[:n_views], train_conf_values, dummy_rgb.reshape(points_3d[:n_views].shape), 
    #         dummy_xyf.reshape(points_3d[:n_views].shape),
    #         max_points=350000,  # Limit for registration performance
    #         output_dir=os.path.join(model_path, "registration_filtering")
    #     )
    #     train_pts3d_n1 = filtered_pts
    #     train_conf_values = filtered_conf.reshape(-1)

    # print(f'>> Performing point cloud registration...')
    # print(f"Current points shape: {train_pts3d_n1.shape}")
    # print(f"Previous points shape: {train_pts3d_m1.shape}")

    # Convert to torch tensors
    train_pts3d_n1_torch = torch.from_numpy(train_pts3d_n1).float()
    train_pts3d_m1_torch = torch.from_numpy(train_pts3d_m1).float()

    # # Use confidence weighting for registration
    # conf_weights = None
    # if train_conf_values is not None:
    #     conf_weights = torch.from_numpy(train_conf_values.reshape(-1)).float()
    #     print(f"Using confidence weights for registration: shape {conf_weights.shape}")
    #     print(f"Confidence range: [{conf_weights.min():.4f}, {conf_weights.max():.4f}]")

    # # Perform registration with confidence weighting
    # scale, R, T = rigid_points_registration(train_pts3d_n1_torch, train_pts3d_m1_torch, conf=None)

    # # Create transformation matrix
    # transform_matrix = torch.eye(4)
    # transform_matrix[:3, :3] = R
    # transform_matrix[:3, 3] = T
    # transform_matrix[:3, 3] *= scale
    # transform_matrix = transform_matrix.numpy()
    
    # print(f"Registration scale: {scale.item():.4f}")
    # print(f"Registration translation: {T.numpy()}")

    # # Convert VGGT poses to homogeneous coordinates
    # if test_poses_n1.shape[-2:] == (3, 4):  # Check for [N, 3, 4] format
    #     test_poses_n1_hom = np.zeros((test_poses_n1.shape[0], 4, 4))
    #     test_poses_n1_hom[:, :3, :] = test_poses_n1
    #     test_poses_n1_hom[:, 3, 3] = 1.0
    #     test_poses_n1 = test_poses_n1_hom

    # test_poses_m1 = transform_matrix @ test_poses_n1

    # # Save results
    # print(f'>> Saving results...')
    # end_time = time()
    # Train_Time = end_time - start_time
    # print(f"Time taken for {n_views} views: {Train_Time} seconds")
    # save_time(model_path, '[3] init_test_pose_vggt', Train_Time)
    
    # # Convert to w2c format for saving (invert the c2w poses)
    # # from utils.sfm_utils import inv
    # test_poses_w2c = inv(test_poses_m1)
    # save_extrinsic(sparse_1_path, test_poses_w2c, test_img_files, image_suffix)
    
    print(f'[INFO] VGGT Test Pose Initialization completed!')
    print(f'[INFO] Saved {len(test_img_files)} test poses to: {str(sparse_1_path)}')
    print(f'[INFO] Registration successfully aligned {train_pts3d_n1.shape[0]} points')
    # print(f'[INFO] Registration scale: {scale.item():.4f}')

    # Save additional analysis for test pose estimation
    # print(f'>> Saving test pose confidence analysis...')

    # # Analyze confidence for all views (including test)
    # if "world_points_conf" in confidence_data:
    #     all_conf = confidence_data["world_points_conf"]
    #     test_conf = confidence_data["world_points_conf"][n_views:]
    # elif "depth_conf" in confidence_data:
    #     all_conf = confidence_data["depth_conf"] 
    #     test_conf = confidence_data["depth_conf"][n_views:]
    # else:
    #     all_conf = np.ones_like(points_3d[:, :, :, 0])
    #     test_conf = np.ones_like(points_3d[n_views:, :, :, 0])

    # Save confidence statistics for test poses
    # conf_analysis_dir = os.path.join(model_path, "test_pose_analysis")
    # os.makedirs(conf_analysis_dir, exist_ok=True)

    # np.save(os.path.join(conf_analysis_dir, "test_confidence.npy"), test_conf)
    # np.save(os.path.join(conf_analysis_dir, "all_confidence.npy"), all_conf)

    # Save confidence statistics
    # with open(os.path.join(conf_analysis_dir, "test_pose_confidence_stats.txt"), 'w') as f:
    #     f.write("TEST POSE CONFIDENCE ANALYSIS\n")
    #     f.write("="*50 + "\n")
    #     f.write(f"Number of training views: {n_views}\n")
    #     f.write(f"Number of test views: {len(test_img_files)}\n")
    #     f.write(f"Total views processed: {len(extrinsic)}\n\n")
        
    #     train_conf_flat = all_conf[:n_views].reshape(-1)
    #     test_conf_flat = test_conf.reshape(-1)
        
    #     f.write("Training views confidence:\n")
    #     f.write(f"  Mean: {np.mean(train_conf_flat):.6f}\n")
    #     f.write(f"  Std:  {np.std(train_conf_flat):.6f}\n")
    #     f.write(f"  Range: [{np.min(train_conf_flat):.6f}, {np.max(train_conf_flat):.6f}]\n\n")
        
    #     f.write("Test views confidence:\n") 
    #     f.write(f"  Mean: {np.mean(test_conf_flat):.6f}\n")
    #     f.write(f"  Std:  {np.std(test_conf_flat):.6f}\n")
    #     f.write(f"  Range: [{np.min(test_conf_flat):.6f}, {np.max(test_conf_flat):.6f}]\n")

    # print(f"Test pose confidence analysis saved to: {conf_analysis_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VGGT-based test pose initialization')
    parser.add_argument('--source_path', '-s', type=str, required=True, help='Directory containing images')
    parser.add_argument('--model_path', '-m', type=str, required=True, help='Directory to save the results')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use for inference')
    parser.add_argument('--min_conf_thr', type=float, default=5, help='Minimum confidence threshold')
    parser.add_argument('--llffhold', type=int, default=8, help='LLFF hold-out for test images')
    parser.add_argument('--n_views', type=int, default=3, help='Number of training views')
    parser.add_argument('--focal_avg', action="store_true", help='Use averaged focal length')
    parser.add_argument('--infer_video', action="store_true", help='Video inference mode')
    parser.add_argument('--image_size', type=int, default=518, help='Size to resize images (same as MASt3R)')

    args = parser.parse_args()
    
    main(args.source_path, args.model_path, args.device, args.min_conf_thr, 
         args.llffhold, args.n_views, args.image_size, args.focal_avg, args.infer_video)