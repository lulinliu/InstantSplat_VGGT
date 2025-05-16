import os
import argparse
import torch
import numpy as np
from pathlib import Path
from time import time

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


from dust3r.utils.device import to_numpy
from dust3r.utils.geometry import inv
from utils.sfm_utils import (save_intrinsics, save_extrinsic, save_points3D, save_time,
                             init_filestructure, get_sorted_image_files, split_train_test, load_images, rigid_points_registration)

def main(source_path, model_path, n_views, llffhold, focal_avg=True, infer_video=False):
    # ---------------- (1) Setup paths and load data ----------------  
    save_path, sparse_0_path, sparse_1_path = init_filestructure(Path(source_path), n_views)
    image_dir = Path(source_path) / 'images'
    image_files, image_suffix = get_sorted_image_files(image_dir)
    
    if infer_video:
        print("Infer video mode is enabled. Only using training images.")
        train_img_files = image_files
        test_img_files = []
    else:
        train_img_files, test_img_files = split_train_test(image_files, llffhold, n_views, verbose=True)
    
    # Skip if no test images
    if len(test_img_files) == 0:
        print("No test images to process. Exiting.")
        return

    print(f"Found {len(train_img_files)} training images and {len(test_img_files)} test images")
    start_time = time()

    # Load points3D from sparse/0
    train_pts_all_path = sparse_0_path / 'points3D_all.npy'
    if not train_pts_all_path.exists():
        # Try to load from points3D.txt and convert
        print("points3D_all.npy not found, attempting to read from points3D.txt")
        points3D_txt = sparse_0_path / 'points3D.txt'
        if points3D_txt.exists():
            from init_geo_vggt import load_points3D
            xyz, _, _ = load_points3D(points3D_txt)
            train_pts_all = xyz
            np.save(train_pts_all_path, train_pts_all)
        else:
            raise FileNotFoundError(f"Could not find points3D data at {train_pts_all_path} or {points3D_txt}")
    else:
        train_pts_all = np.load(train_pts_all_path)
    
    train_pts3d_m1 = train_pts_all

    # Load camera parameters from sparse/0
    cameras_txt = sparse_0_path / 'cameras.txt'
    images_txt = sparse_0_path / 'images.txt'
    
    if not cameras_txt.exists() or not images_txt.exists():
        raise FileNotFoundError(f"Required camera files not found: {cameras_txt} or {images_txt}")
    
    # Parse cameras.txt to get intrinsics
    with open(cameras_txt, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 5:  # SIMPLE_PINHOLE format
                # Camera format: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[fx, cx, cy]
                focal = float(parts[4])  # fx
                break
    
    print(f">> Using focal length: {focal}")
    
    # Parse images.txt to get training poses
    train_poses_w2c = []  # world to camera
    with open(images_txt, 'r') as f:
        lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line or line.startswith('#'):
                i += 1
                continue
                
            parts = line.strip().split()
            if len(parts) >= 8:
                # Image format: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                
                # Convert quaternion to rotation matrix
                R = np.zeros((3, 3))
                R[0, 0] = 1 - 2 * qy**2 - 2 * qz**2
                R[0, 1] = 2 * qx * qy - 2 * qz * qw
                R[0, 2] = 2 * qx * qz + 2 * qy * qw
                R[1, 0] = 2 * qx * qy + 2 * qz * qw
                R[1, 1] = 1 - 2 * qx**2 - 2 * qz**2
                R[1, 2] = 2 * qy * qz - 2 * qx * qw
                R[2, 0] = 2 * qx * qz - 2 * qy * qw
                R[2, 1] = 2 * qy * qz + 2 * qx * qw
                R[2, 2] = 1 - 2 * qx**2 - 2 * qy**2
                
                t = np.array([tx, ty, tz]).reshape(3, 1)
                
                # Create 3x4 extrinsic matrix [R|t]
                extrinsic = np.hstack((R, t))
                
                train_poses_w2c.append(extrinsic)
                
            # Skip the next line (contains point correspondences)
            i += 2
    
    train_poses_w2c = np.array(train_poses_w2c)[:n_views]
    
    # Convert world-to-camera to camera-to-world
    train_poses_c2w = []
    for pose_w2c in train_poses_w2c:
        R = pose_w2c[:, :3]
        t = pose_w2c[:, 3]
        R_inv = R.T
        t_inv = -R_inv @ t
        pose_c2w = np.eye(4)
        pose_c2w[:3, :3] = R_inv
        pose_c2w[:3, 3] = t_inv
        train_poses_c2w.append(pose_c2w)
    
    train_poses_c2w = np.array(train_poses_c2w)
    
    # Generate interpolated poses for test images
    n_train = len(train_img_files)
    n_test = len(test_img_files)
    
    if n_train < n_test:
        # Use interpolation when we have more test images than train images
        from utils.camera_utils import generate_interpolated_path
        n_interp = (n_test // (n_train-1)) + 1
        all_inter_pose = []
        
        for i in range(n_train-1):
            tmp_inter_pose = generate_interpolated_path(poses=train_poses_c2w[i:i+2, :3, :], n_interp=n_interp)
            all_inter_pose.append(tmp_inter_pose)
            
        all_inter_pose = np.concatenate(all_inter_pose, axis=0)
        all_inter_pose = np.concatenate([all_inter_pose, train_poses_c2w[-1][:3, :].reshape(1, 3, 4)], axis=0)
        
        indices = np.linspace(0, all_inter_pose.shape[0] - 1, n_test, dtype=int)
        sampled_poses = all_inter_pose[indices]
        sampled_poses = np.array(sampled_poses).reshape(-1, 3, 4)
        
        assert sampled_poses.shape[0] == n_test
        
        test_poses_c2w = []
        for p in sampled_poses:
            tmp_view = np.eye(4)
            tmp_view[:3, :3] = p[:3, :3]
            tmp_view[:3, 3] = p[:3, 3]
            test_poses_c2w.append(tmp_view)
            
        test_poses_c2w = np.stack(test_poses_c2w, 0)
    else:
        # Sample from existing poses when we have enough train images
        indices = np.linspace(0, train_poses_c2w.shape[0] - 1, n_test, dtype=int)
        test_poses_c2w = train_poses_c2w[indices]
    
    # Convert back to world-to-camera for saving
    test_poses_w2c = []
    for pose_c2w in test_poses_c2w:
        pose_w2c = inv(pose_c2w)
        test_poses_w2c.append(pose_w2c[:3, :])
    
    test_poses_w2c = np.array(test_poses_w2c)
    
    # Save results
    print(f'>> Saving results...')
    end_time = time()
    Train_Time = end_time - start_time
    print(f"Time taken for {n_views} views: {Train_Time} seconds")
    save_time(model_path, '[3] init_test_pose_vggt', Train_Time)
    save_extrinsic(sparse_1_path, test_poses_w2c, test_img_files, image_suffix)
    
    # Save test focals (use the same focal as training)
    # test_focals = np.repeat(focal, n_test)
    # org_imgs_shape = [0, 0]  # Placeholder, not used for focals
    # test_imgs_shape = [0, 0]  # Placeholder, not used for focals
    # save_intrinsics(sparse_1_path, test_focals, org_imgs_shape, test_imgs_shape, save_focals=True)
    
    print(f'[INFO] Test poses successfully saved to: {str(sparse_1_path)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Initialize test poses using VGGT data')
    parser.add_argument('--source_path', '-s', type=str, required=True, help='Directory containing images')
    parser.add_argument('--model_path', '-m', type=str, required=True, help='Directory to save the results')
    parser.add_argument('--llffhold', type=int, default=8, help='Image skip factor for test set')
    parser.add_argument('--n_views', type=int, default=3, help='Number of training views to use')
    parser.add_argument('--focal_avg', action="store_true", help='Use average focal length')
    parser.add_argument('--infer_video', action="store_true", help='Only use training images (no test set)')

    args = parser.parse_args()
    main(
        args.source_path,
        args.model_path,
        args.n_views,
        args.llffhold,
        args.focal_avg,
        args.infer_video
    )
