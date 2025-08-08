import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import sys
from scipy.spatial import cKDTree
from mpl_toolkits.mplot3d import Axes3D
from gtsam import PinholeCameraCal3_S2, Point3

gtsam_path = '/home/shivangi_shah/gtsam'  # Adjust as needed
sys.path.append(gtsam_path)
import gtsam
from gtsam import Point2, Point3

def plot_trajectories(gt_traj=None, traj2D=None, traj3D=None, save_path=None):
    plt.figure(figsize=(10, 7))
    length = None

    if traj3D is not None:
        length = len(traj3D)
    elif traj2D is not None:
        length = len(traj2D)

    if gt_traj is not None and length is not None:
        gt_traj = np.array(gt_traj)[:length]
        plt.plot(gt_traj[:, 0], gt_traj[:, 2], label="Ground Truth", linewidth=2)

    if traj2D is not None:
        traj2D = np.array(traj2D)
        plt.plot(traj2D[:, 0], traj2D[:, 2], label="VO 2D (Essential Matrix)", linestyle='--')

    if traj3D is not None:
        traj3D = np.array(traj3D)
        plt.plot(traj3D[:, 0], traj3D[:, 2], label="VO 3D (PnP)", linestyle='-.')

    plt.xlabel("X (meters)")
    plt.ylabel("Z (meters)")
    plt.title("Trajectory Comparison")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def statistical_outlier_removal(points, k=8, std_ratio=1.0):
    """Remove statistical outliers from a point cloud (landmarks).
    Args:
        points: Nx3 array of 3D points.
        k: Number of neighbors to consider.
        std_ratio: Standard deviation multiplier for outlier threshold.
    Returns:
        Filtered points (inlier array).
    """
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k+1)  # Self is closest (dist=0)
    mean_dists = dists[:, 1:].mean(axis=1)
    global_mean = mean_dists.mean()
    global_std = mean_dists.std()
    mask = np.abs(mean_dists - global_mean) < std_ratio * global_std
    return points[mask]

def ATE(gt_traj=None, traj2D=None, traj3D=None):
    if gt_traj is None or len(gt_traj) == 0:
        print("No ground truth to compare")
        return
    if (traj2D is None or len(traj2D) == 0) and (traj3D is None or len(traj3D) == 0):
        print("No trajectory as input")
        return
    gt_traj = np.array(gt_traj)
    if traj3D is not None:
        traj3D = np.array(traj3D)
        if traj3D.shape[-2:] == (4, 4):
            traj3D = traj3D[:, :3, 3]
        length = min(len(gt_traj), len(traj3D))
        aligned_3D = traj3D[:length]
        gt_traj = gt_traj[:length]
        errors_3D = np.linalg.norm(aligned_3D - gt_traj, axis=1)
        rmse_3D = np.sqrt(np.mean(errors_3D ** 2))
        print(f"[ATE] VO 3D RMSE: {rmse_3D:.4f} m")
    if traj2D is not None:
        traj2D = np.array(traj2D)
        length = min(len(traj2D), len(gt_traj))
        aligned_2D = traj2D[:length]
        gt_traj = gt_traj[:length]
        errors_2D = np.linalg.norm(aligned_2D - gt_traj, axis=1)
        rmse_2D = np.sqrt(np.mean(errors_2D ** 2))
        print(f"[ATE] VO 2D RMSE: {rmse_2D:.4f} m")


def get_point_xy(pt):
    """
    Helper to extract (x, y) from either a GTSAM Point2 or numpy-like array/tuple.
    """
    if hasattr(pt, 'x') and hasattr(pt, 'y'):
        return pt.x(), pt.y()
    elif hasattr(pt, '__getitem__') and len(pt) >= 2:
        return pt[0], pt[1]
    else:
        raise ValueError("Unknown point type provided.")

def reprojection_thresholding(poses, k, points3D_dict, points2D, threshold=15.0):
    """
    Remove outlier observations based on reprojection error threshold.
    
    Args:
        poses: List or dict of camera poses (each Pose3, one per frame).
        k: GTSAM Cal3_S2 intrinsics object.
        points3D_dict: Dict of {landmark_id: Point3 or np.ndarray}
        points2D: Dict of {frame_idx: {landmark_id: Point2 or np.ndarray}}
        threshold: Maximum reprojection error (pixels) to accept an observation.
    
    Returns:
        inlier_points3D: Dict of {landmark_id: Point3}
        inlier_points2D: Dict of {frame_idx: {landmark_id: Point2}}
    """
    from gtsam import PinholeCameraCal3_S2, Point3

    inlier_points3D = {}
    inlier_points2D = {}
    for frame_idx, observations in points2D.items():
        pose = poses[frame_idx]
        camera = PinholeCameraCal3_S2(pose, k)
        inlier_points2D[frame_idx] = {}
        for lid, obs2d in observations.items():
            if lid not in points3D_dict:
                continue
            pt3d = points3D_dict[lid]
            if isinstance(pt3d, np.ndarray):
                pt3d = Point3(*pt3d)
            try:
                proj = camera.project(pt3d)
            except RuntimeError:
                continue
            proj_x, proj_y = get_point_xy(proj)
            obs_x, obs_y = get_point_xy(obs2d)
            reproj_error = np.linalg.norm([proj_x - obs_x, proj_y - obs_y])
            if reproj_error < threshold:
                inlier_points2D[frame_idx][lid] = obs2d
                inlier_points3D[lid] = pt3d
    return inlier_points3D, inlier_points2D



def visualize_landmarks(landmarks_3D, points2D_buffer, image_buffer=None):
    print(f"Point3 from utils: {Point3}")
    lids = sorted(landmarks_3D.keys())
    cmap = plt.get_cmap('tab20')
    lid_to_color = {lid: cmap(i % 20) for i, lid in enumerate(lids)}

    fig = plt.figure(figsize=(15, 5))
    ax3d = fig.add_subplot(131, projection='3d')

    for lid, p in landmarks_3D.items():
        # Handle both NumPy and class-based Point3
        if isinstance(p, np.ndarray) and p.size == 3:
            ax3d.scatter(p[0], p[1], p[2], color=lid_to_color[lid], label=str(lid))
        elif hasattr(p, 'x') and hasattr(p, 'y') and hasattr(p, 'z'):
            ax3d.scatter(p.x(), p.y(), p.z(), color=lid_to_color[lid], label=str(lid))
        else:
            print(f"Warning: Landmark {lid} is not a recognized Point3/ndarray: {p}")
    ax3d.set_title("3D Landmarks")
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')

    for i in range(2):
        if len(points2D_buffer) < i + 1:
            continue
        ax2d = fig.add_subplot(1, 3, i + 2)
        pts2d_frame = points2D_buffer[-(i + 1)]
        if image_buffer is not None:
            ax2d.imshow(image_buffer[-(i + 1)], cmap='gray')
        for lid, p2d in pts2d_frame.items():
            if hasattr(p2d, "x") and hasattr(p2d, "y"):
                ax2d.scatter(p2d.x(), p2d.y(), color=lid_to_color.get(lid, 'r'), label=str(lid), s=30)
                ax2d.annotate(str(lid), (p2d.x(), p2d.y()), fontsize='x-small')
            elif isinstance(p2d, np.ndarray) and p2d.size == 2:
                ax2d.scatter(p2d[0], p2d[1], color=lid_to_color.get(lid, 'r'), label=str(lid), s=30)
                ax2d.annotate(str(lid), (p2d[0], p2d[1]), fontsize='x-small')
            else:
                print(f"Warning: Landmark {lid} in 2D projection is not a recognized Point2/ndarray: {p2d}")
        ax2d.set_title(f"2D Landmarks Frame {-i-1}")
        ax2d.axis('off')

    plt.tight_layout()
    plt.show()



import matplotlib.pyplot as plt
import numpy as np

def plot_traj(optimized_poses, original_poses=None, title="Trajectory"):
    """
    Plots 3D trajectory from pose matrices.
    Args:
        optimized_poses (list of np.ndarray): List of 4x4 pose matrices from BA.
        original_poses (list of np.ndarray): (Optional) List of original T_wc matrices.
        title (str): Title of the plot.
    """

    def extract_xyz(poses):
        xyz = []
        for pose in poses:
            pose = np.asarray(pose)
            if pose.shape == (4, 4):
                xyz.append(pose[:3, 3])  # Extract translation from 4x4 matrix
            elif pose.shape == (6,):
                xyz.append(pose[3:])     # Extract translation from [rvec, tvec]
            elif pose.shape == (3,):
                xyz.append(pose)         # Already a 3D position
            else:
                raise ValueError(f"Unsupported pose format: {pose.shape}")
        return np.array(xyz)


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if optimized_poses is not None and len(optimized_poses) > 0:
        opt_xyz = extract_xyz(optimized_poses)
        ax.plot(opt_xyz[:, 0], opt_xyz[:, 1], opt_xyz[:, 2], label='Optimized Trajectory', color='blue')

    if original_poses is not None and len(original_poses) > 0:
        orig_xyz = extract_xyz(original_poses)
        ax.plot(orig_xyz[:, 0], orig_xyz[:, 1], orig_xyz[:, 2], label='Original Trajectory (VO)', color='red', linestyle='--')

    ax.set_title(title)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.legend()
    ax.grid(True)
    plt.show()
