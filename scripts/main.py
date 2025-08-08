from KittiStereoDataset import KittiDataset  # Make sure this is correct
from Frontend import FrontEnd  # From your core VO code
from trajectory_chaining_2D import TrajectoryChaining3D
from stereo import Stereomatching
from utils import plot_trajectories, ATE, plot_traj
import cv2
import numpy as np
from map import Map
from featureFrame import Frame
from Backend import Backend
# import faulthandler
# faulthandler.enable()

def umeyama_alignment(X, Y, with_scale=True):
    """
    Umeyama alignment: finds the best-fit similarity transform (scale, rotation, translation)
    aligning Y (estimated) to X (ground truth), both of shape (N, 3).
    Args:
        X (np.ndarray): Ground truth trajectory, shape (N, 3)
        Y (np.ndarray): Estimated trajectory to be aligned, shape (N, 3)
        with_scale (bool): If True, estimate scale; otherwise, only rigid alignment
    Returns:
        scale (float): Estimated scale factor
        R (np.ndarray): 3x3 rotation matrix
        t (np.ndarray): 3x1 translation vector
        Y_aligned (np.ndarray): Aligned estimated trajectory, shape (N, 3)
    """
    Y= Y[:, :3, 3]
    assert X.shape == Y.shape, "Input shapes must match and be Nx3."
    n, m = X.shape
    assert m == 3, "Points must be 3D."

    mean_X = np.mean(X, axis=0)
    mean_Y = np.mean(Y, axis=0)
    X_c = X - mean_X
    Y_c = Y - mean_Y

    cov = (Y_c.T @ X_c) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt

    if with_scale:
        var_Y = np.var(Y_c, axis=0).sum()
        scale = 1.0 / var_Y * np.sum(D * S.diagonal())
    else:
        scale = 1.0

    t = mean_X - scale * R @ mean_Y
    Y_aligned = scale * (R @ Y.T).T + t

    return scale, R, t, Y_aligned

def main():

    sequence_path = "/home/shivangi_shah/kitti/odometry/color/dataset/sequences/00"
    calib_file = "/home/shivangi_shah/kitti/odometry/calib/dataset/sequences/00/calib.txt"
    gt= "/home/shivangi_shah/kitti/odometry/dataset/poses/00.txt"

    dataset = KittiDataset(sequence_path, calib_file)
    gt_poses,gt_trajectory= KittiDataset.load_kitti_poses(gt)

    prev_img = None
    calib = dataset.load_calibration()
    K = calib["P2"][:3, :3]
    baseline = calib["baseline"]
    stereo_matcher= Stereomatching(calibration=dataset.get_calibration())
    map_= Map(15)
    back= Backend()
    frontend= FrontEnd(K, baseline, stereo_matcher, map_, back)
    Chain3D= TrajectoryChaining3D()
    
    

    for i in range(500):
        left, right= dataset[i]
        gt_position = gt_trajectory[i]
        print(f"[GT] Frame {i}: x={gt_position[0]:.2f}, y={gt_position[1]:.2f}, z={gt_position[2]:.2f}")

        frame= Frame()
        frame.imageL= left
        frame.imageR= right

        frontend.step(frame)
    
    
    
    trajectory= frontend.get_trajectory()
    gt_trajectory= gt_trajectory[:len(trajectory)]

    # print(f"shape of gt and traj= {gt_trajectory.shape} , {(np.array(trajectory)).shape}")

    # _,_,_, trajectory= umeyama_alignment(gt_trajectory, np.array(trajectory))

    # Now evaluate
    plot_trajectories(gt_traj=gt_trajectory, traj3D=trajectory)
    ATE(gt_traj=gt_trajectory, traj3D=trajectory)



    # plot_trajectories(gt_traj=gt_trajectory, traj3D=trajectory)
    # ATE(gt_traj=gt_trajectory, traj3D=trajectory)
        

if __name__=="__main__":
    main()

