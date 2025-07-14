from KittiStereoDataset import KittiDataset 
from stereo import Stereomatching
from feature_matcher import FeatureMatching 
from pose_estimation_2d import PoseEstimation3D, PoseEstimation2D
from trajectory_chaining_2D import TrajectoryChaining2D, TrajectoryChaining3D
from utils import plot_trajectories, ATE
import cv2
import numpy as np
import os

def main():
    sequence_path = "/home/shivangi_shah/kitti/odometry/color/dataset/sequences/00"
    calib_file = "/home/shivangi_shah/kitti/odometry/calib/dataset/sequences/00/calib.txt"
    gt= "/home/shivangi_shah/kitti/odometry/dataset/poses/00.txt"

    dataset = KittiDataset(sequence_path, calib_file)
    stereo = Stereomatching(calibration=dataset.get_calibration())
    gt_poses,gt_trajectory= KittiDataset.load_kitti_poses(gt)

    os.makedirs("output/depth", exist_ok=True)

    prev_left = None
    prev_kp = None
    prev_desc = None

    chain = TrajectoryChaining2D()
    chain3D= TrajectoryChaining3D()
    total_error=0
    frame=0


    for i in range(0,300):
        #gt_pose= gt_poses[i]
        gt_position = gt_trajectory[i]
        print(f"[GT] Frame {i}: x={gt_position[0]:.2f}, y={gt_position[1]:.2f}, z={gt_position[2]:.2f}")
        try:
            left, right = dataset[i]
        except Exception as e:
            print(f"[ERROR] Reading frame {i}: {e}")
            continue

        try:
            disparity = stereo.compute_disparity(left, right)
            depth = stereo.compute_depth(disparity)
        except Exception as e:
            print(f"[ERROR] Processing frame {i}: {e}")
            continue

        # Visualize disparity
        # disp_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        # disp_vis = np.uint8(disp_vis)
        # disp_colormap = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        # cv2.imshow("Disparity Colormap", disp_colormap)


        # Temporal Feature Matching and Pose Estimation
        if prev_left is not None:
            try:
                matcher = FeatureMatching(prev_left, left)
                match_img = matcher.draw_matches()

                estimator2D = PoseEstimation2D(
                    calib=dataset.get_calibration(),
                    matches=matcher.good,
                    kp1=matcher.kp1,
                    kp2=matcher.kp2
                )
                # cv2.imshow("Temporal Matches", match_img)

                # Estimate pose using 2D-2D correspondences
                estimator3D = PoseEstimation3D(
                    calib=dataset.get_calibration(),
                    matches=matcher.good,
                    kp1=matcher.kp1,
                    kp2=matcher.kp2
                )
                R1, t1 ,_ = estimator2D.EstimatePose()
                R, t, _ , pts_3d, pts_2d= estimator3D.estimate_pose(depth)
                error = estimator3D.compute_reprojection_error(pts_3d, pts_2d, R, t)
                #print(f"[INFO] Frame {i}: Reprojection error = {error:.4f}")
                
                chain.generate_trajectory(R1,t1)
                chain3D.generate_trajectory(R,t)

                # T_global = np.array(chain3D.get_t_global())
                # T_gt= np.array(gt_pose)
                # T_g= np.linalg.inv(T_global)
                
                # print(f"T_global= {T_global}, T_gt = {T_gt}")
                # print(f"T_gt * T_g = {T_gt @ T_g}")


                # Optional: accumulate and plot trajectory here
            except Exception as e:
                print(f"[ERROR] Pose estimation at frame {i}: {e}")

        prev_left = left

    plot_trajectories(gt_traj= gt_trajectory, traj2D= chain.get_trajectory(),traj3D= chain3D.get_trajectory())
    ATE(gt_traj= gt_trajectory,traj2D= chain.get_trajectory(), traj3D= chain3D.get_trajectory())

        # key = cv2.waitKey(1)
        # if key == 27:
        #     break



    
    # cv2.destroyAllWindows()

    

if __name__ == "__main__":
    main()
