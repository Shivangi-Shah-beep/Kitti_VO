import cv2
import numpy as np

class PoseEstimation2D:
    def __init__(self, calib, matches, kp1, kp2):
        self.calibration = calib
        self.matches = matches
        self.kp1 = kp1
        self.kp2 = kp2

    def get_correspondences(self):
        pts1 = []
        pts2 = []
        for m in self.matches:
            pts1.append(self.kp1[m.queryIdx].pt)
            pts2.append(self.kp2[m.trainIdx].pt)
        return np.array(pts1, dtype=np.float32), np.array(pts2, dtype=np.float32)

    def EstimatePose(self):
        pts1, pts2 = self.get_correspondences()
        K = self.calibration["P2"][:3, :3]

        E, mask = cv2.findEssentialMat(
            pts1, pts2, K, method=cv2.RANSAC, threshold=1.0, prob=0.999
        )

        if E is None or E.shape == ():
            raise ValueError("Essential matrix estimation failed.")

        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask)

        self.E = E         
        self.pts1 = pts1
        self.pts2 = pts2
        return R, t, mask_pose

    def compute_epipolar_error(self):
        """
        Computes average symmetric epipolar error: x2^T E x1
        """
        if not hasattr(self, 'E') or not hasattr(self, 'pts1'):
            raise RuntimeError("Call EstimatePose() before computing error.")

        K = self.calibration["P2"][:3, :3]

        x1 = cv2.undistortPoints(self.pts1.reshape(-1, 1, 2), K, None).reshape(-1, 2)
        x2 = cv2.undistortPoints(self.pts2.reshape(-1, 1, 2), K, None).reshape(-1, 2)

       
        x1_h = np.hstack((x1, np.ones((x1.shape[0], 1))))
        x2_h = np.hstack((x2, np.ones((x2.shape[0], 1))))

        errors = np.abs(np.sum(x2_h * (self.E @ x1_h.T).T, axis=1)) 
        return np.mean(errors)

class PoseEstimation3D:
    def __init__(self, calib, matches, kp1, kp2):
        self.calibration = calib  # Dictionary with P2, fx, baseline
        self.matches = matches    # List of cv2.DMatch
        self.kp1 = kp1            # Keypoints in prev_left
        self.kp2 = kp2            # Keypoints in left

    def get_correspondences(self, depth_map):
        pts_3d = []
        pts_2d = []

        # Get intrinsics
        K = self.calibration["P2"][:3, :3]
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        for m in self.matches:
            u, v = self.kp1[m.queryIdx].pt
            u, v = int(round(u)), int(round(v))

            if 0 <= v < depth_map.shape[0] and 0 <= u < depth_map.shape[1]:
                Z = depth_map[v, u]
                if Z > 0 and not np.isnan(Z):
                    X = (u - cx) * Z / fx
                    Y = (v - cy) * Z / fy
                    pts_3d.append([X, Y, Z])
                    pts_2d.append(self.kp2[m.trainIdx].pt)

        return np.array(pts_3d, dtype=np.float32), np.array(pts_2d, dtype=np.float32)

    def estimate_pose(self, depth_map):
        pts_3d, pts_2d = self.get_correspondences(depth_map)
        if len(pts_3d) < 6:
            raise ValueError("Not enough valid 3D-2D correspondences for PnP.")

        K = self.calibration["P2"][:3, :3]

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=pts_3d,
            imagePoints=pts_2d,
            cameraMatrix=K,
            distCoeffs=None,
            flags=cv2.SOLVEPNP_ITERATIVE,
            confidence= 0.999
        )

        if not success:
            raise RuntimeError("PnP failed to estimate pose.")

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.flatten()

        return R, t, inliers, pts_3d, pts_2d
    
    def compute_reprojection_error(self, pts_3d, pts_2d, R, t):

        K = self.calibration["P2"][:3, :3]

        if t.shape == (3,):
            t = t.reshape(3, 1)

            # Project 3D points to 2D using the pose
        projected_points, _ = cv2.projectPoints(
                pts_3d, cv2.Rodrigues(R)[0], t, K, distCoeffs=None
            )
        projected_points = projected_points.reshape(-1, 2)

            # Compute Euclidean distance between observed and projected points
        errors = np.linalg.norm(pts_2d - projected_points, axis=1)
        return np.mean(errors)



