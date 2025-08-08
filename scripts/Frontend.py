from featureFrame import Frame, Feature
from pinhole_camera import PinholeCamera
from map import Map
import numpy as np
import cv2
from mapPoint import MapPoint
# from Backend import Optimization

class FrontEnd:
    def __init__(self, K, baseline, stereo_matcher, map_, backend_):
        print("Frontend")
        self.K = K
        self.baseline = baseline
        self.stereo = stereo_matcher
        self.feature_detector = cv2.GFTTDetector_create(150, 0.01, 20)
        self.trajectory = [np.eye(4)]
        self.current_frame = None
        self.map = map_  # <-- your provided map.py class
        self.camera = PinholeCamera(K[0, 0], K[1, 1], K[0, 2], K[1, 2], baseline)
        self.BA= False
        self.init=True
        self.prev_frame=None
        self.backend= backend_

    def step(self, frame):
        self.current_frame= frame
        if self.init == True:
            self.initialize()
        
        else:
            self.tracking()
        
        if self.current_frame.T_wc is not None:
            self.trajectory.append(self.current_frame.T_wc.copy())
        self.prev_frame= self.current_frame
        print("Moved to the next frame")
        print(f"length of previous features= {len(self.prev_frame.feature_ptrs)}")

    def initialize(self):
        self.current_frame.T_wc = np.eye(4)  # Equivalent to Sophus::SE3d()
        self.current_frame.T_d = np.eye(4)   # You can omit this if unused in your system

        features= self.create_left_feature()
        print(f"Number of features = {features}")
        self.match_in_right()

        self.create_map_points()  # ← triangulates stereo features to MapPoints

        self.current_frame.set_keyframe()         # ← promote this frame to keyframe
        self.map.add_keyframe(self.current_frame) 

        self.backend.set_camera(self.camera)
        self.backend.build_graph(self.map)

        print("Initialized")
        self.init= False
    
    def create_left_feature(self):
        mask = np.full(self.current_frame.imageL.shape, 255, dtype=np.uint8)

        for feat in self.current_frame.feature_ptrs:
            pt = feat.point.pt  # cv2.KeyPoint → (x, y)
            top_left = (int(pt[0] - 10), int(pt[1] - 10))
            bottom_right = (int(pt[0] + 10), int(pt[1] + 10))
            cv2.rectangle(mask, top_left, bottom_right, color=0, thickness=cv2.FILLED)

        # Detect new features with GFTT and the mask
        keypoints = self.feature_detector.detect(self.current_frame.imageL, mask)

        # Add each keypoint as a Feature to the frame
        for kp in keypoints:
            feature = Feature(self.current_frame, kp)
            self.current_frame.feature_ptrs.append(feature)

        return len(self.current_frame.feature_ptrs)
    
    def match_in_right(self):
        
        left_keypoints = []
        right_keypoints = []

        # Prepare initial points for LK tracking
        for feature in self.current_frame.feature_ptrs:
            pt = feature.point.pt  # cv2.KeyPoint → (x, y)
            left_keypoints.append(pt)
            right_keypoints.append(pt)  # initialize with same points

        # Convert to numpy arrays
        left_keypoints = np.array(left_keypoints, dtype=np.float32)
        right_keypoints = np.array(right_keypoints, dtype=np.float32)

        # Track features from left to right using optical flow
        right_keypoints, status, error = cv2.calcOpticalFlowPyrLK(
            self.current_frame.imageL,
            self.current_frame.imageR,
            left_keypoints,
            right_keypoints,
            winSize=(11, 11),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW
        )

        num_good_pts = 0
        self.current_frame.right_feature_ptrs = []  # clear previous data

        for i in range(len(status)):
            if status[i]:
                kp = cv2.KeyPoint(right_keypoints[i][0], right_keypoints[i][1], 1.0)
                feat = Feature(self.current_frame, kp)
                feat.is_left_feature = False
                self.current_frame.right_feature_ptrs.append(feat)
                num_good_pts += 1
            else:
                self.current_frame.right_feature_ptrs.append(None)

        print(f"[Right Match] Found {num_good_pts} points in the right image.")

        #self.visualize_features()
        return num_good_pts

    def visualize_features(self):
        imgL = cv2.cvtColor(self.current_frame.imageL.copy(), cv2.COLOR_GRAY2BGR)
        imgR = cv2.cvtColor(self.current_frame.imageR.copy(), cv2.COLOR_GRAY2BGR)

        # Draw keypoints in left image
        for feature in self.current_frame.feature_ptrs:
            pt = tuple(map(int, feature.point.pt))
            cv2.circle(imgL, pt, 2, (0, 255, 0), -1)

        # Draw matched points in right image
        for feat in self.current_frame.right_feature_ptrs:
            if feat is not None:
                pt = tuple(map(int, feat.point.pt))
                cv2.circle(imgR, pt, 2, (0, 0, 255), -1)

        # Combine the images horizontally
        stacked = np.hstack((imgL, imgR))
        cv2.imshow("Left (green) vs Right (red) Features", stacked)
        cv2.waitKey(0)  # 1 for real-time, 0 for manual step-through
    
    def tracking(self):
        trackingFeatureCount= self.trackFeatures()
        NumInliers= self.estimate_pose()

        print(f"Inliers= {NumInliers}")
        if NumInliers < 30:
            print("Replenish")
            self.update_observation()

            
            if len(self.map.get_active_keyframes()) >= 2:
                self.backend.set_camera(self.camera)
                self.backend.build_graph(self.map)
                keyframes = self.map.get_all_keyframes()
                self.backend.optimize(max_iterations=25)
                self.backend.update_map_poses(self.map)
                self.backend.update_map_landmarks(self.map)

            self.current_frame.set_keyframe()         # ← promote this frame to keyframe
            self.map.add_keyframe(self.current_frame) 

            self.create_left_feature()
            self.match_in_right()
            self.create_map_points()


    def trackFeatures(self):
        if self.prev_frame is None:
            print("[Track] Please initialize first.")
            return 0

        # Extract points from previous frame
        prev_pts = np.array([f.point.pt for f in self.prev_frame.feature_ptrs], dtype=np.float32)
        curr_pts = prev_pts.copy()

        # Optical flow from prev → current
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_frame.imageL,
            self.current_frame.imageL,
            prev_pts,
            curr_pts,
            winSize=(11, 11), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        # Keep only successfully tracked points
        good_prev_pts = prev_pts[status.ravel() == 1]
        good_curr_pts = curr_pts[status.ravel() == 1]

        good_count = 0
        # Clear current frame feature pointers
        self.current_frame.feature_ptrs = []

        # Map from previous points to features for propagation
        prev_pts_to_feature = {
            tuple(np.round(f.point.pt).astype(int)): f for f in self.prev_frame.feature_ptrs
        }

        # Add only successfully tracked features to current frame
        for pt_prev, pt_curr in zip(good_prev_pts, good_curr_pts):
            pt_prev_tuple = tuple(np.round(pt_prev).astype(int))
            prev_feature = prev_pts_to_feature.get(pt_prev_tuple, None)
            if prev_feature is None or prev_feature.map_point is None:
                continue

            kp = cv2.KeyPoint(pt_curr[0], pt_curr[1], 1.0)
            new_feature = Feature(self.current_frame, kp, prev_point=prev_feature.point.pt)
            new_feature.map_point = prev_feature.map_point
            self.current_frame.feature_ptrs.append(new_feature)
            good_count += 1

        print(f"[Track] Tracked and reused {good_count} features after optical flow tracking.")
        return good_count


    def create_map_points(self):
        max_norm = 50.0 
        frame_pose_inv = np.linalg.inv(self.current_frame.T_wc)
        landmark_count = 0

        # Compute disparity and depth map
        disparity = self.stereo.compute_disparity(self.current_frame.imageL, self.current_frame.imageR)
        depth_map = self.stereo.compute_depth(disparity)

        for i in range(len(self.current_frame.feature_ptrs)):
            featL = self.current_frame.feature_ptrs[i]

            if featL.map_point is not None:
                continue

            u, v = int(featL.point.pt[0]), int(featL.point.pt[1])
            if u < 0 or v < 0 or u >= depth_map.shape[1] or v >= depth_map.shape[0]:
                continue

            depth = depth_map[v, u]
            if depth <= 0 or depth > 50:
                continue

            # Backproject pixel to 3D (in camera frame)
            x = (u - self.camera.cx) * depth / self.camera.fx
            y = (v - self.camera.cy) * depth / self.camera.fy
            z = depth
            point_cam = np.array([x, y, z, 1.0])  # Homogeneous

            point_world = frame_pose_inv @ point_cam  # transform to world
            mp = MapPoint(point_world[:3])
            mp.add_observe(featL)
            featL.map_point = mp
            self.map.add_map_point(mp)
            landmark_count += 1

        print(f"[Depth-Based Triangulation] Created {landmark_count} map points.")
        return landmark_count


    
    def estimate_pose(self):
        rvec = None
        tvec = None
        world_points = []
        pixel_points = []
        features = []

        # Collect 2D-3D correspondences
        for feature in self.current_frame.feature_ptrs:
            if feature.map_point is not None:
                world_pt = feature.map_point.world_point
                world_points.append(world_pt)
                pixel_points.append(feature.point.pt)
                features.append(feature)
                feature.is_inlier = False

        if len(world_points) < 6:
            print("[PnP] Not enough points for pose estimation.")
            return 0

        world_points = np.array(world_points, dtype=np.float32)
        pixel_points = np.array(pixel_points, dtype=np.float32)

        # Camera intrinsics and distortion
        K = np.array([
            [self.camera.fx, 0, self.camera.cx],
            [0, self.camera.fy, self.camera.cy],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))  # Or your real distortion coefficients

        # PnP RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            world_points, pixel_points, K, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
            iterationsCount=150,
            reprojectionError=1,
            confidence=0.99
        )

        if success:
            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec.ravel()
            self.current_frame.T_wc = T  # Pose: world to camera

            # Mark inliers
            inliers = inliers.flatten()
            for i in inliers:
                features[i].is_inlier = True

            # Remove map points from outliers
            for i, feature in enumerate(features):
                if not feature.is_inlier:
                    feature.map_point = None
                    feature.is_inlier = True  # reset flag for future use

            print(f"[PnP] Pose estimated with {len(inliers)} inliers.")
            return len(inliers)

        else:
            print("[PnP] Pose estimation failed!")
            if self.prev_frame is not None and self.prev_frame.T_wc is not None:
                self.current_frame.T_wc = self.prev_frame.T_wc.copy()
            return 0

    def update_observation(self):
        for feature in self.current_frame.feature_ptrs:
            if feature.map_point is not None:
                feature.map_point.add_observe(feature)

    def get_trajectory(self):
        inverted_traj = []
        for T_wc in self.trajectory:
            R = T_wc[:3, :3]
            t = T_wc[:3, 3].reshape(3, 1)

            R_inv = R.T
            t_inv = -R_inv @ t

            T_cw = np.eye(4)
            T_cw[:3, :3] = R_inv
            T_cw[:3, 3] = t_inv.flatten()

            inverted_traj.append(T_cw)

        return inverted_traj
        
       
        






    
        
    










       
        



    
    # def match_in_right(self, left_img, right_img, pts_left):
    #     pts_right, status, _ = cv2.calcOpticalFlowPyrLK(
    #         left_img, right_img, pts_left, pts_left.copy(),
    #         winSize=(11, 11), maxLevel=3,
    #         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    #         flags=cv2.OPTFLOW_USE_INITIAL_FLOW
    #     )

    #     # Filter only valid matches
    #     good_left = pts_left[status.ravel() == 1]
    #     good_right = pts_right[status.ravel() == 1]
    #     return good_left, good_right

    # def leftFeature(self, left_img, right_img):
    #     kp = self.feature_detector.detect(left_img)
    #     pts_left = np.array([k.pt for k in kp], dtype=np.float32)

    #     good_left, good_right = self.match_in_right(left_img, right_img, pts_left)

    #     disparity = self.stereo.compute_disparity(left_img, right_img)
    #     depth = self.stereo.compute_depth(disparity)

    #     # Step 4: Get valid 3D points for matched keypoints
    #     valid_2d = []
    #     valid_3d = []

    #     for pt in good_left:
    #         u, v = int(pt[0]), int(pt[1])
    #         if 0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]:
    #             d = depth[v, u]
    #             if np.isfinite(d) and d > 0:
    #                 Z = d
    #                 X = (u - self.K[0, 2]) * Z / self.K[0, 0]
    #                 Y = (v - self.K[1, 2]) * Z / self.K[1, 1]
    #                 valid_2d.append([u, v])
    #                 valid_3d.append([X, Y, Z])

    #     self.prev_left = left_img
    #     self.prev_pts = np.array(valid_2d, dtype=np.float32)
    #     self.prev_pts_3D = np.array(valid_3d, dtype=np.float32)

    
    # def track_and_estimate_pose(self, curr_left):
    #     curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
    #         self.prev_left, curr_left, self.prev_pts, self.prev_pts.copy(),
    #         winSize=(11, 11), maxLevel=3,
    #         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    #     )

    #     prev_pts = self.prev_pts[status.ravel() == 1]
    #     curr_pts = curr_pts[status.ravel() == 1]
    #     pts_3d = self.prev_pts_3D[status.ravel() == 1]

    #     if len(prev_pts) < 6:
    #         print("[VO] Too few tracked points, skipping frame")
    #         return None, 0

    #     vis = cv2.cvtColor(curr_left.copy(), cv2.COLOR_GRAY2BGR)
    #     for (p_prev, p_curr) in zip(prev_pts, curr_pts):
    #         p_prev = tuple(np.round(p_prev).astype(int))
    #         p_curr = tuple(np.round(p_curr).astype(int))

    #         # Arrow from previous to current location
    #         cv2.arrowedLine(vis, p_prev, p_curr, (0, 255, 255), 1, tipLength=0.3)
    #         cv2.circle(vis, p_curr, 2, (0, 255, 0), -1)  # current point
    #         cv2.circle(vis, p_prev, 2, (0, 0, 255), -1)  # previous point

    #     # cv2.imshow("Tracked Point Motion", vis)
    #     # cv2.waitKey(0)  # or 0 for manual step-through

    #     print("changed left to prev_left")
    #     self.prev_left = curr_left
    #     self.prev_pts = curr_pts
    #     self.prev_pts_3D = pts_3d
        
    #     success, rvec, tvec, inliers = cv2.solvePnPRansac(
    #     pts_3d, curr_pts, self.K, None,
    #     flags=cv2.SOLVEPNP_ITERATIVE, reprojectionError=2.0, confidence=0.99
    #     )

    #     if not success:
    #         print("[VO] PnP failed")
    #         return None, 0

    #     R, _ = cv2.Rodrigues(rvec)
    #     T = np.eye(4)
    #     T[:3, :3] = R
    #     T[:3, 3] = tvec.ravel()



    #     return None, len(inliers), R, tvec



        


