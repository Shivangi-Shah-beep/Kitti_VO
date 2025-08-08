import g2o
import numpy as np

class Backend:
    def __init__(self):
        print("Starting backend")
        self.optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
        algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
        self.optimizer.set_algorithm(algorithm)
        self.optimizer.set_verbose(True)
        self.pose = {}
        self.pose_vertices=set()
        self.landmarks_added=set()
        self.landmarks={}
        self.camera= False
        self.residuals= []

    def add_pose_vertices(self, map_obj):
        poses = map_obj.get_active_keyframes()
        frame_count = 0

        if not poses:
            print("[ERROR] No keyframes available.")
            return

        fixed_id = min(f.frame_id for f in poses) # Fix the first keyframe

        for frame in poses:
            id = frame.frame_id
            if id in self.pose_vertices:
                continue

            vertex = g2o.VertexSE3Expmap()
            vertex.set_id(id)

            T_wc = frame.T_wc
            R_wc = T_wc[:3, :3]
            t_wc = T_wc[:3, 3]
            se3 = g2o.SE3Quat(R_wc, t_wc)
            vertex.set_estimate(se3)

            vertex.set_fixed(id == fixed_id)

            self.optimizer.add_vertex(vertex)
            print(f"[DEBUG] Pose vertex {id} added, fixed={vertex.fixed()}")
            self.pose[id] = vertex
            self.pose_vertices.add(id)
            frame_count += 1

    def set_camera(self, camera):
        if not self.camera:
            self.fx, self.fy = camera.fx, camera.fy
            self.cx, self.cy = camera.cx, camera.cy

            if self.optimizer.parameter(0) is None:
                cam = g2o.CameraParameters(self.fx, np.array([self.cx, self.cy]), 0)
                cam.set_id(0)
                self.optimizer.add_parameter(cam)

                if self.optimizer.parameter(0) is None:
                    print("[ERROR] Failed to add camera parameters to optimizer")
                else:
                    print("[Backend] Camera parameters added to optimizer")
            else:
                print("[Backend] Camera parameters already exist in optimizer")

            self.camera = True

        
    def add_landmarks_and_edges(self, map_obj):
        if self.camera is None:
            print("[ERROR] No camera parameters initialized")
            return

        landmarks = map_obj.get_active_map_points()
        print(f"[Backend] Active Landmarks: {len(landmarks)}")

        added = 0
        for lm in landmarks:
            lm_id = lm.id
            obs = lm.get_observations()

            if lm_id in self.landmarks_added:
                continue
            if len(obs) < 2:
                continue

            # Validate 3D point
            if lm.world_point is None or len(lm.world_point) != 3:
                print(f"[ERROR] Invalid 3D world point for landmark {lm_id}")
                continue

            world_point = np.array(lm.world_point, dtype=np.float64)
            if np.any(np.isnan(world_point)) or np.any(np.isinf(world_point)):
                print(f"[ERROR] Invalid 3D point (NaN or inf) for landmark {lm_id}")
                continue

            vertex = g2o.VertexSBAPointXYZ()
            vertex.set_id(lm_id + 10000)
            vertex.set_estimate(world_point)
            vertex.set_marginalized(True)
            self.optimizer.add_vertex(vertex)
            self.landmarks[lm_id] = vertex
            self.landmarks_added.add(lm_id)

            for feat in obs:
                frame = feat.frame
                if frame is None or feat.point is None:
                    print(f"[WARNING] Skipping feature with null frame or point")
                    continue

                pt = feat.point.pt
                if pt is None or len(pt) != 2:
                    print(f"[ERROR] Invalid 2D observation for landmark {lm_id} in frame {frame.frame_id}")
                    continue

                pt = np.array(pt, dtype=np.float64)
                if np.any(np.isnan(pt)) or np.any(np.isinf(pt)):
                    print(f"[ERROR] 2D measurement has NaN/inf for landmark {lm_id} in frame {frame.frame_id}")
                    continue

                pose_id = frame.frame_id
                if pose_id not in self.pose:
                    continue

                cam_param = self.optimizer.parameter(0)
                if cam_param is None:
                    print(f"[ERROR] Camera parameter ID 0 not found in optimizer. Skipping edge.")
                    continue


                edge = g2o.EdgeProjectXYZ2UV()
                

                v_landmark = self.optimizer.vertex(lm_id + 10000)
                v_pose = self.optimizer.vertex(pose_id)

                if v_landmark is None or v_pose is None:
                    print(f"[ERROR] One of the edge vertices is None: landmark {lm_id + 10000}, pose {pose_id}")
                    continue

                edge.set_vertex(0, v_landmark)
                edge.set_vertex(1, v_pose)


                edge.set_measurement(pt)
                edge.set_information(np.identity(2))
                edge.set_parameter_id(0, 0)

                # You can add the robust kernel later once stable
                # rk = g2o.RobustKernelHuber()
                # edge.set_robust_kernel(rk)

                # Validation checks
                if edge.vertex(0) is None:
                    print(f"[ERROR] Edge vertex 0 (landmark {lm_id + 10000}) is None")
                    continue
                if edge.vertex(1) is None:
                    print(f"[ERROR] Edge vertex 1 (pose {pose_id}) is None")
                    continue
                m = edge.measurement()
                if m.shape != (2,) or np.any(np.isnan(m)) or np.any(np.isinf(m)):
                    print(f"[ERROR] Invalid edge measurement: {m}")
                    continue
                
                rk = g2o.RobustKernelCauchy()
                rk.set_delta(5.0)
                edge.set_robust_kernel(rk)

                self.optimizer.add_edge(edge)

            #print(f"[Backend] Successfully added landmark {lm_id} and its edges.")
            added += 1
            # if added >= 5:
            #     break  # Only process one landmark to isolate crash

    
    def compute_reprojection_error(self, map_obj):
        if self.camera is False:
            print("[Reprojection Error] No camera parameters initialized")
            return

        total_error = 0.0
        count = 0
        frame_errors = {}  # Dict to hold per-frame error lists

        for lm in map_obj.get_active_map_points():
            if lm.id not in self.landmarks:
                continue

            point_3d = lm.world_point  # (3,)
            for feat in lm.get_observations():
                frame = feat.frame
                if frame is None or frame.frame_id not in self.pose or feat.point is None:
                    continue

                vertex_pose = self.pose[frame.frame_id]
                pose_est = vertex_pose.estimate()

                # Transform world point to camera frame
                point_cam = pose_est.map(point_3d)

                if point_cam[2] <= 0:
                    continue  # Point is behind the camera

                # Project to image plane
                u = self.fx * (point_cam[0] / point_cam[2]) + self.cx
                v = self.fy * (point_cam[1] / point_cam[2]) + self.cy
                projected = np.array([u, v])
                observed = np.array(feat.point.pt)

                error = np.linalg.norm(projected - observed)
                total_error += error
                count += 1

                # Store error per frame
                frame_errors.setdefault(frame.frame_id, []).append(error)

        if count == 0:
            print("[Reprojection Error] No valid observations.")
            return

        avg_error = total_error / count
        print(f"[Reprojection Error] Total = {total_error:.2f}, Count = {count}, Avg = {avg_error:.2f} px")

        # Print error per frame
        for fid, errs in frame_errors.items():
            avg = np.mean(errs)
            print(f"[Reprojection] Frame {fid} → Avg Error = {avg:.2f} px, N = {len(errs)}")



    def build_graph(self, map_):
        self.add_pose_vertices(map_)
        self.add_landmarks_and_edges(map_)
        self.compute_reprojection_error(map_)


    def optimize(self, max_iterations=10):
    # Print all vertices
        # print(f"\n[OPTIMIZER] Total Vertices: {len(self.optimizer.vertices())}")
        # for v_id, vertex in self.optimizer.vertices().items():
        #     estimate = None
        #     if isinstance(vertex, g2o.VertexSE3Expmap):
        #         se3 = vertex.estimate()
        #         estimate = se3.translation()
        #     elif isinstance(vertex, g2o.VertexSBAPointXYZ):
        #         estimate = vertex.estimate()
        #     else:
        #         estimate = str(vertex.estimate())
        #     print(f"  [Vertex] ID={v_id}, Type={type(vertex).__name__}, Estimate={estimate}")

        # # Print all edges
        # print(f"\n[OPTIMIZER] Total Edges: {len(self.optimizer.edges())}")
        # for i, edge in enumerate(self.optimizer.edges()):
        #     v0 = edge.vertex(0)
        #     v1 = edge.vertex(1)
        #     v0_id = v0.id() if v0 is not None else "None"
        #     v1_id = v1.id() if v1 is not None else "None"
        #     measurement = edge.measurement()
        #     print(f"  [Edge {i}] Vertex0 ID={v0_id}, Vertex1 ID={v1_id}, Measurement={measurement}")
        
        # param = self.optimizer.parameter(0)
        # if param is None:
        #     print("[ERROR] Optimizer has no camera parameter.")
        # else:
        #     print(f"[DEBUG] Camera parameter fx: {param.focal_length}, cx, cy: {param.principal_point}")

        print("\n[OPTIMIZER] Starting optimization...\n")
        self.optimizer.initialize_optimization()
        self.optimizer.optimize(max_iterations)
        print("\n[OPTIMIZER] Optimization complete.")


    def get_trajectory(self):
        """
        Returns a list of (tx, ty, tz) tuples for optimized camera poses,
        rotated 90 degrees anticlockwise around Z axis.
        """
        trajectory = []
        R = np.array([
            [0, 1, 0],
            [-1,  0, 0],
            [0,  0, 1]
        ])

        for id, vertex in sorted(self.pose.items()):
            est = vertex.estimate()
            t = est.translation()
            t_rotated = R @ np.array([t[0], t[1], t[2]])
            trajectory.append((t_rotated[0], t_rotated[1], t_rotated[2]))
        
        return trajectory

    def update_map_poses(self, map_obj):
        for frame in map_obj.get_active_keyframes():
            vertex = self.pose.get(frame.frame_id)
            if vertex is not None:
                se3 = vertex.estimate()
                T_wc = np.eye(4)
                T_wc[:3, :3] = se3.rotation().matrix()
                T_wc[:3, 3] = se3.translation()
                frame.T_wc = T_wc
                frame.need_update = True 
    
    def update_map_landmarks(self, map_obj):
        for lm in map_obj.get_active_map_points():
            vertex = self.landmarks.get(lm.id)
            if vertex is not None:
                pos = vertex.estimate()
                lm.world_point = pos

    def collect_reprojection_residuals(self, map_obj):
        for lm in map_obj.get_active_map_points():
            if lm.id not in self.landmarks:
                continue
            point_3d = lm.world_point
            for feat in lm.get_observations():
                frame = feat.frame
                if frame is None or frame.frame_id not in self.pose or feat.point is None:
                    continue
                vertex_pose = self.pose[frame.frame_id]
                pose_est = vertex_pose.estimate()
                point_cam = pose_est.map(point_3d)
                if point_cam[2] <= 0:
                    continue
                u = self.fx * (point_cam[0] / point_cam[2]) + self.cx
                v = self.fy * (point_cam[1] / point_cam[2]) + self.cy
                projected = np.array([u, v])
                observed = np.array(feat.point.pt)
                error = np.linalg.norm(projected - observed)
                self.residuals.append(error)
        return self.residuals













    






