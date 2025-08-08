import g2o
import numpy as np

# Create optimizer
optimizer = g2o.SparseOptimizer()
solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
optimizer.set_algorithm(g2o.OptimizationAlgorithmLevenberg(solver))

# Set camera intrinsics (fx, [cx, cy], baseline)
fx, fy = 500, 500
cx, cy = 320, 240
cam = g2o.CameraParameters(fx, np.array([cx, cy]), 0)
cam.set_id(0)
optimizer.add_parameter(cam)

# Define two poses
poses = [
    np.eye(4),
    np.array([[1, 0, 0, 0.1],
              [0, 1, 0, 0],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
]

for i, T_wc in enumerate(poses):
    R = T_wc[:3, :3]
    t = T_wc[:3, 3]
    se3 = g2o.SE3Quat(R, t)

    v = g2o.VertexSE3Expmap()
    v.set_id(i)
    v.set_estimate(se3)
    v.set_fixed(i == 0)  # Fix the first pose
    optimizer.add_vertex(v)

# Add a landmark
landmark = np.array([0, 0, 1.5])
v_lm = g2o.VertexSBAPointXYZ()
v_lm.set_id(100)
v_lm.set_estimate(landmark)
v_lm.set_marginalized(True)
optimizer.add_vertex(v_lm)

# Projected observations in both frames
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0,  0,  1]])

def project(T_wc, P_w):
    T_cw = np.linalg.inv(T_wc)
    P_c = T_cw[:3, :3] @ P_w + T_cw[:3, 3]
    u = fx * (P_c[0] / P_c[2]) + cx
    v = fy * (P_c[1] / P_c[2]) + cy
    return np.array([u, v])

obs0 = project(poses[0], landmark)
obs1 = project(poses[1], landmark)

# Add edges
for i, obs in zip([0, 1], [obs0, obs1]):
    edge = g2o.EdgeProjectXYZ2UV()
    edge.set_vertex(0, v_lm)
    edge.set_vertex(1, optimizer.vertex(i))
    edge.set_measurement(obs)
    edge.set_information(np.identity(2))
    edge.set_parameter_id(0, 0)

    kernel = g2o.RobustKernelHuber()
    edge.set_robust_kernel(kernel)

    optimizer.add_edge(edge)

# Run optimization
print("Vertices:", len(optimizer.vertices()))
print("Edges:", len(optimizer.edges()))

optimizer.initialize_optimization()
optimizer.set_verbose(True)
optimizer.optimize(10)

# Results
print("Optimized landmark:", v_lm.estimate())
print("Optimized pose 1:", optimizer.vertex(1).estimate().translation())

import g2o
import numpy as np

class Backend:
    def __init__(self):
        print("Starting backend")
        self.optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
        algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
        self.optimizer.set_algorithm(algorithm)
        self.optimizer.set_verbose(False)
        self.pose = {}
        self.pose_vertices=set()
        self.landmarks_added=set()
        self.landmarks={}
        self.camera= False

    def add_pose_vertices(self, map_obj):
        poses= map_obj.get_active_keyframes()
        frame_count = 0

        for frame in poses:
            id= frame.frame_id
            if id in self.pose_vertices:
                continue

            vertex = g2o.VertexSE3Expmap()
            vertex.set_id(id)

            T_wc = frame.T_wc
            R_wc = T_wc[:3, :3]
            t_wc = T_wc[:3, 3]
            se3 = g2o.SE3Quat(R_wc, t_wc)
            vertex.set_estimate(se3)

            # Fix the first 2 poses
            if id in (0, 1):
                vertex.set_fixed(True)
            else:
                vertex.set_fixed(False)

            self.optimizer.add_vertex(vertex)
            self.pose[id] = vertex
            self.pose_vertices.add(id)
            frame_count += 1

    def set_camera(self, camera):
        if self.camera is False:
            self.fx, self.fy = camera.fx, camera.fy
            self.cx, self.cy = camera.cx, camera.cy
            if self.optimizer.parameter(0) is None:
                cam = g2o.CameraParameters(self.fx, np.array([self.cx, self.cy]), 0)
                cam.set_id(0)
                self.optimizer.add_parameter(cam)
            self.camera= True
        
    def add_landmarks_and_edges(self, map_obj):
        if self.camera is None:
            print("No camera parameters initialized")
            return 


        # Add camera intrinsics only once

        landmarks = map_obj.get_active_map_points()
        print(f"[Backend] Active Landmarks: {len(landmarks)}")

        for lm in landmarks:
            lm_id = lm.id
            obs = lm.get_observations()

            if lm_id in self.landmarks_added:
                continue
            if len(obs) < 2:
                continue

            vertex = g2o.VertexSBAPointXYZ()
            vertex.set_id(lm_id + 10000)
            vertex.set_estimate(lm.world_point)
            vertex.set_marginalized(True)

            self.optimizer.add_vertex(vertex)
            self.landmarks[lm_id] = vertex  # Store reference
            self.landmarks_added.add(lm_id)

            for feat in obs:
                frame = feat.frame
                if frame is None or feat.point is None:
                    continue

                pt = feat.point.pt
                if pt is None or len(pt) != 2:
                    continue

                pt = np.array(pt, dtype=np.float64)
                if np.any(np.isnan(pt)) or np.any(np.isinf(pt)):
                    print(f"[WARNING] Invalid measurement (NaN/inf) for landmark {lm_id} in frame {frame.frame_id}")
                    continue


                pose_id = frame.frame_id
                if pose_id not in self.pose:
                    continue

                edge = g2o.EdgeProjectXYZ2UV()
                edge.set_vertex(0, self.optimizer.vertex(lm_id + 10000))
                edge.set_vertex(1, self.optimizer.vertex(pose_id))
                edge.set_measurement(pt)
                edge.set_information(np.identity(2))
                edge.set_parameter_id(0, 0)  # Link to existing camera param

                # robust_kernel = g2o.RobustKernelHuber()
                # edge.set_robust_kernel(robust_kernel)

                # print(f"[DEBUG] Adding edge: Point {lm.id} <-> Pose {frame.frame_id}")
                # print(f"  Vertex0 exists: {lm.id in self.landmarks}")
                # print(f"  Vertex1 exists: {frame.frame_id in self.pose}")
                # print(f"  Measurement: {feat.position}")

                if edge.vertex(0) is None:
                    print(f"[ERROR] Edge vertex 0 (landmark {lm_id + 10000}) is None")
                if edge.vertex(1) is None:
                    print(f"[ERROR] Edge vertex 1 (pose {pose_id}) is None")

                m = edge.measurement()
                if m.shape != (2,) or np.any(np.isnan(m)) or np.any(np.isinf(m)):
                    print(f"[ERROR] Invalid measurement for landmark {lm_id} in frame {pose_id}: {m}")

                self.optimizer.add_edge(edge)

        print(f"[Backend] Landmarks added: {len(self.landmarks_added)}")
    
    def compute_reprojection_error(self, map_obj):
        if self.camera is None:
            print("No camera parameters initialized")
            return

     
        total_error = 0.0
        count = 0

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
                point_cam = pose_est.map(point_3d)  # (3,) in camera coords

                if point_cam[2] <= 0:
                    continue  # Behind camera

                # Project to image plane
                u = self.fx * (point_cam[0] / point_cam[2]) + self.cx
                v = self.fy * (point_cam[1] / point_cam[2]) + self.cy
                projected = np.array([u, v])
                observed = np.array(feat.point.pt)

                error = np.linalg.norm(projected - observed)
                total_error += error
                count += 1

        if count == 0:
            print("[Reprojection Error] No valid observations.")
            return

        avg_error = total_error / count
        print(f"[Reprojection Error] Total = {total_error:.2f}, Count = {count}, Avg = {avg_error:.2f} px")


    def build_graph(self, map_):
        self.add_pose_vertices(map_)
        self.add_landmarks_and_edges(map_)
        self.compute_reprojection_error(map_)


    def optimize(self, max_iterations=10):
        print("Sanity check before optimization:")

        # 1. Check edges for missing vertices
        for edge in self.optimizer.edges():
            if edge.vertex(0) is None or edge.vertex(1) is None:
                print("[ERROR] Edge has a missing vertex.")

        # 2. Check pose vertices
        for pose_id in self.pose:
            v = self.optimizer.vertex(pose_id)
            if v is None:
                print(f"[ERROR] Pose vertex {pose_id} is missing from optimizer.")
            elif not isinstance(v, g2o.VertexSE3Expmap):
                print(f"[ERROR] Pose vertex {pose_id} has incorrect type.")

        # 3. Check landmark vertices
        for lm_id in self.landmarks:
            v = self.optimizer.vertex(lm_id + 10000)
            if v is None:
                print(f"[ERROR] Landmark vertex {lm_id + 10000} is missing from optimizer.")
            elif not isinstance(v, g2o.VertexSBAPointXYZ):
                print(f"[ERROR] Landmark vertex {lm_id + 10000} has incorrect type.")

        # 4. Check camera parameters
        cam_param = self.optimizer.parameter(0)
        if cam_param is None:
            print("[ERROR] Camera parameter with ID 0 not found.")
        elif not isinstance(cam_param, g2o.CameraParameters):
            print("[ERROR] Parameter ID 0 is not a valid CameraParameters object.")
        else:
            print("[OK] Camera parameters loaded.")
        
        param = self.optimizer.parameter(0)
        if param is not None:
            print(f"[DEBUG] Found camera parameter with ID 0")
        else:
            print(f"[ERROR] No camera parameter found with ID 0")

        print(f"[Optimizer] Starting optimization with {len(self.optimizer.vertices())} vertices and {len(self.optimizer.edges())} edges.")
        self.optimizer.initialize_optimization()
        self.optimizer.optimize(max_iterations)
        print("[Optimizer] Optimization complete.")









    






