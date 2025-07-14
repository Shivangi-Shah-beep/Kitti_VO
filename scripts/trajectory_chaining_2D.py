import cv2
import numpy as np

class TrajectoryChaining2D:
    def __init__(self):
        self.trajectory = []
        self.T_global = np.eye(4)  # Global pose

    def generate_trajectory(self, R, t):
    # Invert the pose
        R_inv = R.T
        t_inv = -R_inv @ t.flatten()

        T = np.eye(4)
        T[:3, :3] = R_inv
        T[:3, 3] = t_inv

        self.T_global = self.T_global @ T
        position = self.T_global[:3, 3]
        self.trajectory.append(position)

        print(f"[Trajectory3D] Frame: {len(self.trajectory)} - Position: {position}")
        
    
    def get_trajectory(self):
        return self.trajectory
    

class TrajectoryChaining3D:
    def __init__(self):
        self.trajectory= []
        self.T_global= np.eye(4)
    
    def generate_trajectory(self, R, t):
        # Invert the pose
        R_inv = R.T
        t_inv = -R_inv @ t.flatten()

        T = np.eye(4)
        T[:3, :3] = R_inv
        T[:3, 3] = t_inv

        self.T_global = self.T_global @ T
        position = self.T_global[:3, 3]
        self.trajectory.append(position)

        print(f"[Trajectory3D] Frame: {len(self.trajectory)} - Position: {position}")


    
    def get_trajectory(self):
        return self.trajectory
    
    def get_t_global(self):
        return self.T_global

