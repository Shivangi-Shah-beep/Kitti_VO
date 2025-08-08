import numpy as np

class PinholeCamera:
    def __init__(self, fx, fy, cx, cy, baseline):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.baseline = baseline

    def pixel2camera(self, pt):
        """Convert pixel coordinates to normalized camera coordinates"""
        x = (pt[0] - self.cx) / self.fx
        y = (pt[1] - self.cy) / self.fy
        return np.array([x, y])

    def get_pose_matrix(self):
        # Stereo baseline is along the X-axis (common for KITTI)
        T = np.array([[1, 0, 0, -self.baseline],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0]], dtype=np.float32)
        return T