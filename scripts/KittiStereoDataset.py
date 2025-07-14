import cv2
import numpy as np
import os

class KittiDataset:
    def  __init__(self, image_path, calib_path):
        self.left_dir = os.path.join(image_path, "image_2")
        self.right_dir = os.path.join(image_path, "image_3")
        self.calib_path = calib_path
        self.calib = self.load_calibration()
        self.image_pairs = self.load_image_pairs()
    
    def load_calibration(self):
        with open(self.calib_path, 'r') as f:
            lines = f.readlines()
        P2 = None
        P3 = None
        for line in lines:
            if line.startswith("P2:"):
                P2 = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 4)
            if line.startswith("P3:"):
                P3 = np.array([float(x) for x in line.strip().split()[1:]]).reshape(3, 4)

        fx = P2[0, 0]
        baseline = abs(P2[0, 3] - P3[0, 3]) / fx
        return {"P2": P2, "P3": P3, "fx": fx, "baseline": baseline}

    def load_image_pairs(self):
        left_images = sorted(os.listdir(self.left_dir))
        right_images = sorted(os.listdir(self.right_dir))
        return list(zip(left_images, right_images))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        left_img = cv2.imread(os.path.join(self.left_dir, self.image_pairs[idx][0]), cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(os.path.join(self.right_dir, self.image_pairs[idx][1]), cv2.IMREAD_GRAYSCALE)
        if left_img is None or right_img is None:
            raise ValueError(f"Missing image at index {idx}")
        return left_img, right_img

    def get_calibration(self):
        return self.calib
    
    @staticmethod
    def load_kitti_poses(pose_file):
        poses_4x4 = []
        positions = []

        with open(pose_file, 'r') as f:
            for line in f:
                vals = list(map(float, line.strip().split()))
                pose_matrix = np.array(vals).reshape(3, 4)
                T = np.vstack((pose_matrix, [0, 0, 0, 1]))  # 4x4 homogeneous pose

                poses_4x4.append(T)
                positions.append(pose_matrix[:, 3])  # tx, ty, tz

        return np.array(poses_4x4), np.array(positions)

