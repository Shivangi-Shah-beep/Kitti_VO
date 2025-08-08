import cv2
import numpy as np

class Stereomatching:
    def __init__(self, method='SGBM', num_disparities=64,block_size=9, calibration=None):
        self.calibration= calibration

        if method == "SGBM":
            self.matcher= cv2.StereoSGBM_create(minDisparity=0, numDisparities= num_disparities, blockSize= block_size, 
                                        P1=8 * 3 * block_size ** 2,
                                        P2=32 * 3 * block_size ** 2,
                                        disp12MaxDiff=1,
                                        uniquenessRatio=12,       
                                        speckleWindowSize=100,
                                        speckleRange=32,
                                        preFilterCap=4,
                                        mode= cv2.STEREO_SGBM_MODE_SGBM_3WAY
                                        )
        
        elif method == "BM":
            self.matcher = cv2.StereoBM_create(numDisparities= num_disparities, blockSize= block_size)
        
        else:
            raise ValueError("Unknown stereo matching method")

    def preprocess_image(self, img):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        return clahe.apply(img) if len(img.shape) == 2 else img

    def compute_disparity(self, left, right):
        # print("Entered compute_disparity")
        # if len(left.shape)==3:
        #     left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

        # if len(right.shape)==3:
        #     right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        disparity = self.matcher.compute(left, right).astype(np.float32) / 16.0
        disparity = cv2.medianBlur(disparity, 5)

        return disparity


    def compute_depth(self, disparity):
        fx = self.calibration['fx']
        baseline = self.calibration['baseline']
        depth = np.zeros_like(disparity)

        valid = disparity > 0
        depth[valid] = fx * baseline / disparity[valid]
        return depth
