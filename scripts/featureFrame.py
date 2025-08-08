import numpy as np
class Feature:
    def __init__(self, frame, keypoint, prev_point=None):
        self.frame = frame                    # Reference to the Frame it belongs to
        self.point = keypoint                # cv2.KeyPoint
        self.prev_point = prev_point         # Optional cv2.Point2f (used for optical flow)
        self.map_point = None                # Associated MapPoint (or None)
        self.is_left_feature = True          # Whether it's from left image
        self.is_inlier = True                # For pose estimation filtering
    
    @property
    def position(self):
        return np.array(self.point.pt, dtype=np.float64)

class Frame:
    next_frame_id = 0  # Class-level counter for unique frame IDs

    def __init__(self):
        Frame.next_frame_id += 1
        self.frame_id = Frame.next_frame_id  # Unique ID

        self.imageL = None       # Left image (grayscale)
        self.imageR = None       # Right image (grayscale)

        self.feature_ptrs = []   # List of Feature objects (left image)
        self.right_feature_ptrs = []  # List of matched Features in right image

        self.T_wc = None         # SE3 pose: world → camera (set after pose estimation)
        self.T_d = None          # Optional: used for relative pose between frames

        self.is_keyframe = False # Set when promoted to keyframe

    def set_keyframe(self):
        self.is_keyframe = True
