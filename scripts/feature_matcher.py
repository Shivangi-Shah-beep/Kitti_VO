import cv2

class FeatureMatching:
    def __init__(self, img1, img2, match_method="bf", lowe_ratio=0.70):
        self.img1 = img1
        self.img2 = img2
        self.match_method = match_method
        self.lowe_ratio = lowe_ratio

        # Detect keypoints and descriptors
        self.kp1, self.desc1, self.kp2, self.desc2 = self.detect_features()

        matches= self.match_detector()    

        self.good= self.good_matches(matches)
        
    
    def detect_features(self):
        detector = cv2.SIFT_create()
        kp1, desc1 = detector.detectAndCompute(self.img1, None)
        kp2, desc2 = detector.detectAndCompute(self.img2, None)

        return kp1, desc1, kp2, desc2
    
    def match_detector(self):
        if self.match_method=="bf":
            matcher= matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            raise ValueError("Only BFMatcher is currently supported")

        matches = matcher.knnMatch(self.desc1, self.desc2, k=2)
        return matches
    
    def good_matches(self, matches):
        good = []
        for m, n in matches:
            if m.distance < self.lowe_ratio * n.distance:
                good.append(m)
        #print(f"Number of good matches= {len(good)}")
        return good

    def draw_matches(self):
        match_img = cv2.drawMatches(
            self.img1, self.kp1, self.img2, self.kp2, self.good, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        return match_img

        

        