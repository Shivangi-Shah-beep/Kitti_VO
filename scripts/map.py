# map.py

class Map:
    def __init__(self, local_window_size: int):
        self.local_window_size = local_window_size
        self.map_points = {}       # map_point_id -> MapPoint
        self.active_map_points = {}  # map_point_id -> MapPoint
        self.keyframes = {}        # frame_id -> Frame
        self.active_keyframes = {}  # frame_id -> Frame
        self.viewer_update_required = False

    def set_viewer_update_required(self, flag: bool):
        self.viewer_update_required = flag

    def add_map_point(self, map_point):
        if map_point.id not in self.map_points:
            self.map_points[map_point.id] = map_point
            self.active_map_points[map_point.id] = map_point
            self.set_viewer_update_required(True)
            return True
        return False

    def add_keyframe(self, frame):
        print("Adding new Keyframe")
        if frame.frame_id not in self.keyframes:
            self.keyframes[frame.frame_id] = frame
            self.active_keyframes[frame.frame_id] = frame

            if len(self.active_keyframes) > self.local_window_size:
                oldest_id = next(iter(self.active_keyframes))
                self.remove_active_keyframe(oldest_id)

            self.set_viewer_update_required(True)
            return True
        return False

    def remove_active_keyframe(self, frame_id):
        if frame_id in self.active_keyframes:
            print(f"Remove Active Frame: {frame_id}")
            frame = self.active_keyframes[frame_id]

            for feature in frame.feature_ptrs:
                if feature.map_point is not None:
                    feature.map_point.remove_observe(feature)

            del self.active_keyframes[frame_id]
            self.clean_map()
            return True
        return False

    def clean_map(self):
        to_remove = [mp_id for mp_id, mp in self.active_map_points.items() if mp.get_observation_count() == 0]
        for mp_id in to_remove:
            self.active_map_points[mp_id].is_local_point = False
            del self.active_map_points[mp_id]

    def remove_active_map_point(self, map_point_id):
        if map_point_id in self.active_map_points:
            map_point = self.active_map_points[map_point_id]
            if map_point.get_observation_count() == 0:
                map_point.is_local_point = False
                del self.active_map_points[map_point_id]
                return True
        return False

    def get_all_keyframes(self):
        return list(self.keyframes.values())

    def get_all_landmarks(self):
        return list(self.map_points.values())
    
    def get_active_keyframes(self):
        return list(self.active_keyframes.values())

    def get_active_map_points(self):
        return list(self.active_map_points.values())

