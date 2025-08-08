import numpy as np
import weakref

class MapPoint:
    _next_map_point_id = 0

    def __init__(self, world_point=None):
        MapPoint._next_map_point_id += 1
        self.id = MapPoint._next_map_point_id
        self.world_point = np.array(world_point) if world_point is not None else None
        self._observation_count = 0
        self._obs_features = []

    @property
    def observation_count(self):
        return self._observation_count

    def add_observe(self, feature):
        """Add a feature observation (expects a shared Feature object)."""
        self._observation_count += 1
        self._obs_features.append(weakref.ref(feature))
        return True

    def remove_observe(self, feature):
        """Remove a feature observation if it exists."""
        self._observation_count -= 1
        found = False

        # Filter out the matching feature (dereferenced from weakref)
        new_obs = []
        for wf in self._obs_features:
            f = wf()
            if f is None or f == feature:
                if f == feature:
                    found = True
                continue
            new_obs.append(wf)
        
        self._obs_features = new_obs
        return found
    
    def get_observation_count(self):
        return self._observation_count
    
    def get_observations(self):
        return [f() for f in self._obs_features if f() is not None]

