import math
import random
import numpy as np

class Route(object):
    def __init__(self, route_id, start_stop, end_stop, route_length, max_speed, route_speed_history):
        self.route = []
        self.maximum_velocity = 0
        self.variant_velocity = 0

        self.sigma = 1.5
        self.route_id = route_id
        self.route_max_speed = max_speed
        self.speed_history = route_speed_history
        self.speed_limit = 15

        self.start_stop = start_stop
        self.end_stop = end_stop
        self.distance = route_length

    def route_update(self, current_time, effective_period):
        current_hour = effective_period[min(current_time//3600, len(effective_period) -1)]
        v = np.clip(math.log(random.lognormvariate(self.speed_history.loc[current_hour], self.sigma)), 2, 15)
        # v = math.log(random.lognormvariate(self.speed_history.loc[current_hour], self.sigma))
        self.speed_limit = min(self.route_max_speed, max(int(v), 0))
