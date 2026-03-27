from enum import Enum, auto
import numbers
import numpy as np


class BusState(Enum):
    HOLDING = auto()
    WAITING_ACTION = auto()
    DWELLING = auto()
    TRAVEL = auto()


class Bus(object):
    def __init__(self, bus_id, trip_id, launch_time, direction, routes, stations):
        self.bus_id = bus_id
        self.trip_id = trip_id
        self.trip_id_list = [trip_id]
        self.launch_time = launch_time
        self.direction = direction

        self.routes_list = routes
        self.stations_list = stations
        self.in_station = True
        self.passengers = np.array([]) # list of passengers on bus
        self.capacity = 50 # upper bound of passengers on bus
        self.current_speed = 0. # current speed of bus

        self.trip_turn = len(self.trip_id_list)
        self.effective_station = self.stations_list[:round(len(self.stations_list) / 2)] if self.direction else self.stations_list[round(len(self.stations_list) / 2) - 1:] # 从所有站点中抽取有效站点
        self.last_station = self.effective_station[0] # 初始化首站
        self.next_station = self.effective_station[1] # 初始化次站
        self.last_station_dis = 0. # 上一站到当前站的距离
        self.route_index = {(route.start_stop, route.end_stop): route for route in routes} # GPT优化方案1 构建索引字典
        self.next_station_dis = self.current_route.distance # 当前站到下一站的距离
        self.absolute_distance = 0. if self.direction else len(self.stations_list) // 2 * 500 # 在上行时，绝对距离从0开始，下行时从11500开始
        self.trajectory = [] # 轨迹记录
        self.trajectory_dict = {} # 轨迹字典
        for station in self.effective_station:
            self.trajectory_dict[station.station_name] = []

        self.obs = [] # 状态值
        self.forward_bus = None # 前车对象
        self.backward_bus = None  # 后车对象
        self.forward_headway = 360. # 前车车头时距
        self.backward_headway = 360. # 后车车头时距
        self.reward = None # 奖励值

        self.alight_num = 0. # 下车人数
        self.board_num = 0. # 上车人数
        self.back_to_terminal_time = None

        self.acceleration = 3 # 加速度
        self.deceleration = 5 # 刹车加速度

        self.state = BusState.HOLDING  # 初始状态：在站内上下客
        self.on_route = True # 是否在路上，如果在路上，为True，否则为False，用于判断是否到达终点站

        self.holding_time = 0. # 停站时间，用于上下乘客
        self.dwelling_time = 0. # 驻站时间，用于执行动作，停车等待

        self.headway_dif = []
        self.is_unhealthy = False # False if the bus is healthy, True if the bus is unhealthy, then terminate env early

        # record of stop intervals [station_name, start_time, end_time]
        self.stop_records = []
        self._stop_start_time = None
        self._stop_station = None

    @property
    def occupancy(self):
        return str(len(self.passengers)) + '/' + str(self.capacity)

    # decide if the negative or positive of step_length, when direction == 1, step_length > 0, vise versa
    @property
    def direction_int(self):
        return 1 if self.direction else -1

    # effective_route is effective routes for every bus, same as effective_station
    @property
    def effective_route(self):
        return self.routes_list[:round(len(self.routes_list) / 2)] if self.direction else self.routes_list[round(len(self.routes_list) / 2):]

    # searching for next_station when last_station changed
    @property
    def travel_distance(self):
        return self.absolute_distance if self.direction else sum([route.distance for route in self.effective_route]) - self.absolute_distance

    def next_station_func(self):
        return self.effective_station[self.last_station.station_id + self.direction_int] if self.direction else self.effective_station[-(self.last_station.station_id + self.direction_int + 1)]

    @property
    def station_after_the_next(self):
        # return the station after the next station
        return self.effective_station[self.last_station.station_id + 2 * self.direction_int] if self.direction else self.effective_station[-(self.last_station.station_id + 2 * self.direction_int + 1)]

    @property
    def station_before_the_last(self):
        # return the station before the last station
        return self.effective_station[self.last_station.station_id - 2 * self.direction_int] if self.direction else self.effective_station[-(self.last_station.station_id - 2 * self.direction_int + 1)]
    # searching for current_route when last_station and next_station changed
    # @property
    # def current_route(self):
    #     return list(filter(lambda i: i.start_stop == self.last_station.station_name and i.end_stop == self.next_station.station_name, self.effective_route))[0]

    # GPT优化方案1 构建索引字典
    @property
    def current_route(self):
        # 从字典中查找对应路段
        key = (self.last_station.station_name, self.next_station.station_name)
        return self.route_index[key]

    # When bus is arrived in a station, passengers have to alight and boarding.
    def exchange_passengers(self, current_time, debug):
        # Because we cannot mutate the list inter iteration. Record the index of every passenger we want to remove from
        # original passengers list then remove them with the pre-record index
        index_of_passenger_on_bus = []
        index_of_passenger_in_station = []
        # passengers alight from bus(self)
        for i, passenger in enumerate(self.passengers):
            if passenger.destination_station.station_name == self.next_station.station_name:
                passenger.arrived = True
                passenger.arrive_time = current_time
                self.alight_num += 1
                index_of_passenger_on_bus.append(i)
        # remove passengers from bus
        self.passengers = self.passengers[
            list(set(range(len(self.passengers))) - set(index_of_passenger_on_bus))] if len(
            self.passengers) > 0 else np.array([])
        # passengers boarding from station(self.next_station)
        for i, passenger in enumerate(self.next_station.waiting_passengers):
            if len(self.passengers) < self.capacity:
                passenger.boarded = True
                passenger.boarding_time = current_time
                passenger.travel_bus = self
                self.passengers = np.append(self.passengers, passenger)
                self.board_num += 1
                index_of_passenger_in_station.append(i)

        self.next_station.waiting_passengers = self.next_station.waiting_passengers[
            list(set(range(len(self.next_station.waiting_passengers))) - set(index_of_passenger_in_station))] if len(
            self.next_station.waiting_passengers) > 0 else np.array([])

        self.holding_time = max(self.alight_num, (self.board_num * 2.)) + 4.
        # print('Bus id: ',self.bus_id, ', stop id: ', self.last_station.station_id," ,holding time: ", self.holding_time)
        # if self.bus_id == 2 and debug:
        #     print('Bus: ', self.bus_id, ' at station: ', self.next_station.station_id ,' ,current time: ', current_time,' ,holding time: ', self.holding_time)
        self.alight_num = 0.
        self.board_num = 0.

    def bus_update(self):
        # update the bus state
        self.last_station = self.next_station
        self.next_station = self.next_station_func()
        self.last_station_dis = 0
        self.next_station_dis = self.current_route.distance

    def drive(self, current_time, action, bus_all, debug):
        # absolute_distance & last_station_dis is divided by 1000 as kilometers rather than meters. forward_headway & backward_headway
        # is divided by 60 minutes rather than seconds. passengers on bus, boarding passengers and alighting passengers are divided by self.capacity
        # step_length = 0, which means how long a bus moves in a time step, calculated by speeding up and original velocity.

        if self.state == BusState.TRAVEL:
            if self.next_station_dis <= self.current_speed:
                self.exchange_passengers(current_time, debug)  # self.holding_time is set in this function

                self.trajectory.append([self.next_station.station_name, current_time, self.absolute_distance, self.direction, self.trip_id])
                self.trajectory_dict[self.next_station.station_name].append([
                    self.next_station.station_name,
                    current_time + self.holding_time + 0.01,
                    self.absolute_distance,
                    self.direction,
                    self.trip_id
                ])

                self.arrive_station(current_time, bus_all, debug)
                self.state = BusState.HOLDING
                self.in_station = True
            else:
                self._advance_on_route()
        elif self.state == BusState.HOLDING:
            self._process_holding(current_time, bus_all, debug)
        elif self.state == BusState.WAITING_ACTION:
            self._start_dwelling(action)
        elif self.state == BusState.DWELLING:
            self._process_dwelling(current_time)
        else:
            # Recover gracefully if state was not initialised as expected
            self.state = BusState.TRAVEL
            self._advance_on_route()

    def _advance_on_route(self):
        if self.current_route.speed_limit >= self.current_speed:
            if self.current_route.speed_limit - self.current_speed > self.acceleration:
                step_length = (self.current_speed + self.acceleration / 2) * self.direction_int
                self.current_speed += self.acceleration
            else:
                step_length = (self.current_speed + self.current_route.speed_limit) * 0.5 * self.direction_int
                self.current_speed = self.current_route.speed_limit
        else:
            if self.current_speed - self.current_route.speed_limit > self.deceleration:
                step_length = (self.current_speed - self.deceleration / 2) * self.direction_int
                self.current_speed -= self.deceleration
            else:
                step_length = (self.current_speed + self.current_route.speed_limit) * 0.5 * self.direction_int
                self.current_speed = self.current_route.speed_limit

        self.last_station_dis += abs(step_length)
        self.next_station_dis -= abs(step_length)
        self.absolute_distance += step_length

    def _process_holding(self, current_time, bus_all, debug):
        if self.holding_time <= 1:
            self.holding_time = 0
            self._prepare_for_action(current_time, bus_all, debug)
        else:
            self.holding_time -= 1

    def _prepare_for_action(self, current_time, bus_all, debug):
        self.forward_bus = list(filter(lambda x: self.trip_id - 2 in x.trip_id_list, bus_all))
        self.backward_bus = list(filter(lambda x: self.trip_id + 2 in x.trip_id_list, bus_all))

        if self.next_station in self.effective_station[2:] and (len(self.forward_bus) != 0 or len(self.backward_bus) != 0):
            self.obs = [
                self.bus_id,
                self.last_station.station_id,
                current_time // 3600,
                self.direction,
                self.forward_headway,
                self.backward_headway,
                len(self.next_station.waiting_passengers) * 1.5 + self.current_route.distance / self.current_route.speed_limit
            ]
            all_route = self.routes_list[:len(self.routes_list) // 2] if self.direction else self.routes_list[len(self.routes_list) // 2:]
            speed_list = [all_route[i].speed_limit for i in range(len(all_route))]
            self.obs.extend(speed_list)

            def headway_reward(headway):
                return -abs(headway - 360)

            forward_reward = headway_reward(self.forward_headway) if len(self.forward_bus) != 0 else None
            backward_reward = headway_reward(self.backward_headway) if len(self.backward_bus) != 0 else None
            if forward_reward is not None and backward_reward is not None:
                weight = abs(self.forward_headway - 360) / (abs(self.forward_headway - 360) + abs(self.backward_headway - 360) + 1e-6)
                similarity_bonus = -abs(self.forward_headway - self.backward_headway) * 0.5
                self.reward = forward_reward * weight + backward_reward * (1 - weight) + similarity_bonus
            elif forward_reward is not None:
                self.reward = forward_reward
            elif backward_reward is not None:
                self.reward = backward_reward
            else:
                self.reward = -50

            if abs(self.forward_headway - 360) > 180 or abs(self.backward_headway - 360) > 180:
                self.reward -= 20
                self.is_unhealthy = True

        self.state = BusState.WAITING_ACTION

    def _start_dwelling(self, action):
        dwell_time = self._normalize_action(action)

        if (self.trip_id in [0, 1] and action is None) or dwell_time == 0:
            self.dwelling_time = 0
        else:
            self.dwelling_time = dwell_time

        self.state = BusState.DWELLING

    def _process_dwelling(self, current_time):
        if self.dwelling_time is None or self.dwelling_time <= 1:
            self.in_station = False
            if self._stop_start_time is not None:
                self.stop_records.append([
                    self._stop_station,
                    self._stop_start_time,
                    current_time
                ])
                self._stop_start_time = None
                self._stop_station = None
            self.dwelling_time = 0
            self.state = BusState.TRAVEL
        else:
            self.dwelling_time -= 1

    def _normalize_action(self, action):
        if action is None:
            return None
        if isinstance(action, numbers.Number):
            return float(action)
        if isinstance(action, np.ndarray):
            if action.size == 0:
                return None
            return float(action.reshape(-1)[0])
        if isinstance(action, (list, tuple)):
            if not action:
                return None
            return self._normalize_action(action[0])
        if hasattr(action, 'item'):
            try:
                return float(action.item())
            except (TypeError, ValueError):
                return None
        try:
            return float(action)
        except (TypeError, ValueError):
            return None

    def arrive_station(self, current_time, bus_all, debug):
        # Because we have to use the self.holding_time later, so we exchange passenger first when arrived a station
        # self.exchange_passengers(current_time) # self.holding_time is set in this function
        # Update forward_bus backward_bus and relative reward when a bus is arrived a station(except terminal)

        # record the start time and station when the bus stops
        self.current_speed = 0
        self._stop_start_time = current_time
        self._stop_station = self.next_station.station_name

        self.forward_bus = list(filter(lambda x: self.trip_id - 2 in x.trip_id_list, bus_all))
        if len(self.forward_bus) != 0:
            # print('there is a forward bus')
            forward_record = [record[1] for record in
                              self.forward_bus[0].trajectory_dict[self.next_station.station_name] if
                              record[-1] == self.trip_id - 2]
            # 当前车到达过当前站点，此时用当前时间减去前车到达当前站点的时间，再加上本车在当前站点的停车时间，减去前车在当前站点的停车时间，即为前车车头时距
            if len(forward_record) != 0:
                self.forward_headway = current_time + self.holding_time - min(forward_record)
            # 当前车没有到达当前站点，此时用当前车的绝对距离减去前车的绝对距离，再除以前车的速度，即为前车车头时距
            else:
                if not self.forward_bus[0].on_route:
                    forward_travel_distance = len(self.stations_list) // 2 * 500 + self.forward_bus[
                        0].travel_distance
                else:
                    forward_travel_distance = self.forward_bus[0].travel_distance
                # absolute_distance should be 10000 if direction is 0 else 0
                self.forward_headway = -(self.travel_distance - forward_travel_distance) / (
                        self.travel_distance / (current_time + self.holding_time - self.launch_time))
        else:
            # If there is no bus in the forward
            self.forward_headway = 360

        self.backward_bus = list(filter(lambda x: self.trip_id + 2 in x.trip_id_list, bus_all))
        self.backward_headway = self.backward_bus[0].forward_headway if len(self.backward_bus) != 0 else 360
        # self.backward_headway = 360
        # when the bus arrives at a station, drive() will switch the state to HOLDING so this logic only executes once
        self.absolute_distance += self.next_station_dis * self.direction_int
        # station_type == 0, means the next_station is terminal, then put this bus to terminal_bus rather than on_route
        # then change the direction of the bus.
        if self.next_station.station_type == 0 and self.on_route:
            self.on_route = False
            self.back_to_terminal_time = current_time
            self.last_station = self.effective_station[-1]
            self.direction = int(not self.direction)
            self.effective_station = self.stations_list[:round(len(self.stations_list) / 2)] if self.direction else self.stations_list[round(len(self.stations_list) / 2) - 1:]
            self.next_station = self.next_station_func()
        else:
            # if next_station is normal station, update last_station to its next_station, reset the relative distance of bus
            # if len(self.forward_bus) != 0:
            #     print('original_reward_place:', self.reward)
            station_id = self.last_station.station_id + 1 if self.direction else self.last_station.station_id - 1
            self.headway_dif.append([self.forward_headway - self.backward_headway, station_id])
            self.bus_update()

    # When a bus is re-launched from terminal, we have to reset the bus like a new bus we created, which means
    # we have to reset many attribute of the bus, then we add the trip_id to the trip history list. absolute_distance is 0
    # if it begins from terminal up, rather than 11500 if it begins from terminal down.

    def reset_bus(self, trip_num, launch_time):
        self.trip_id = trip_num
        self.trip_id_list.append(trip_num)
        self.launch_time = launch_time
        self.last_station = self.effective_station[0]

        self.forward_headway = 360
        self.backward_headway = 360

        self.last_station_dis = 0.
        self.next_station_dis = self.current_route.distance
        self.absolute_distance = 0. if self.direction else len(self.stations_list) // 2 * 500

        self.passengers = np.array([])
        self.current_speed = 0.
        self.holding_time = 0.
        self.back_to_terminal_time = None
        self.board_num = 0.
        self.alight_num = 0.
        self.in_station = False
        self.forward_bus = None
        self.backward_bus = None
        self.reward = None
        self.obs = []

        self.state = BusState.TRAVEL
        self.on_route = True
        self.trip_turn = len(self.trip_id_list)
        self.is_unhealthy = False # False if the bus is healthy, True if the bus is unhealthy, then terminate env early

