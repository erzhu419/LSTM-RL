import numpy as np


class Bus(object):
    def __init__(self, bus_id, trip_id, launch_time, direction, routes, stations):
        self.bus_id = bus_id
        self.trip_id = trip_id
        self.trip_id_list = [trip_id]
        self.launch_time = launch_time
        self.direction = direction

        self.routes_list = routes
        self.stations_list = stations
        self.in_station = False
        self.passengers = np.array([])
        self.capacity = 50
        self.current_speed = 0.

        self.trip_turn = len(self.trip_id_list)
        self.line_station = self.stations_list[:round(len(self.stations_list) / 2)] if self.direction else self.stations_list[round(len(self.stations_list) / 2) - 1:]
        self.last_station = self.line_station[0]
        self.next_station = self.line_station[1]
        self.last_station_dis = 0.
        self.next_station_dis = self.current_route.distance
        self.absolute_distance = 0. if self.direction else len(self.stations_list) // 2 * 500
        self.trajectory = []
        self.trajectory_dict = {}
        for station in self.line_station:
            self.trajectory_dict[station.station_name] = []

        self.obs = []
        self.forward_bus = None
        self.backward_bus = None
        self.forward_headway = 360.
        self.backward_headway = 360.
        self.reward = None

        self.alight_num = 0.
        self.board_num = 0.
        self.w = 0.
        self.back_to_terminal_time = None

        self.acceleration = 3
        self.deceleration = 5

        self.holding = False
        self.held = False
        self.dwelling = False
        self.on_route = True

        self.holding_time = 0.
        self.dwelling_time = 0.

        self.headway_dif = []

    @property
    def occupancy(self):
        return str(len(self.passengers)) + '/' + str(self.capacity)

    # decide if the negative or positive of step_length, when direction == 1, step_length > 0, vise versa
    @property
    def direction_int(self):
        return 1 if self.direction else -1

    # line route is effective routes for every bus, same as line station
    @property
    def line_route(self):
        return self.routes_list[:round(len(self.routes_list) / 2)] if self.direction else self.routes_list[round(
            len(self.routes_list) / 2):]

    # searching for next_station when last_station changed
    @property
    def travel_distance(self):
        return self.absolute_distance if self.direction else sum([route.distance for route in self.line_route]) - self.absolute_distance

    def next_station_func(self):
        return self.line_station[self.last_station.station_id + self.direction_int] if self.direction else self.line_station[-(self.last_station.station_id + self.direction_int + 1)]

    # searching for current_route when last_station and next_station changed
    @property
    def current_route(self):
        return list(filter(
            lambda i: i.start_stop == self.last_station.station_name and i.end_stop == self.next_station.station_name,
            self.line_route))[0]

    # When bus is arrived in a station, passengers have to alight and boarding.
    def exchange_passengers(self, current_time):
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

        self.holding_time = max(self.alight_num, (self.board_num * 1.5)) + 3.5

        self.alight_num = 0.
        self.board_num = 0.

    def update(self):
        # update the bus state
        self.last_station = self.next_station
        self.next_station = self.next_station_func()
        self.last_station_dis = 0
        self.next_station_dis = self.current_route.distance

    def drive(self, current_time, action, bus_all):
        # absolute_distance & last_station_dis is divided by 1000 as kilometers rather than meters. forward_headway & backward_headway
        # is divided by 60 as minutes rather than seconds. passenger on bus, boarding passengers & alighting passengers are divided by self.capacity
        # self.obs = [self.forward_headway/360, self.backward_headway/360, len(self.passengers)/self.capacity] if self.on_route else [0] * 3
        # step_length = 0, which means how long a bus move in a time step, calculated by accelerate and original velocity.

        if self.next_station_dis <= self.current_speed and not self.holding and not self.dwelling:
            # when bus is arriving at station first time, set self.holding = True
            self.arrive_station(current_time, bus_all)
            self.holding = True
            # print('Bus: ', self.bus_id, ' ,holding at:', self.last_station.station_id)
            self.trajectory.append([self.last_station.station_name, current_time, self.absolute_distance, self.direction, self.trip_id])

            self.trajectory_dict[self.last_station.station_name].append([self.last_station.station_name, current_time + self.holding_time, self.absolute_distance,self.direction, self.trip_id])
            # self.trajectory_dict[self.last_station.station_name].append(
            #     [self.last_station.station_name, current_time + self.holding_time, self.absolute_distance,
            #      self.direction, self.trip_id])
            self.in_station = True
        elif not self.holding and not self.dwelling:
            # when bus on road
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
            # update the relative distance of stations, which is always positive. But the absolute distance is negative
            # if direction is negative, which means the absolute_distance start from 11500 rather than 0, so step_length
            # should be negative, until absolute_distance reduced to 0, which means this bus is arrive to terminal_up,
            # then the direction_int becomes to positive 1, over and over
            self.last_station_dis += abs(step_length)
            self.next_station_dis -= abs(step_length)
            self.absolute_distance += step_length

        elif self.dwelling and not self.holding:
            if self.dwelling_time <= 1:
                self.dwelling = False
                self.in_station = False
            else:
                self.dwelling_time -= 1

        elif self.holding and not self.dwelling:
            if self.holding_time <= 1:
                self.holding_time = 0
                if not self.held:
                    self.held = True
                    if self.trip_id not in [0, 1]:
                        self.obs = [self.bus_id, self.last_station.station_id, self.trip_id, self.direction,
                                    current_time//1800, (self.forward_headway - self.backward_headway)/60,
                                    len(self.next_station.waiting_passengers) * 1.5 +
                                    self.current_route.distance/self.current_route.speed_limit]
                else:
                    self.holding = False
                    self.dwelling = True
                    self.held = False

                    if action is None or self.trip_id in [0, 1] or action <= 1:

                        self.dwelling_time = 0
                    else:
                        self.dwelling_time = int(action)
            else:
                self.holding_time -= 1

    def arrive_station(self, current_time, bus_all):
        # Because we have to use the self.holding_time later, so we exchange passenger first when arrived a station
        self.exchange_passengers(current_time) # self.holding_time is set in this function
        # Update forward_bus backward_bus and relative reward when a bus is arrived a station(except terminal)
        self.forward_bus = list(filter(lambda x: self.trip_id - 2 in x.trip_id_list, bus_all))

        if len(self.forward_bus) != 0:
            # print('there is a forward bus')
            self.forward_bus = list(filter(lambda x: self.trip_id - 2 in x.trip_id_list, bus_all))
            if len(self.forward_bus) != 0:
                forward_record = [record[1] for record in
                                  self.forward_bus[0].trajectory_dict[self.next_station.station_name] if
                                  record[-1] == self.trip_id - 2]
                if len(forward_record) != 0:
                    self.forward_headway = current_time + self.holding_time - min(forward_record)
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
                self.forward_headway = 360

        self.backward_bus = list(filter(lambda x: self.trip_id + 2 in x.trip_id_list, bus_all))
        self.backward_headway = self.backward_bus[0].forward_headway if len(self.backward_bus) != 0 else 360
        # self.backward_headway = 360
        # when the bus arriving in a station, set self.holding = True. Then in outer loop, the iteration will skip this
        # function, to guarantee each bus arrive in each, this function just work ones
        self.absolute_distance += self.next_station_dis * self.direction_int
        # station_type == 0, means the next_station is terminal, then put this bus to terminal_bus rather than on_route
        # then change the direction of the bus.
        if self.next_station.station_type == 0 and self.on_route:
            self.on_route = False
            self.back_to_terminal_time = current_time
            self.last_station = self.line_station[-1]
            self.direction = int(not self.direction)
            self.line_station = self.stations_list[:round(len(self.stations_list) / 2)] if self.direction else self.stations_list[round(len(self.stations_list) / 2) - 1:]
            self.next_station = self.next_station_func()
        else:
            # if next_station is normal station, update last_station to its next_station, reset the relative distance of bus
            self.reward = np.exp(-abs(self.forward_headway - 360)) if len(self.forward_bus) != 0 else None
            # if len(self.forward_bus) != 0:
            #     print('original_reward_place:', self.reward)
            station_id = self.last_station.station_id + 1 if self.direction else self.last_station.station_id - 1
            self.headway_dif.append([self.forward_headway - self.backward_headway, station_id])
            self.update()

    # When a bus is re-launched from terminal, we have to reset the bus like a new bus we created, which means
    # we have to reset many attribute of the bus, then we add the trip_id to the trip history list. absolute_distance is 0
    # if it begins from terminal up, rather than 11500 if it begins from terminal down.

    def reset_bus(self, trip_num, launch_time):
        self.trip_id = trip_num
        self.trip_id_list.append(trip_num)
        self.launch_time = launch_time
        self.last_station = self.line_station[0]

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

        self.holding = False
        self.held = False
        self.on_route = True
        self.trip_turn = len(self.trip_id_list)