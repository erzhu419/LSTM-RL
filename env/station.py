from .passenger import Passenger
import numpy as np


class Station(object):
    def __init__(self, station_type, station_id, station_name, direction, od):
        np.random.seed(3)
        # if the station is terminal or not
        self.station_type = station_type
        # the id of stations
        self.station_id = station_id
        self.station_name = station_name
        # waiting passengers in this station
        self.waiting_passengers = np.array([])
        self.total_passenger = []
        # the direction is True if upstream, else False
        self.direction = direction
        # od is the passengers demand of every hour
        self.od = od

    def update(self, current_time, stations):
        # update station state

        # if self.od is not None:
        #     period_od = self.od.loc[effective_period[current_time // 3600]]
        #     for destination_name in effective_station_name:
        #         destination_demand_num = np.random.poisson(period_od[destination_name]/3600) if destination_name in period_od.index else 0
        #         for _ in range(destination_demand_num):
        #             destination = list(filter(lambda x: x.station_name == destination_name and x.direction == self.direction, stations))[0]
        #             passenger = Passenger(current_time, self, destination)
        #             self.waiting_passengers = np.append(self.waiting_passengers, passenger)
        #             self.total_passenger.append(passenger)
        #
        #     sorted(self.waiting_passengers, key=lambda i: i.appear_time)

        if self.od is not None:
            # effective_period_str = effective_period[current_time//3600].strftime("%H:%M:%S")
            effective_period_str = '0'+str(6+current_time//3600)+':00:00' if 6+current_time//3600 < 10 else str(6+current_time//3600)+':00:00'
            period_od = self.od[effective_period_str]
            for destination_name in list(period_od):
            # for destination_name in effective_station_name:
                destination_demand_num = np.random.poisson(period_od[destination_name]/3600) if destination_name in period_od.keys() else 0
                for _ in range(destination_demand_num):
                    destination = list(filter(lambda x: x.station_name == destination_name and x.direction == self.direction, stations))[0]
                    passenger = Passenger(current_time, self, destination)
                    self.waiting_passengers = np.append(self.waiting_passengers, passenger)
                    self.total_passenger.append(passenger)
            sorted(self.waiting_passengers, key=lambda i: i.appear_time)
