import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pygame

class visualize(object):
    def __init__(self, env):
        self.env = env
        self.cnames = {
        'aquamarine':           '#7FFFD4',
        'beige':                '#F5F5DC',
        'bisque':               '#FFE4C4',
        'blanchedalmond':       '#FFEBCD',
        'blue':                 '#0000FF',
        'blueviolet':           '#8A2BE2',
        'brown':                '#A52A2A',
        'burlywood':            '#DEB887',
        'cadetblue':            '#5F9EA0',
        'chartreuse':           '#7FFF00',
        'chocolate':            '#D2691E',
        'coral':                '#FF7F50',
        'cornflowerblue':       '#6495ED',
        'darkblue':             '#00008B',
        'darkcyan':             '#008B8B',
        'darkgoldenrod':        '#B8860B',
        'darkgray':             '#A9A9A9',
        'darkgreen':            '#006400',
        'darkkhaki':            '#BDB76B',
        'darkmagenta':          '#8B008B',
        'darkolivegreen':       '#556B2F',
        'darkorange':           '#FF8C00',
        'darkorchid':           '#9932CC',
        'darkred':              '#8B0000',
        'darksalmon':           '#E9967A',
        'darkseagreen':         '#8FBC8F',
        'darkslateblue':        '#483D8B',
        'darkslategray':        '#2F4F4F',
        'darkturquoise':        '#00CED1',
        'darkviolet':           '#9400D3',
        'deeppink':             '#FF1493',
        'deepskyblue':          '#00BFFF',
        'dimgray':              '#696969',
        'dodgerblue':           '#1E90FF',
        'firebrick':            '#B22222',
        'forestgreen':          '#228B22',
        'gainsboro':            '#DCDCDC',
        'gold':                 '#FFD700',
        'goldenrod':            '#DAA520',
        'gray':                 '#808080',
        'green':                '#008000',
        'greenyellow':          '#ADFF2F',
        'honeydew':             '#F0FFF0',
        'hotpink':              '#FF69B4',
        'indianred':            '#CD5C5C',
        'ivory':                '#FFFFF0',
        'khaki':                '#F0E68C',
        'lavender':             '#E6E6FA',
        'lavenderblush':        '#FFF0F5',
        'lawngreen':            '#7CFC00',
        'lemonchiffon':         '#FFFACD',
        'lightblue':            '#ADD8E6',
        'lightcoral':           '#F08080',
        'lightcyan':            '#E0FFFF',
        'lightgoldenrodyellow': '#FAFAD2',
        'lightgreen':           '#90EE90',
        'lightgray':            '#D3D3D3',
        'lightpink':            '#FFB6C1',
        'lightsalmon':          '#FFA07A',
        'lightseagreen':        '#20B2AA',
        'lightskyblue':         '#87CEFA',
        'lightslategray':       '#778899',
        'lightsteelblue':       '#B0C4DE',
        'lightyellow':          '#FFFFE0',
        'limegreen':            '#32CD32',
        'linen':                '#FAF0E6',
        'magenta':              '#FF00FF',
        'maroon':               '#800000',
        'mediumaquamarine':     '#66CDAA',
        'mediumblue':           '#0000CD',
        'mediumorchid':         '#BA55D3',
        'mediumpurple':         '#9370DB',
        'mediumseagreen':       '#3CB371',
        'mediumslateblue':      '#7B68EE',
        'mediumspringgreen':    '#00FA9A',
        'mediumturquoise':      '#48D1CC',
        'mediumvioletred':      '#C71585',
        'midnightblue':         '#191970',
        'mintcream':            '#F5FFFA',
        'mistyrose':            '#FFE4E1',
        'moccasin':             '#FFE4B5',
        'navy':                 '#000080',
        'oldlace':              '#FDF5E6',
        'olivedrab':            '#6B8E23',
        'orange':               '#FFA500',
        'orangered':            '#FF4500',
        'orchid':               '#DA70D6',
        'palegoldenrod':        '#EEE8AA',
        'palegreen':            '#98FB98',
        'paleturquoise':        '#AFEEEE',
        'palevioletred':        '#DB7093',
        'papayawhip':           '#FFEFD5',
        'peachpuff':            '#FFDAB9',
        'peru':                 '#CD853F',
        'pink':                 '#FFC0CB',
        'plum':                 '#DDA0DD',
        'powderblue':           '#B0E0E6',
        'purple':               '#800080',
        'red':                  '#FF0000',
        'rosybrown':            '#BC8F8F',
        'royalblue':            '#4169E1',
        'saddlebrown':          '#8B4513',
        'salmon':               '#FA8072',
        'sandybrown':           '#FAA460',
        'seagreen':             '#2E8B57',
        'seashell':             '#FFF5EE',
        'sienna':               '#A0522D',
        'skyblue':              '#87CEEB',
        'slateblue':            '#6A5ACD',
        'slategray':            '#708090',
        'snow':                 '#FFFAFA',
        'springgreen':          '#00FF7F',
        'steelblue':            '#4682B4',
        'tan':                  '#D2B48C',
        'thistle':              '#D8BFD8',
        'tomato':               '#FF6347',
        'turquoise':            '#40E0D0',
        'violet':               '#EE82EE',
        'yellow':               '#FFFF00',
        'yellowgreen':          '#9ACD32'}
        
        self.bus_color = np.random.choice(list(self.cnames.keys()), env.max_agent_num)
        
    def draw_bus(self, surface, x, y, color, scale=1):
            # Scale dimensions with minimum size constraints
        bus_width = max(100 * scale, 20)  # Minimum width of 20
        bus_height = max(120 * scale, 24)  # Minimum height of 24
        wheel_radius = max(15 * scale, 5)  # Minimum wheel radius of 3
        offset = max(20 * scale, 4)  # Minimum offset of 4

        # Main body of the bus (rectangle)
        bus_body = [
            (x, y),  # Top-left
            (x + bus_width, y),  # Top-right
            (x + bus_width, y + bus_height),  # Bottom-right
            (x, y + bus_height),  # Bottom-left
        ]

        # Left side protrusion
        left_protrusion = [
            (x - offset, y + max(20 * scale, 4)),  # Top-left
            (x, y + max(20 * scale, 4)),  # Top-right
            (x, y + max(60 * scale, 12)),  # Bottom-right
            (x - offset, y + max(60 * scale, 12)),  # Bottom-left
        ]

        # Right side protrusion
        right_protrusion = [
            (x + bus_width, y + max(20 * scale, 4)),  # Top-left
            (x + bus_width + offset, y + max(20 * scale, 4)),  # Top-right
            (x + bus_width + offset, y + max(60 * scale, 12)),  # Bottom-right
            (x + bus_width, y + max(60 * scale, 12)),  # Bottom-left
        ]

        # Draw main body
        pygame.draw.polygon(surface, color, bus_body)

        # Draw side protrusions
        pygame.draw.polygon(surface, color, left_protrusion)
        pygame.draw.polygon(surface, color, right_protrusion)

        # Draw wheels as circles
        pygame.draw.circle(surface, color, (int(x + max(20 * scale, 4)), int(y + bus_height - max(10 * scale, 2))), int(wheel_radius))
        pygame.draw.circle(surface, color, (int(x + bus_width - max(20 * scale, 4)), int(y + bus_height - max(10 * scale, 2))), int(wheel_radius))

        # Add "window" as a white rectangle
        window_width = max(60 * scale, 12)
        window_height = max(40 * scale, 8)
        pygame.draw.rect(surface, (255, 255, 255), (x + max(20 * scale, 4), y + max(20 * scale, 4), window_width, window_height))

        # Add "logo" as a white rounded rectangle at the top
        logo_width = max(30 * scale, 6)
        logo_height = max(10 * scale, 2)
        pygame.draw.rect(surface, (255, 255, 255), (x + max(35 * scale, 7), y + max(5 * scale, 1), logo_width, logo_height))

        # Add "headlights" as two white circles
        headlight_radius = max(5 * scale, 1)
        pygame.draw.circle(surface, (255, 255, 255), (int(x + max(20 * scale, 4)), int(y + bus_height - max(25 * scale, 5))), int(headlight_radius))
        pygame.draw.circle(surface, (255, 255, 255), (int(x + bus_width - max(20 * scale, 4)), int(y + bus_height - max(25 * scale, 5))), int(headlight_radius))

        # Add "grille" as a white rectangle
        grille_width = max(30 * scale, 6)
        grille_height = max(5 * scale, 1)
        pygame.draw.rect(surface, (255, 255, 255), (x + max(35 * scale, 7), y + bus_height - max(30 * scale, 6), grille_width, grille_height))
    

    def render(self):
        
        screen_width = 2100
        screen_length = 1600
        screen = pygame.display.set_mode((screen_width, screen_length))
        
        pygame.display.set_caption("Bus Simulation")
        
        font = pygame.font.Font(None, 36)
        font_small = pygame.font.SysFont('arial', 20)
        # get current time in the format of "HH:MM:SS"
        hour = "0" + str(self.env.current_time//3600 + 6) if self.env.current_time//3600 + 6 < 10 else str(self.env.current_time//3600 + 6)
        minute = "0" + str(self.env.current_time%3600//60) if self.env.current_time%3600//60 < 10 else str(self.env.current_time%3600//60)
        second = "0" + str(self.env.current_time%60) if self.env.current_time%60 < 10 else str(self.env.current_time%60)
        
        # print current time on the top left corner
        current_time = hour + ":" + minute + ":" + second
        text_color = (0, 0, 0)
        
        time_surface = font.render(current_time, True, text_color)
        
        # define the upper and lower line y position of the bus
        upper_direction_y = screen_length/2 - 70
        lower_direction_y = screen_length/2 + 70        
                
        pygame_station_interval = 100
        pygame_simulation_distance_ratio = self.env.routes[0].distance / pygame_station_interval
        
        screen.fill((255, 255, 255))
        
        screen.blit(time_surface, (50, 100))
        
        
        station_location = []
        for i,station in enumerate(self.env.effective_station_name):
            # record station location and draw waiting passengers in each station
            station_location.append((pygame_station_interval * (i+1), upper_direction_y))
            
            for j in range(len(list(filter(lambda x: x.station_name == station and x.direction == True, self.env.stations))[0].waiting_passengers)):
                pygame.draw.rect(screen, (0, 0, 0), (pygame_station_interval *(i+1), upper_direction_y - 20 - j*10, 5, 5))

            station_location.append((pygame_station_interval * (i+1), lower_direction_y))

            for j in range(len(list(filter(lambda x: x.station_name == station and x.direction == False, self.env.stations))[0].waiting_passengers)):
                pygame.draw.rect(screen, (0, 0, 0), (pygame_station_interval *(i+1), lower_direction_y + 20 + j*10, 5, 5))
        
        # draw stations
        for i,sta_loc in enumerate(station_location):
            pygame.draw.circle(screen, (0, 0, 0), sta_loc, 5)

            # print station name on top of each station
            station_name = self.env.effective_station_name[i//2]
            station_surface = font.render(station_name, True, text_color)
            
            # Cause there are two directions, we only print the station name once
            if i % 2 == 0:
                screen.blit(station_surface, (sta_loc[0] - 10, sta_loc[1] + 60))
                
        # draw upper routes speed limit, red means slow, green means fast
        for i,station in enumerate(self.env.effective_station_name[:-1]):
            route_list = list(filter(lambda x: x.start_stop==station and x.end_stop==self.env.effective_station_name[i+1], self.env.routes))
            route = route_list[0]
            
            # Map the speed limit to a color between red and green
            speed = route.speed_limit
            red = int(255 * (15 - speed) / 13)  # Speed 2 -> Red 255, Speed 15 -> Red 0
            green = int(255 * (speed - 2) / 13)  # Speed 2 -> Green 0, Speed 15 -> Green 255
            color = (red, green, 0)
            # draw route status between stations
            # get the speed from self.env.route.speed_limit  map the speed of the route to the color, the faster the route, the greener the color, the slower the route, the redder the color
            pygame.draw.line(screen, color, station_location[i*2], station_location[i*2+2], 5)
            
            speed_surface = font_small.render(str(speed), True, text_color)
            screen.blit(speed_surface, ((station_location[i*2][0] + station_location[i*2+2][0])//2, (station_location[i*2][1] + station_location[i*2+2][1])//2 - 20))
        
        # draw lower routes speed limit, red means slow, green means fast
        for i,station in enumerate(self.env.effective_station_name[::-1][:-1]):
            route_list = list(filter(lambda x: x.start_stop==station and x.end_stop==self.env.effective_station_name[::-1][i+1], self.env.routes))
            route = route_list[0]
            
            # Map the speed limit to a color between red and green
            speed = route.speed_limit
            red = int(255 * (15 - speed) / 13)  # Speed 2 -> Red 255, Speed 15 -> Red 0
            green = int(255 * (speed - 2) / 13)  # Speed 2 -> Green 0, Speed 15 -> Green 255
            color = (red, green, 0)
            # draw route status between stations
            # get the speed from self.env.route.speed_limit  map the speed of the route to the color, the faster the route, the greener the color, the slower the route, the redder the color
            pygame.draw.line(screen, color, station_location[::-1][i*2], station_location[::-1][i*2+2], 5)
            
            speed_surface = font_small.render(str(speed), True, text_color)
            screen.blit(speed_surface, ((station_location[::-1][i*2][0] + station_location[::-1][i*2+2][0])//2, (station_location[::-1][i*2][1] + station_location[::-1][i*2+2][1])//2))
        # # draw buses
        for i,bus in enumerate(self.env.bus_all):
            
            if bus.on_route:
                
                color = self.bus_color[i]
                dis = np.clip(bus.absolute_distance/pygame_simulation_distance_ratio, 100, 2000)
                
                position = np.array((dis, upper_direction_y + 10)) if bus.direction == 1 else np.array((dis, lower_direction_y - 55))

                try:
                    # use exception handling to check if the color is valid
                    self.draw_bus(screen, position[0], position[1], color, scale=0.13)
                    # print bus occupancy
                    occupancy = bus.occupancy
                    occupancy_surface = font_small.render(occupancy, True, text_color)
                    screen.blit(occupancy_surface, (position[0], position[1] + 25))

                    holding_time = bus.holding_time if bus.holding_time is not None else 0
                    holding_time_surface = font_small.render('H:'+str(int(holding_time)), True, text_color)
                    screen.blit(holding_time_surface, (position[0] + 30, position[1] - 10))

                    dwelling_time = bus.dwelling_time if bus.dwelling_time is not None else 0
                    dwelling_time_surface = font_small.render('D:'+str(int(dwelling_time)), True, text_color)
                    screen.blit(dwelling_time_surface, (position[0] + 30, position[1] + 10))

                    bus_id_surface = font_small.render(str(bus.bus_id), True, text_color)
                    screen.blit(bus_id_surface, (position[0] - 20, position[1]))


                    
                except ValueError as e:
                    print(f"Invalid color value: {color}")
                    raise e
                
        pygame.display.flip()

                
    def plot(self, exp='0'):
        exp = str(exp)
        path = os.getcwd()
        plt.figure(figsize=(96, 24), dpi=300)
        total_headway = []
        
        global_time = [bus.trajectory[i][1] for bus in self.env.bus_all for i in range(len(bus.trajectory))]
        min_time, max_time = min(global_time), max(global_time)
    
        for bus in self.env.bus_all:
            # 提取当前 bus 的轨迹坐标
            x = [bus.trajectory[i][1] for i in range(len(bus.trajectory))]  # 时间坐标
            y = [bus.trajectory[i][2] for i in range(len(bus.trajectory))]  # 站点坐标
            plt.plot(x, y, label=bus.bus_id, color=self.cnames[self.bus_color[bus.bus_id]])
            plt.scatter(x, y, s=5)  # 添加散点，点大小更小更清晰

            # 记录 headway 数据
            total_headway.extend([bus.headway_dif[i][0] for i in range(len(bus.headway_dif))])

        # 绘制站点参考线
        x1 = np.linspace(min_time, max_time, num=500)  # 生成等间隔的时间点
        station_names = ['Terminal up'] + [f'X{i:02d}' for i in range(1, 21)] + ['Terminal down']
        for j in range(len(station_names)):
            y1 = [j * 500] * len(x1)
            plt.plot(x1, y1, color="red", linewidth=0.3, linestyle='-')
        
        
        # 坐标轴设置
        plt.xticks(fontsize=16)
        plt.yticks(ticks=[j * 500 for j in range(len(station_names))], labels=station_names, fontsize=16)
        plt.legend(fontsize=18)
        plt.xlabel('time', fontsize=20)
        plt.ylabel('station', fontsize=20)
        plt.title('result', fontsize=20)
        plt.xlim(min_time, max_time)  # 横坐标范围是全局时间
        plt.savefig(path + '/pic/exp ' + exp + ', bus trajectory.jpg')
        plt.close()

        total_passengers = []
        for station in self.env.stations:
            total_passengers.extend(station.total_passenger)

        total_waiting_time = [passenger.waiting_time for passenger in total_passengers if passenger.boarded]
        total_traveling_time = [passenger.travel_time for passenger in total_passengers if passenger.arrived]

        average_waiting_time = round(float(np.mean(total_waiting_time)), 2)
        average_travel_time = round(float(np.mean(total_traveling_time)), 2)

        plt.figure(figsize=(48, 12), dpi=400)
        max_x = 0
        max_stations = []
        for bus in self.env.bus_all:
            axis = range(len(bus.headway_dif))
            x = list(axis)
            y = [bus.headway_dif[i][0] for i in axis]
            plt.plot(x, y, label=bus.bus_id, color=self.cnames[self.bus_color[bus.bus_id]])
            max_x = max(len(bus.headway_dif), max_x)
            if max_x == len(bus.headway_dif):
                max_stations = [bus.headway_dif[i][1] for i in axis]
        plt.legend()
        positions = list(range(max_x))
        labels = max_stations
        plt.xticks(positions[::5], labels[::5], fontsize=16)  # Show every 5th label
        plt.ylim((-1000, 1000))
        plt.yticks(fontsize=16)
        plt.xlabel('station')
        plt.ylabel('difference of headway')
        plt.savefig(path+'/pic/exp ' + exp + ', bus headway.jpg')
        plt.close()

        fig = plt.figure()

        ax1 = fig.add_subplot(111)
        n, bins, patches = ax1.hist(total_waiting_time, bins=20, weights=np.ones_like(total_waiting_time)/float(len(total_waiting_time)), rwidth=0.85)
        ax1.plot()
        ax1.set_ylabel('probability')
        ax2 = ax1.twinx()
        y = np.cumsum(n)
        y /= y[-1]
        ax2.plot(bins[:-1] + 45, y, color='red', marker='o', linestyle='dashed', linewidth=1.5, label='cumulative probability')
        ax2.set_ylabel('cumulative probability')
        plt.legend()
        plt.savefig(path+'/pic/exp ' + exp + ', headway variance.jpg')
        total_headway = total_headway + [np.nan] * (len(total_waiting_time) - len(total_headway))
        df = pd.DataFrame({'traveling_time': total_traveling_time,
                           'waiting_time': total_waiting_time,
                           'headway': total_headway})
        df.to_csv(path+'/pic/exp ' + exp + '.csv', index=False)
        plt.close()



