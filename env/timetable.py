class Timetable(object):

    def __init__(self, launch_time, launch_turn, direction):
        self.launch_time = launch_time
        self.direction = direction
        self.launch_turn = launch_turn
        self.launched = False
