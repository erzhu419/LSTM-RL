from collections import defaultdict
import matplotlib.pyplot as plt


def find_simultaneous_stops(buses):
    """Return a list of overlapping stop intervals.

    Each element is a tuple (station_name, start_time, end_time, (bus_id1, bus_id2)).
    """
    station_events = defaultdict(list)
    for bus in buses:
        for station, start, end in getattr(bus, 'stop_records', []):
            station_events[station].append((start, end, bus.bus_id))

    overlaps = []
    for station, events in station_events.items():
        events.sort(key=lambda x: x[0])
        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                start = max(events[i][0], events[j][0])
                end = min(events[i][1], events[j][1])
                if start < end:
                    overlaps.append((station, start, end, (events[i][2], events[j][2])))
    return overlaps


def plot_overlaps(overlaps):
    """Plot a space-time diagram for overlapping stops."""
    if not overlaps:
        print("No simultaneous stops found")
        return
    stations = sorted({o[0] for o in overlaps})
    mapping = {s: i for i, s in enumerate(stations)}

    plt.figure()
    for station, start, end, buses in overlaps:
        y = mapping[station]
        plt.plot([start, end], [y, y], label=f"{station} {buses[0]}&{buses[1]}")

    plt.yticks(list(mapping.values()), stations)
    plt.xlabel("time")
    plt.ylabel("station")
    plt.title("Simultaneous Bus Stops")
    plt.legend()
    plt.tight_layout()
    plt.show()
