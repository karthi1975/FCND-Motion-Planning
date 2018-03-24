import argparse
import time
import msgpack
from enum import Enum, auto
import utm
import numpy as np

from planning_utils import a_star, heuristic, create_grid,a_star_graph,create_grid_and_edges
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local
import networkx as nx
import pkg_resources
pkg_resources.require("networkx==2.1")
import numpy.linalg as LA
from bresenham import bresenham
import sys




class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 3.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 5.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)


    def closest_point(self,graph, current_point):
        """
        Compute the closest point in the `graph`
        to the `current_point`.
        """
        closest_point = None
        dist = 100000
        for p in graph.nodes:
            d = LA.norm(np.array(p) - np.array(current_point))
            if d < dist:
                closest_point = p
                dist = d
        return closest_point

    def point(self,p):
        return np.array([p[0], p[1], 1.]).reshape(1, -1)

    def collinearity_check(self, p1, p2, p3, epsilon=0.9e-1):
        m = np.concatenate((p1, p2, p3), 0)
        det = np.linalg.det(m)
        return abs(det) < epsilon

    # We're using collinearity here, but you could use Bresenham as well!
    def prune_path(self, path):
        pruned_path = [p for p in path]
        # TODO: prune the path!

        i = 0
        while i < len(pruned_path) - 2:
            #print("current path********* ",pruned_path[i])
            p1 = self.point(pruned_path[i])
            p2 = self.point(pruned_path[i + 1])
            p3 = self.point(pruned_path[i + 2])


            if self.collinearity_check(p1, p2, p3):

                #print("pruned_path ", pruned_path[i+1])

                pruned_path.remove(pruned_path[i + 1])
            else:
                i += 1
        return pruned_path

    # This is helper functiom to get arbitrary goal location by taking the lat,lon and northing and easting offset
    def global_to_local(self,lat,lon,north_offset, east_offset):

        # self.set_home_position(np.float64(37.795345), np.float64(-122.398013), 0)
        (east_home, north_home, _, _) = utm.from_latlon(self.global_home[1], self.global_home[0])

        # (east, north, _, _) = utm.from_latlon(np.float64(37.795620), np.float64(-122.401727))

        (east, north, _, _) = utm.from_latlon(np.float64(lat), np.float64(lon))

        grid_position = np.array([north - north_home, east - east_home, -0])

        grid_goal = (-north_offset + int(grid_position[0]), -east_offset + int(grid_position[1]))

        return grid_goal


    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")
        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 10

        self.target_position[2] = TARGET_ALTITUDE

        # TODO: read lat0, lon0 from colliders into floating point values
        first_line = next(open("colliders.csv"))
        gps = first_line.split(',')
        lat0 = np.float64(gps[0].lstrip().split(' ')[1])
        lon0 = np.float64(gps[1].lstrip().split(' ')[1])
        #print("The Lat and Long = ", lat0, lon0)

        # TODO: set home position to (lat0, lon0, 0)
        self.set_home_position(lon0, lat0, 0)

        # TODO: retrieve current global position
        # self.global_position = self.global_position
        #print("global_home ", self.global_home)


        # TODO: convert to current local position using global_to_local()
        current_local_pos = global_to_local(self.global_position, self.global_home)

        print('global home {0}, position {1}, local position {2}'.format(self.global_home, self.global_position,
                                                                         self.local_position))
        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)
        
        # Define a grid for a particular altitude and safety margin around obstacles
        grid, north_offset, east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))


        # Define starting point on the grid (this is just grid center)
        #grid_start = (-north_offset, -east_offset)

        #(37.795345, -122.398013)? Mine is (633, 393).

        # TODO: convert start position to current position rather than map center
        #Setting the currposstion relative to the north_offset and  eastoffset  by adding to get start postion in the Grid
        grid_start = (-north_offset + int(current_local_pos[0]), -east_offset + int(current_local_pos[1]))
        #grid_start = (-north_offset-100, -east_offset -100)


        #start_ne = (203, 699)
        #goal_ne = (690, 300)



        # This section is for Grid Based Implementation

        # Set goal as some arbitrary position on the grid
        #grid_goal = (-north_offset + 10, -east_offset + 10)


        # TODO: adapt to set goal as latitude / longitude position and convert

        grid_goal = self.global_to_local(sys.argv[2], sys.argv[4], north_offset, east_offset)

        # Run A* to find a path from start to goal
        # TODO: add diagonal motions with a cost of sqrt(2) to your A* implementation
        # or move to a different search space such as a graph (not done here)
        print('Local Start and Goal: ', grid_start, grid_goal)
        path, _ = a_star(grid, heuristic, grid_start, grid_goal)

        print("path ", len(path))
        # TODO: prune path to minimize number of waypoints
        # TODO (if you're feeling ambitious): Try a different approach altogether!
        pruned_path = self.prune_path(path)

        print("pruned_path = ",pruned_path)
        # Convert path to waypoints
        waypoints = [[p[0] + north_offset, p[1] + east_offset, TARGET_ALTITUDE, 0] for p in pruned_path]






        """ 
       
    
        # Graph Implementation
        # This is now the routine using Voronoi
        grid, edges = create_grid_and_edges(data, TARGET_ALTITUDE, SAFETY_DISTANCE)

        G = nx.Graph()
        for e in edges:
            p1 = e[0]
            p2 = e[1]
            dist = LA.norm(np.array(p2) - np.array(p1))
            G.add_edge(p1, p2, weight=dist)

        #start_ne = (815, 639)

        # Map to Grid location without any obstacles



        print('Local Start and Goal: ', grid_start, grid_goal)

        start_ne_g = self.closest_point(G, grid_start)
        goal_ne_g = self.closest_point(G, grid_goal)

        #print(start_ne_g,goal_ne_g)

        path1, cost = a_star_graph(G, heuristic, start_ne_g, goal_ne_g)
        print("path1 ",len(path1))
        pruned_path = self.prune_path(path1)
        waypoints = [[int(p[0] + north_offset), int(p[1] + east_offset), TARGET_ALTITUDE, 0] for p in pruned_path]
        
         """



        # Set self.waypoints
        self.waypoints = waypoints
        # TODO: send waypoints to sim
        self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-lat", "--latitude", help="enter latitude ", action="store_true")
    parser.add_argument("-lon", "--longitude", help="enter longitude ", action="store_true")
    #parser.add_argument('--port', type=int, default=5760, help='Port number')
    #parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    #args = parser.parse_args()

    #conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)

    conn = MavlinkConnection('tcp:127.0.0.1:5760', timeout=60)
    drone = MotionPlanning(conn)
    time.sleep(1)

    drone.start()
