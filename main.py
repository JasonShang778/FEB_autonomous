import numpy as np
from simulator import Simulator, centerline

sim = Simulator()

#some basic parameters for the controller
track_length = 104.758  
track_sample_count = 1000
setpoint_distance = 2
wheelbase = 1.58
dt = 0.1

track_spacing = track_length / track_sample_count
track_centerline_distance = np.arange(0.0, track_length, track_spacing)
track_centerline_points = centerline(track_centerline_distance)

straight_speed_setpoints = 3.0



def controller(x):
    """controller for a car

    Args:
        x (ndarray): numpy array of shape (5,) containing [x, y, heading, velocity, steering angle]

    Returns:
        ndarray: numpy array of shape (2,) containing [fwd acceleration, steering rate]
    """
    xpos   = x[0]                   # current x position
    ypos   = x[1]                   # current y position
    phi    = np.mod(x[2], 2*np.pi)  # current heading (radians)
    v      = x[3]                   # current velocity
    theta   = x[4]                  # current steering angle




sim.set_controller(controller)
sim.run()

lap_time = calculate_lap_time()
print(f"First lap time: {lap_time:.2f} seconds")

sim.animate()
sim.plot()
print(sim.get_results)