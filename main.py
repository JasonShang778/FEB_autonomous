import numpy as np
from simulator import Simulator, centerline

sim = Simulator()

#some basic parameters for the controller
track_length = 104.758  
track_sample_count = 1000
setpoint_distance = 2
wheelbase = 1.58
dt = 0.01



track_spacing = track_length / track_sample_count
track_centerline_distance = np.arange(0.0, track_length, track_spacing)
track_centerline_points = centerline(track_centerline_distance)



lookahead_distance = 2.5
corner_preview_distance = 8.0
straight_threshold = np.deg2rad(8.0)
corner_entry_speed = 3.5


#PID constants, kp, ki, kd, for steering rate
steering_kp = 3
steering_ki = 0.01
steering_kd = 0.1

steering_previous_error = None
steering_integral = 0.0


#PID constants, kp, ki, kd, for velocity 
velocity_kp = 1.5
velocity_ki = 0.1
velocity_kd = 0.1


velocity_previous_error = None
velocity_integral = 0.0
change_of_v=0.0


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def pid_controller(setpoint, pv, kp, ki, kd, dt, previous_error=None, integral=0.0):
    error = setpoint - pv

    integral += error * dt

    derivative = (error - previous_error) / dt

    control = (kp * error + ki * integral + kd * derivative)

    return control, error, integral


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

    global steering_previous_error, steering_integral
    global velocity_previous_error, velocity_integral, change_of_v

    car_position = np.array([xpos, ypos])
    steering_pv = theta

    distances = np.linalg.norm(
        track_centerline_points - car_position,
        axis=1
    )
    nearest_index = np.argmin(distances)


    #PID for steering rate
    steering_lookahead_steps = max(1 ,round(lookahead_distance / track_spacing))

    steering_target_index = (nearest_index+ steering_lookahead_steps)%track_sample_count
    steering_target = track_centerline_points[steering_target_index]
    target_vector = steering_target - car_position
    target_heading = np.arctan2(target_vector[1], target_vector[0])
    heading_error = wrap_angle(target_heading - phi)
    actual_lookahead_distance = max(1e-6, np.linalg.norm(target_vector))
    steering_setpoint = np.arctan2(2.0*wheelbase*np.sin(heading_error), actual_lookahead_distance)
    steering_setpoint = np.clip(steering_setpoint, -0.7, 0.7)


    if steering_previous_error is None:
        steering_previous_error = (steering_setpoint - steering_pv)

    (change_of_theta, steering_previous_error, steering_integral) = pid_controller(
        setpoint=steering_setpoint,
        pv=steering_pv,
        kp=steering_kp,
        ki=steering_ki,
        kd=steering_kd,
        previous_error=steering_previous_error,
        integral=steering_integral,
        dt=dt
    )

    theta_dot = np.clip(change_of_theta, -1.0, 1.0)


    #PID for velocity
    
    #setpoint for velocity is based on theta
    maximum_velocity = 16.0

    if (theta == 0.0):
        theta = 1e-6

    velocity_pv = v
    velocity_setpoint = np.sqrt(0.79 * np.sqrt(10.0**2 - change_of_v**2) / abs(np.sin(np.arctan(0.5 * np.tan(theta)))))
    velocity_setpoint = np.clip(velocity_setpoint, 0.0, 6.0)
    velocity_previous_error = (velocity_setpoint - velocity_pv)
    (change_of_v, velocity_previous_error, velocity_integral) = pid_controller(
        setpoint=velocity_setpoint,
        pv=velocity_pv,
        kp=velocity_kp,
        ki=velocity_ki,
        kd=velocity_kd,
        previous_error=velocity_previous_error,
        integral=velocity_integral,
        dt=dt
    )
    change_of_v = np.clip(change_of_v, -10, 4)

    

    return np.array([
        change_of_v,
        theta_dot
    ])





sim.set_controller(controller)
sim.run()
sim.animate()
sim.plot()

