import numpy as np
from simulator import Simulator, centerline

sim = Simulator()

#some basic parameters for the controller
track_length = 104 
setpoint_distance = 2
dt = 0.01

track_spacing = track_length / 1000
track_centerline_points = centerline(np.arange(0.0, track_length, track_spacing))

theta_kp = 3
theta_ki = 0.01
theta_kd = 0.1

theta_previous_error = None
theta_integral = 0.0

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

    global theta_previous_error, theta_integral, velocity_previous_error, velocity_integral, change_of_v

    car_position = np.array([xpos, ypos])
    pv = theta

    nearest_index = np.argmin(np.linalg.norm(track_centerline_points - car_position, axis=1))


    #PID theta

    pure_pursuit_point = track_centerline_points(nearest_index+ np.round(2.5 / track_spacing))%1000
    vector = pure_pursuit_point - car_position


    setpoint_angle = wrap_angle(np.arctan2(vector[1], vector[0]) - phi)
    setpoint = np.clip(setpoint_angle, -0.7, 0.7)
    
    if steering_previous_error is None:
        steering_previous_error = setpoint - pv

    (change_of_theta, steering_previous_error, steering_integral) = pid_controller(
        setpoint=setpoint,
        pv=pv,
        kp=theta_kp,
        ki=theta_ki,
        kd=theta_kd,
        previous_error=steering_previous_error,
        integral=steering_integral,
        dt=dt
    )

    theta_dot = np.clip(change_of_theta, -1.0, 1.0)


    
    #below is for velocity!!!!
    # 
    vpv = v
    maxv = 10.0

    larger_angle = max(abs(theta), abs(setpoint))
    lateral_acc_limit_index = abs(np.sin(np.arctan(0.5 * np.tan(larger_angle))))

    vsetpoint = np.clip(np.sqrt(0.79 * 10.0/lateral_acc_limit_index), 0.0, maxv)

    threshold_turn = np.deg2rad(70.0)
    curve_speed = 3.5
    planned_braking = 6.0

    d_to_slow_down= (v**2 - curve_speed**2) / (2.0 * 6)
    index_array = nearest_index + np.arange((10 + d_to_slow_down) / track_spacing)
    delta_vec = np.diff(track_centerline_points[index_array], axis=0)
    deltaerror = np.arctan2(delta_vec[:, 1], delta_vec[:, 0])
    max_angle_turn = np.max(np.abs(deltaerror - deltaerror[0]))

    if max_angle_turn >= threshold_turn:
        vsetpoint = min(vsetpoint, curve_speed)

    if velocity_previous_error is None:
        velocity_previous_error = vsetpoint - vpv

    change_of_v, velocity_previous_error, velocity_integral = pid_controller(
        setpoint=vsetpoint,
        pv=vpv,
        kp=velocity_kp,
        ki=velocity_ki,
        kd=velocity_kd,
        previous_error=velocity_previous_error,
        integral=velocity_integral,
        dt=dt
    )

    lateral_accel = v**2 * abs(np.sin(np.arctan(0.5 * np.tan(theta)))) / 0.79
    accel_possible = np.sqrt(max(0.0, 11.9**2 - lateral_accel**2))
    change_of_v = np.clip(change_of_v, -min(10.0, accel_possible), min(4.0, accel_possible))

    
    return np.array([
        change_of_v,
        theta_dot
    ])





sim.set_controller(controller)
sim.run()
sim.animate()
sim.plot()

