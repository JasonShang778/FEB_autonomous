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


#PID settings for steering rate
steering_kp = 2.5
steering_ki = 0.05
steering_kd = 0.03

steering_previous_error = None
steering_integral = 0.0

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

    car_position = np.array([xpos, ypos])
    steering_pv = theta

    distances = np.linalg.norm(
        track_centerline_points - car_position,
        axis=1
    )
    nearest_index = np.argmin(distances)

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


    #Determine if we are in a corner or a straight section of the track
    corner_preview_steps = max(1, round(corner_preview_distance / track_spacing))

    middle_preview_steps = max(1, round((corner_preview_distance / 2) / track_spacing))

    first_point = track_centerline_points[nearest_index]
    second_point = track_centerline_points[(nearest_index + middle_preview_steps)%track_sample_count]
    third_point = track_centerline_points[(nearest_index + corner_preview_steps)%track_sample_count]

    first_path_vector = (second_point - first_point)

    second_path_vector = (third_point - second_point)

    first_path_heading = np.arctan2(first_path_vector[1],first_path_vector[0])

    second_path_heading = np.arctan2(second_path_vector[1],second_path_vector[0])

    upcoming_turn = abs(
        wrap_angle(second_path_heading - first_path_heading)
    )

    straight_ahead = (
        upcoming_turn < straight_threshold
    )

    if straight_ahead:
        a = 4.0
    elif v > corner_entry_speed:
        a = -6.6
    else:
        a = 1.0

    return np.array([
        a,
        theta_dot
    ])




def calculate_result(
    simulator,
    reference_path
):
    timestamps, states, controls, crash, slip = (
        simulator.get_results()
    )

    positions = states[:2].T
    number_of_points = len(reference_path)

    starting_distances = np.linalg.norm(
        reference_path - positions[0],
        axis=1
    )

    previous_index = np.argmin(
        starting_distances
    )

    total_progress = 0
    lap_time = None

    for time_index in range(1,len(timestamps)):
        car_position = positions[time_index]

        distances = np.linalg.norm(
            reference_path - car_position,
            axis=1
        )

        current_index = np.argmin(
            distances
        )

        index_change = (
            current_index
            - previous_index
            + number_of_points // 2
        ) % number_of_points - number_of_points // 2

        total_progress += index_change
        previous_index = current_index

        if abs(total_progress) >= number_of_points:
            lap_time = (
                timestamps[time_index]
                - timestamps[0]
            )
            break

    print(f"Lap time: {lap_time:.2f}")

    print("Crash:", np.any(crash))
    print("Slip:", np.any(slip))

    return lap_time



sim.set_controller(controller)
sim.run()
sim.animate()
sim.plot()
lap_time = calculate_result(
    sim,
    track_centerline_points
)
print(f"Lap time: {lap_time:.2f} seconds")
