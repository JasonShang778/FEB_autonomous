import numpy as np
from simulator import Simulator, centerline

sim = Simulator()



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

     # The simulator calls this function every 0.01 seconds.
    dt = 0.01

    # Physical and path settings
    wheelbase = 1.58
    track_length = 104.758
    sample_count = 1200
    lookahead_distance = 2.5

        # Speed PID gains
    speed_kp = 1.5
    speed_ki = 0.15
    speed_kd = 0.02

    # Steering PID gains
    steer_kp = 2.5
    steer_ki = 0.05
    steer_kd = 0.03

    if not hasattr(controller, "track_points"):
        track_distances = np.linspace(
            0.0,
            track_length,
            sample_count,
            endpoint=False
        )

        controller.track_points = centerline(track_distances)

        # PID memory
        controller.speed_integral = 0.0
        controller.steer_integral = 0.0

        controller.previous_speed_error = None
        controller.previous_steer_error = None

        controller.speed_derivative = 0.0
        controller.steer_derivative = 0.0

    car_position = np.array([xpos, ypos])
    
    # Find the nearest point on the centerline.
    distances = np.linalg.norm(
        controller.track_points - car_position,
        axis=1
    )

    nearest_index = np.argmin(distances)

    # Choose a point ahead of the nearest centerline point.
    track_spacing = track_length / sample_count

    lookahead_steps = max(
        1,
        round(lookahead_distance / track_spacing)
    )

    target_index = (
        nearest_index + lookahead_steps
    ) % sample_count

    target = controller.track_points[target_index]

    # Calculate the target direction.
    target_vector = target - car_position

    target_heading = np.arctan2(
        target_vector[1],
        target_vector[0]
    )

    # Keep heading error between -pi and pi.
    heading_error = (
        target_heading - phi + np.pi
    ) % (2.0 * np.pi) - np.pi

    actual_lookahead = max(
        np.linalg.norm(target_vector),
        1e-6
    )

    # Calculate the desired steering angle using pure pursuit.
    desired_theta = np.arctan2(
        2.0 * wheelbase * np.sin(heading_error),
        actual_lookahead
    )

    desired_theta = np.clip(
        desired_theta,
        -0.7,
        0.7
    )

    # -------------------------
    # Steering PID
    # -------------------------

    steering_error = desired_theta - theta

    controller.steer_integral += steering_error * dt

    # Integral anti-windup
    controller.steer_integral = np.clip(
        controller.steer_integral,
        -0.5,
        0.5
    )

    if controller.previous_steer_error is None:
        raw_steering_derivative = 0.0
    else:
        raw_steering_derivative = (
            steering_error
            - controller.previous_steer_error
        ) / dt

    # Smooth the derivative value.
    controller.steer_derivative = (
        0.9 * controller.steer_derivative
        + 0.1 * raw_steering_derivative
    )

    theta_dot = (
        steer_kp * steering_error
        + steer_ki * controller.steer_integral
        + steer_kd * controller.steer_derivative
    )

    controller.previous_steer_error = steering_error

    # Slow down when the required turn is large.
    turn_amount = min(
        abs(heading_error) / (np.pi / 2.0),
        1.0
    )

    desired_v = 3.0 * (1.0 - 0.5 * turn_amount)

    # -------------------------
    # Speed PID
    # -------------------------

    speed_error = desired_v - v

    controller.speed_integral += speed_error * dt

    # Integral anti-windup
    controller.speed_integral = np.clip(
        controller.speed_integral,
        -2.0,
        2.0
    )

    if controller.previous_speed_error is None:
        raw_speed_derivative = 0.0
    else:
        raw_speed_derivative = (
            speed_error
            - controller.previous_speed_error
        ) / dt

    controller.speed_derivative = (
        0.9 * controller.speed_derivative
        + 0.1 * raw_speed_derivative
    )

    a = (
        speed_kp * speed_error
        + speed_ki * controller.speed_integral
        + speed_kd * controller.speed_derivative
    )

    controller.previous_speed_error = speed_error

    # Enforce the control limits.
    a = np.clip(a, -10.0, 4.0)
    theta_dot = np.clip(theta_dot, -1.0, 1.0)

    return np.array([a, theta_dot])

def calculate_lap_time():
    """Return the time needed to complete the first lap."""

    ts, states, controls, crash, slip = sim.get_results()

    # If you used the PID controller from the previous message:
    track_points = controller.track_points

    # If TRACK_POINTS is defined globally, use this instead:
    # track_points = TRACK_POINTS

    number_of_points = len(track_points)

    # states[:2].T gives every recorded [x, y] position.
    positions = states[:2].T

    nearest_indices = []

    for position in positions:
        distances = np.linalg.norm(
            track_points - position,
            axis=1
        )

        nearest_index = np.argmin(distances)
        nearest_indices.append(nearest_index)

    nearest_indices = np.array(nearest_indices)

    # Calculate movement through the centerline indices.
    index_changes = np.diff(nearest_indices)

    # Correct changes when the index wraps from the end of the
    # centerline back to zero.
    index_changes = (
        index_changes + number_of_points // 2
    ) % number_of_points - number_of_points // 2

    progress = np.concatenate([
        [0],
        np.cumsum(index_changes)
    ])

    # One complete lap means advancing through all centerline points.
    completed_lap = np.where(
        progress >= number_of_points
    )[0]

    if len(completed_lap) == 0:
        return None

    finish_index = completed_lap[0]
    return ts[finish_index]


sim.set_controller(controller)
sim.run()

lap_time = calculate_lap_time()
print(f"First lap time: {lap_time:.2f} seconds")

sim.animate()
sim.plot()