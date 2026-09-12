import numpy as np
from simulator import Simulator, centerline

sim = Simulator()

#some basic parameters for the controller

track_spacing = 104/ 1000
center_arrays = centerline(np.arange(0.0, 104, track_spacing))
a = 0

# https://www.digikey.com/en/maker/tutorials/2024/implementing-a-pid-controller-algorithm-in-python
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





    pv = theta
    dt = 0.01

    nearest_index = np.argmin(np.linalg.norm(center_arrays - np.array([xpos, ypos]), axis=1))


    #PID theta
    #pure pursuit

    pt_ahead = center_arrays[nearest_index + np.round(2.5 / track_spacing).astype(int)]  
    deltavector = pt_ahead - np.array([xpos, ypos])


    setpoint_angle = np.arctan2(deltavector[1], (deltavector[0] - phi))
    setpoint = np.clip(setpoint_angle, -0.7, 0.7)

    global a
    if a == 0:
        steering_previous_error = setpoint - pv
        steering_integral = 0.0

        
        

    (change_of_theta, steering_previous_error, steering_integral) = pid_controller(
        setpoint=setpoint,
        pv=pv,
        kp=3,
        ki=0.01,
        kd=0.1,
        previous_error=steering_previous_error,
        integral=steering_integral,
        dt=dt
    )

    theta_dot = np.clip(change_of_theta, -1.0, 1.0)


    
    #below is for velocity!!!!
    # 
    vpv = v
    maxv = 10.0

    larger_angle = max(abs(theta), abs(setpoint))       #

    lateral_acc_limit_index = abs(np.sin(np.arctan(0.5 * np.tan(larger_angle))))
    vsetpoint = np.clip(np.sqrt(0.79 * 10.0/lateral_acc_limit_index), 0.0, maxv)

    d_to_slow_down= (v**2 - 3.5**2) / (2.0 * 6)
    index_array = nearest_index + np.arange((10 + d_to_slow_down) / track_spacing)        #speed to enter curve: 3.5

    delta_vec = np.diff(center_arrays[index_array], axis=0)
    deltaerror = np.arctan2(delta_vec[:, 1], delta_vec[:, 0])
    max_angle_turn = np.max(np.abs(deltaerror - deltaerror[0]))

    if max_angle_turn >= 1.2:
        vsetpoint = min(vsetpoint, 3.5)
    if a == 0:
        velocity_previous_error = vsetpoint - vpv
        velocity_integral = 0.0
        a+=1
    change_of_v, velocity_previous_error, velocity_integral = pid_controller(
        setpoint=vsetpoint,
        pv=vpv,
        kp=1.5,
        ki=0.1,
        kd=0.1,
        previous_error=velocity_previous_error,
        integral=velocity_integral,
        dt=dt
    )

    lateral_a = v**2 * abs(np.sin(np.arctan(0.5 * np.tan(theta)))) / 0.79
    a_max = np.sqrt(max(0.0, 11.9**2 - lateral_a**2))
    change_of_v = np.clip(change_of_v, -min(10.0, a_max), min(4.0, a_max))

    
    return np.array([
        change_of_v,
        theta_dot
    ])





sim.set_controller(controller)
sim.run()
sim.animate()
sim.plot()

