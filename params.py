'''
params.py

Reference file that contains all important variables for our system
Here, all factors that we would want to change are easily accessible
'''

import numpy as np


QUAT_INITIAL = np.array([1.0, 0.0, 0.0, 0.0])
RW_INITIAL = np.array([0.0, 0.0, 0.0, 0.0])

# Pulse Width Modulation (PWM) signal that generates the max speed in our motors
MAX_PWM = 65535

# =======  OPTIONS  ======================================

IDEAL_KNOWN = True
DATA_FILE = "data.txt"

# should be true if not doing controls
RUN_STATISTICAL_TESTS = False

# 0 = create pdf report, 1 = show 3D animation visualization, 2 = both, 3 = none
RESULT = 2


OUTPUT_FILE = "output.pdf"

# =======  CONTROLS  ===============================
# target orientation for if we're simulating controls
TARGET = np.array([1.0, 0.0, 0.5, 0.0])

# gains for our PID controller
KP = MAX_PWM * 4.0e-8       # Proportional gain
KI = MAX_PWM * 1e-9         # Integral gain
KD = MAX_PWM * 9e-9         # Derivative gain

TF = 10
DT = .02

# =======  UKF  ====================================
COVARIANCE_INITIAL_MAG = 5e-7

# TODO: notes from Estimation II
# filter process noise Q
PROCESS_NOISE_MAG = 0.00001
PROCESS_NOISE_K = 10

# filter measurement noise R
MEASUREMENT_MAGNETOMETER_NOISE = 0.001
MEASUREMENT_GYROSCOPE_NOISE = 0.01

# =======  SENSORS  ==================================
# noise sd = noise density * sqrt(sampling rate)
# vn100 imu sampling rate from user manual = 200 Hz

# mag noise density from vn100 website = 140 uGauss /sqrt(Hz)
SENSOR_MAGNETOMETER_SD = (140 * 10e-6) * np.sqrt(200)

# gyro noise density from vn100 website = 0.0035 /s /sqrt(Hz)
SENSOR_GYROSCOPE_SD = 0.0035 * np.sqrt(200)

# =======  PHYSICS  ===================================
# motor model - Maxon DCX 8 M (9 volts)
# TODO: rename everything to english lol

# Store principal moment of inertia for reaction wheels about spin axis and about axis transverse to spin axis respectively
# TODO: what is this and should it be 1e-7
# SPIN_AXIS_INERTIA = 5.1e-7
SPIN_AXIS_INERTIA = 1e-7
TRANSVERSE_AXIS_INERTIA = 0.0

# store moment of inertia tensor w/o reaction wheel inertias
CUBESAT_BODY_INERTIA = (1e-7) * np.array([[46535.388, 257.834, 536.12],
                                          [257.834, 47934.771, -710.058],
                                          [536.12, -710.058, 23138.181]])

# Moments of Inertia of reaction wheels [g cm^2] - measured
Iw1 = (1/2)*38*1.8**2 # I_disc = 1/2 * M * R^2
Iw2 = Iw1
Iw3 = Iw1
Iw4 = Iw1

# Moment of inertia tensor of rxn wheels [kg m^2]
RW_CONFIG_INERTIA = np.array([[Iw1, 0, 0, 0],
                              [0, Iw2, 0, 0],
                              [0, 0, Iw3, 0],
                              [0, 0, 0, Iw4]])

# Transformation matrix for NASA config given in Fundamentals pg 153-154
TRANSFORMATION = np.array([[1, 0, 0, 1/np.sqrt(3)],
                           [0, 1, 0, 1/np.sqrt(3)],
                           [0, 0, 1, 1/np.sqrt(3)]])


class params:
    MAX_CURRENT = 1
    MIN_CURRENT = -MAX_CURRENT
    thermal_resistance = 0.01  # °C per A^2 (or Kelvin per A^2). how much the current flowing through the system causes heat generation
    cooling_constant = 0.1     # 1/s (rate of cooling). how quickly the temperature difference between the system and its surroundings dissipates
    wheel_coupling_factor = 0.5  # coupling between ambient and reaction wheel temperature
    # Importing motor parameters - Maxon DCX 8 M (9 volts)
    Rwa = 3.54      # Ohms, winding resistance at ambient temperature
    Lw = 0.424e-3  # Henry
    Kt = 8.82e-3   # Torque constant Nm/A
    Kv = Kt    # Voltage constant V*s/rad
    Jm = 5.1*(1e-7)   # Kg m^2
    bm = 3.61e-6   # [N·m·s/rad] Viscous friction
    Rha = 16.5      # K/W
    Rwh = 2.66     # K/W
    Cwa = 2.31/Rwh     # Thermal Capacitance
    Cha = 162/Rha      # Thermal Capacitance
    alpha_Cu = 0.00393 # copper's temperature coefficient [1/K]
    # Moments of Inertia [g cm^2]- from CAD of CubeSat test bed
    Ixx = 46535.388 
    Ixy = 257.834 
    Ixz = 536.12
    Iyx = 257.834 
    Iyy = 47934.771 
    Iyz = -710.058
    Izx = 546.12 
    Izy = -710.058 
    Izz = 23138.181

    # Moment of Inertia Tensor of full 2U cubesat [kg m^2]
    J_B = (1e-7)*np.array([[Ixx, Ixy, Ixz],
                    [Iyx, Iyy, Iyz],
                    [Izx, Izy, Izz]])
    
    J_B_inv = np.linalg.inv(J_B)

    # Moments of Inertia of rxn wheels [g cm^2] - measured
    Iw1 = (1/2)*38*1.8**2 # I_disc = 1/2 * M * R^2
    Iw2 = Iw1 
    Iw3 = Iw1 
    Iw4 = Iw1

    # Moment of inertia tensor of rxn wheels [kg m^2]
    J_w = (1e-7)*np.array([[Iw1, 0, 0, 0],
                    [0, Iw2, 0, 0],
                    [0, 0, Iw3, 0],
                    [0, 0, 0, Iw4]])

    # External torques (later this can be from magnetorquers)
    L_B = np.array([0, 0, 0])

    # Transformation matrix for NASA config given in Fundamentals pg
    # 153-154
    W = np.array([[1, 0, 0, 1/np.sqrt(3)],
            [0, 1, 0, 1/np.sqrt(3)],
            [0, 0, 1, 1/np.sqrt(3)]])