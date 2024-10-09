'''
PID_controller.py
Authors: Andrew Gaylord, Patrick Schwartz

PID controller for cubesat attitude control using quaternion error kinematics. 
the PID controller computes the PWM signals to send to our reaction wheels to achieve our target orientation.
Takes in a target quaternion, gains parameters, and current state

'''


import numpy as np
from UKF_algorithm import normalize

MAX_PWM = 65535 # pwm val that gives max speed according to Tim


class PIDController:
    def __init__(self, kp, ki, kd, dt):
        '''
        PID controller class with gains for proportional, integral, and derivative control.

        @params:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            dt: Time step (sampling time)
        '''
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.integral_error = np.zeros(3)  # Initialize integral error as a zero vector (3 x 1)

    def pid_controller(self, state, target, omega, old_pwm):
        '''
        PID controller to compute PWM signals for reaction wheels.

        @params:
            state: Current quaternion of cubesat (4 x 1) [q0 ; q1:3]
            target: Target quaternion of cubesat (4 x 1) [q0 ; q1:3]
            omega: Current angular velocity of cubesat (3 x 1)
            old_pwm: Previous PWM values for each motor (4 x 1)

        @returns:
            pwm: PWM signals for each motor (4 x 1)
        '''

        # Find the error quaternion (delta_q_out) between current and target quaternion
        delta_q_out = delta_q(state, target)  # outputs 4x1, where first element is the scalar part (q0)

        # Quaternion error (vector part only) and its proportional control component
        proportional = -self.kp * np.sign(delta_q_out[0]) * delta_q_out[1:4]

        # Update the integral error (accumulate error over time)
        self.integral_error += delta_q_out[1:4] * self.dt

        # Integral control component
        integral = self.ki * self.integral_error

        # Derivative control component based on angular velocity (omega)
        # derivative term responds to how fast the error quaternion is changing over time (which is related to how fast we're spinning)
        # this allows us to anticipate and dampen rapid changes, preventing overshooting
        derivative = -self.kd * omega

        # Total control output (torque command)
        L = proportional + integral + derivative

        # Reaction wheel transformation matrix for the NASA configuration
        alpha = 1 / np.sqrt(3)
        beta = 1 / np.sqrt(3)
        gamma = 1 / np.sqrt(3)

        # Transformation matrix that maps 3-axis torque to the 4 reaction wheels
        W_inv = np.array([[(1 + beta**2 + gamma**2), -alpha * beta, -alpha * gamma],
                          [-alpha * beta, (1 + alpha**2 + gamma**2), -beta * gamma],
                          [alpha * gamma, -beta * gamma, (1 + alpha**2 + beta**2)],
                          [alpha, beta, gamma]]) / (1 + alpha**2 + beta**2 + gamma**2)

        # Convert torque (L) to reaction wheel space (4 motors)
        motor_torques = np.matmul(W_inv, L)

        # PWM calculation: Map torque to PWM values
        max_torque = 0.01  # Define the maximum torque your reaction wheels can handle (example: 0.01 Nm)

        # Map the torque output to PWM range
        pwm = (motor_torques / max_torque) * MAX_PWM

        # Update PWM using finite difference: pwm = old_pwm + torque * dt
        pwm = np.add(pwm * self.dt, old_pwm)

        # Convert to integer values for actual PWM signals
        pwm = np.array([int(p) for p in pwm])

        # Ensure PWM is within bounds (0 to MAX_PWM)
        pwm = np.clip(pwm, -MAX_PWM, MAX_PWM)

        return pwm

def delta_q(q_actual, q_target):
    '''
    Calculate the quaternion error (q_error) between the actual and target quaternions.
    q_error = q_target * inv(q_actual)
    
    @params
        q_actual: Current quaternion (4x1)
        q_target: Target quaternion (4x1)

    @returns:
        q_error: Quaternion error (4x1)
    '''
    # Quaternion multiplication: q_error = q_target * inv(q_actual)
    q_actual_inv = np.array([q_actual[0], -q_actual[1], -q_actual[2], -q_actual[3]])  # Inverse of the quaternion
    q_error = quat_mult(q_target, q_actual_inv)
    return q_error

def quat_mult(q1, q2):
    '''
    Quaternion multiplication of two quaternions q1 and q2.
    
    @params
        q1, q2: Quaternions (4x1)

    @returns:
        q3: Resulting quaternion from q1 * q2
    '''
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w3 = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x3 = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y3 = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z3 = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w3, x3, y3, z3])

# Example usage
if __name__ == "__main__":
    # Initialize PID controller
    dt = 0.01  # Time step of 10ms
    kp = 0.5   # Proportional gain
    ki = 0.1   # Integral gain
    kd = 0.05  # Derivative gain
    pid = PIDController(kp, ki, kd, dt)

    # Example state (current quaternion), target quaternion, and angular velocity
    q_actual = np.array([0.707, 0.707, 0, 0])  # Example current quaternion
    q_target = np.array([1, 0, 0, 0])          # Target quaternion (identity quaternion)
    omega = np.array([0.01, 0.02, 0.03])        # Example current angular velocity
    old_pwm = np.array([120, 120, 120, 120])    # Example old PWM values

    # Compute new PWM signals
    pwm_output = pid.pid_controller(q_actual, q_target, omega, old_pwm)

    # Output the computed PWM signals
    print(f"Computed PWM signals: {pwm_output}")







#=======================================================================================================




import numpy as np
from scipy.spatial.transform import Rotation as R
from UKF_algorithm import normalize

class PIDController:
    def __init__(self, kp, ki, kd, dt):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        
        self.integral_error = np.zeros(3)
        self.previous_error = np.zeros(3)

    def compute_control(self, error):
        # Proportional term
        proportional = self.kp * error
        
        # Integral term
        self.integral_error += error * self.dt
        integral = self.ki * self.integral_error
        
        # Derivative term
        derivative = self.kd * (error - self.previous_error) / self.dt
        self.previous_error = error
        
        # PID control output (torque command)
        return proportional + integral + derivative

class ReactionWheelController:
    def __init__(self, kp, ki, kd, dt):
        # PID controller for each axis (x, y, z)
        self.pid = PIDController(kp, ki, kd, dt)

    def quaternion_error(self, q_target, q_actual):
        # Calculate the relative quaternion (q_error = q_target * q_actual_inverse)
        q_actual_inv = R.from_quat(q_actual).inv()
        q_error = R.from_quat(q_target) * q_actual_inv
        
        # Convert quaternion error to axis-angle representation to get the rotational error
        axis_angle_error = q_error.as_rotvec()  # Rotation vector (3D error)
        return axis_angle_error

    def compute_pwm(self, q_target, q_actual):
        # Step 1: Compute the quaternion error
        error = self.quaternion_error(q_target, q_actual)
        
        # Step 2: Compute the control output (torque) using PID controller
        control_output = self.pid.compute_control(error)
        
        # Step 3: Map the control output (torque) to PWM signals for the reaction wheels
        pwm_signal = self.torque_to_pwm(control_output)
        return pwm_signal

    def torque_to_pwm(self, torque):
        # Map the torque output from PID to PWM range (assumed between 0 and 255)

        magic = 1
        pwms = torque * magic

        # transformation matrix for NASA configuration
        alpha = 1/np.sqrt(3)
        beta = 1/np.sqrt(3)
        gamma = 1/np.sqrt(3)

        # If fourth reaction wheel is not mounted exactly how we want we can adjust alpha, beta, gamma
        W =  [[1, 0, 0, alpha],
                [0, 1, 0, beta],
                [0, 0, 1, gamma]]
        
        # normalizing scalar
        n = (1 + alpha**2 + beta**2 + gamma**2)

        # Calculates pseudoinverse needed for transformation (pg 157 of Fundamentals book)
        W_inv = np.array([[(1 + beta**2 + gamma**2), -alpha*beta, -alpha*gamma],
                [-alpha*beta, (1 + alpha**2 + gamma**2), -beta*gamma],
                [alpha*gamma, -beta*gamma, (1 + beta**2 + beta**2)],
                [alpha, beta, gamma]])/n
        
        # convert output for 3 rx wheels to 4
        pwms = np.matmul(W_inv,pwms)

        return pwms

# Example usage
if __name__ == "__main__":
    # Time step
    dt = 0.01  # 10ms time step

    # PID gains (for x, y, z axes)
    kp = np.array([0.5, 0.5, 0.5])  # Proportional gain
    ki = np.array([0.1, 0.1, 0.1])  # Integral gain
    kd = np.array([0.05, 0.05, 0.05])  # Derivative gain

    # Initialize Reaction Wheel Controller
    rw_controller = ReactionWheelController(kp, ki, kd, dt)

    # Target and actual quaternions (example values)
    q_target = normalize([1.0, 0.0, -1.0, 0.0])  # Target quaternion (identity quaternion)
    q_actual = normalize([1.0, 0.0, 0.0, 0.0])  # Example actual quaternion (small rotation)

    # Compute the PWM signals
    pwm_signals = rw_controller.compute_pwm(q_target, q_actual)

    print(f"PWM signals for reaction wheels: {pwm_signals}")
