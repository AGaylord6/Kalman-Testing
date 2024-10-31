# CubeSat_Model.py
# AME 40652 - Intermediate Controls
# Create model of 2U CubeSat and solve numerically

# Patrick Schwartz and Elena Dansdill

# In collaboration with the University of Notre Dame's IrishSat club https://ndirishsat.com/

# April 30, 2024

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12
import scipy
from scipy.integrate import solve_ivp
import time

import control as ct
import pdb
import sympy

begin_time = time.time()

# Patrick code notes: 
# J = inertia, L = tau (torque), omega = angular velocity
#   i (current, based on voltage), Th_Ta (temp diff between housing and ambient), and Tw_Ta (winding and ambient) are states he's tracking that we don't care about
#   l_w is torque produced by pwm
#   H_B_w is angular momentum of wheel
#   J_w is 2D array of moment of inertias of wheels (= rw_config)
#   omega_B is angular velocity of body, omega_w is angular velocity of wheel
#   Rw is winding resistance (depends on temp)
#   he gets the pwm (u = input) from the solution states... but doesn't actually implement states. just based on time. 
    # tau_e = external torque. last 4 elements of input array
    # omega_w = last wheel speed
# use alpha_rw or omega_w_dot (would have to impliment wheel info) to calc H_B_w_dot/L_w???
# we use i_trans to edit intertia of body??

# Function that defines system of 1st order ODE's
def f(t,x,u,params):
        '''
        Inputs:
        t - time
        x - column vector containing the states (23x1) 
        u - column vector containing the inputs (8x1)
        params - dictionary full of motor dyamics parameters

        Returns:
        x_dot - column vector containing the velocity and acceleration
               vectors of the system at time t
        '''

        #________Define States and Inputs__________ #

        # Quaternion
        # NOTE: Fundamentals book uses q = [q1:3 q4] (3-vector part q1:3 and
        # scalar part q4
        # IrishSat uses q = [q0 q1:3] (scalar part is first instead of last)
        # We will continue with the IrishSat representation 
        q = x[0:4] # [q0 q1:3]

        # Angular velocity of the body with respect to inertial frame
        omega_B = x[4:7] #[x(5) x(6) x(7)]
        
        # Angular velocity of reaction wheels with respect to their
        # respective axis
        omega_w = x[7:11] #[x(8) x(9) x(10) x(11)]
        
        # Motor states
        i = x[11:15] # Current to each motor
        Th_Ta = x[15:19] # diff in temp between housing and ambient
        Tw_Ta = x[19:23] # diff in temp between winding and ambient

        # Inputs
        Vin = (9/65535)*u[0:4] # convert from pwm to voltage
        tau_e = u[4:8]

        Rw = params.Rwa *(1+params.alpha_Cu*Tw_Ta)

        # Define motor dynamics
        omega_w_dot = (params.Kt*i + tau_e - params.bm*omega_w)/params.Jm
        i_dot = (Vin - i*Rw - params.Kv*omega_w)/params.Lw
        Th_Ta_dot = ((Th_Ta - Tw_Ta)/params.Rwh - Th_Ta/params.Rha)/params.Cha
        Tw_Ta_dot = (i**2*Rw - (Th_Ta - Tw_Ta)/params.Rwh)/params.Cwa

        print("old current: ", i)
        # print("old wheel: ", omega_w)
        # print("new wheel: ", omega_w_dot)
        # print("voltage: ", Vin)
        # print("current_dot: ", i_dot)
        # print("Th_Ta: ", Th_Ta_dot)
        # print("Tw_Ta: ", Tw_Ta_dot)
        # print("")
        # print("iteration")

        # Torques of reaction wheels along their axes
        L_w = np.matmul(params.J_w,omega_w_dot)

        # Quaternion product matrix for omega_B as given in Fundmentals pg 39 - Eq 2.96 and 2.97a give quaternion product for vector in R3 
        # like omega is. Except because of our notation change it is [0 x] instead of [x0] as given in book
        # The matrix representation for the quaternion product is given on page 38, eq 2.85
        # This is also given as omega operator at https://ahrs.readthedocs.io/en/latest/filters/angular.html
        q_prod_mat = np.array([[0, -omega_B[0], -omega_B[1], -omega_B[2]],
                      [omega_B[0], 0, omega_B[2], -omega_B[1]],
                      [omega_B[1], -omega_B[2], 0, omega_B[0]],
                      [omega_B[2], omega_B[1], -omega_B[0], 0]])

        # Angular momentum of reaction wheels along their axes
        H_B_w = np.matmul(params.J_w,omega_w)

        #___________Define ODE's_______________# 

        # d/dt(q) as given in Fundamentals pg 84 eq 3.79d
        q_dot = 0.5*np.matmul(q_prod_mat,q)
        
        # d/dt(omega_B) as given in Fundamentals pg 102 eq 3.147
        # This is just Newton's 2nd law: Tau = I*alpha
        omega_B_dot = np.matmul(params.J_B_inv,(params.L_B - np.matmul(params.W,L_w) - (np.cross(-omega_B, np.matmul(params.J_B,omega_B) + np.matmul(params.W,H_B_w)))))

        # Combine ODE's into one column vector to output
        x_dot = np.concatenate((q_dot, omega_B_dot, omega_w_dot, i_dot, Th_Ta_dot, Tw_Ta_dot))

        return x_dot

def u(t,x):
    # The input is the pwm signal for each of the 4 rxn wheels, and
    # then the 4 external torques on each of the 4 rxn wheels
    MAX_PWM = 65535 # pwm val that gives max speed (2^16-1)

    pwm = np.array([0, 3000, 0, 0])
    # External torques (set to 0 for now)
    tau_e = np.array([0, 0, 0, 0]) # 

    u = np.concatenate((pwm, tau_e))
    return u

def define_init_states():
    #______________Define Initial States, x0__________________#
    # Define initial state vector
    # Initial Quaternion (use identity quaternion)
    # NOTE: Fundamentals book uses q = [q1:3 q4] (3-vector part q1:3 and
    # scalar part q4
    # IrishSat uses q = [q0 q1:3] (scalar part is first instead of last)
    # We will continue with the IrishSat representation 
    q_0 = np.array([1, 0, 0, 0]) # [q0 q1:3]
    #q_0 = q_0/np.linalg.norm(q_0)

    # Angular velocity of the body with respect to inertial frame
    omega_B_0 = np.array([0, 0, 0]) # [rad/s]

    # Angular velocity of reaction wheels with respect to their
    # respective axis
    omega_w_0 = np.array([0, 0, 0, 0]) # [rad/s]

    # Initial current going to each motor
    i_0 = np.array([0, 0, 0, 0]) # [A]

    # Initial Th-Ta
    Th_Ta_0 = np.array([0, 0, 0, 0]) 

    # Initial Tw-Ta
    Tw_Ta_0 = np.array([0, 0, 0, 0])

    # Combine into one initial state vector
    x0 = np.concatenate((q_0, omega_B_0, omega_w_0, i_0, Th_Ta_0, Tw_Ta_0))
    return x0

class params:
    # Importing motor parameters - Maxon DCX 8 M (9 volts)
    Rwa = 3.54      # Ohms, winding resistance at ambient temperature
    Lw = 0.424e-3  # Motor inductance [Henry]
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

## Define system 
# Define ode45 parameters
t0 = 0
tf = 4
dt = .02
n = int((tf-t0)/dt)
tim = np.linspace(t0, tf, n+1) # Array of time 

# Define initial states
x0 = define_init_states()

#______________Solve System__________________#
# Wrap function into function just in terms of t and x
F = lambda t, x: f(t,x,u(t,x),params)

solver_start = time.time()
# Solve system of ODE's
sol = solve_ivp(F, [t0, tf], x0, method = 'Radau', t_eval = tim)
print("Solver time = ", time.time()-solver_start)


# Generate gaussian noises to simulate real data collection
# Need different values of noise because omegas and q's have different orders of magnitude
noise_q = np.random.normal(0,.04,np.shape(sol.y[0,:])) 
noise_omegaB = np.random.normal(0,.04,np.shape(sol.y[4,:])) 
noise_omegaw = np.random.normal(0,10,np.shape(sol.y[7,:])) 

# Redefine state vector as more understandable things to plot
# We will continue with the IrishSat representation 
q = np.array([sol.y[0,:], sol.y[1,:], sol.y[2,:], sol.y[3,:]]) # [q0 q1:3]
# normalize quaternion
# for i in range(1,len(q)):
#      q[:,i] = q[:,i]/np.linalg.norm(q[:,i])
# Add noise to simulate real data collection
q_noisy = q + noise_q

# Angular velocity of the body with respect to inertial frame
omega_B = np.array([sol.y[4,:], sol.y[5,:], sol.y[6,:]])
# Add noise to simulate real data collection
omega_B_noisy = omega_B + noise_omegaB

# Angular velocity of reaction wheels with respect to their
# respective axis
omega_w = np.array([sol.y[7,:], sol.y[8,:], sol.y[9,:], sol.y[10,:]])
# Add noise to simulate real data collection
omega_w_noisy = omega_w + noise_omegaw

# Generate pwm values from states
pwm = np.array(u(sol.t[0],sol.y[0,:])[0:4])
tau_e = np.array(u(sol.t[0],sol.y[0,:])[4:8])
# Loop through u to find values at each time step
for i in range(1,len(sol.t)):
    pwm = np.vstack((pwm, np.array(u(sol.t[i],sol.y[:,i])[0:4])))
    tau_e = np.vstack((tau_e, np.array(u(sol.t[i],sol.y[:,i])[4:8])))

u_traj = np.transpose(np.hstack((pwm,tau_e)))


#_______________________________Observer - Linearized Discrete Kalman Filter__________________________________#

def symbolic():
    # Setup symbolic equations so that the partial derivative with respect to each state can be taken
    # Symbolic states
    q0_sym, q1_sym, q2_sym, q3_sym = sympy.symbols('q0_sym, q1_sym, q2_sym, q3_sym', real=True)
    q_sym = np.array([q0_sym, q1_sym, q2_sym, q3_sym])

    omega_Bx_sym, omega_By_sym, omega_Bz_sym = sympy.symbols('omega_Bx_sym, omega_By_sym, omega_Bz_sym', real=True)
    omega_B_sym = np.array([omega_Bx_sym, omega_By_sym, omega_Bz_sym])

    omega_wx_sym, omega_wy_sym, omega_wz_sym, omega_wskew_sym = sympy.symbols('omega_wx_sym, omega_wy_sym, omega_wz_sym, omega_wskew_sym', real=True)
    omega_w_sym = np.array([omega_wx_sym, omega_wy_sym, omega_wz_sym, omega_wskew_sym])


    ix_sym, iy_sym, iz_sym, iskew_sym = sympy.symbols('ix_sym, iy_sym, iz_sym, iskew_sym', real=True)
    i_sym = np.array([ix_sym, iy_sym, iz_sym, iskew_sym])

    Th_Tax_sym, Th_Tay_sym, Th_Taz_sym, Th_Taskew_sym = sympy.symbols('Th_Tax_sym, Th_Tay_sym, Th_Taz_sym, Th_Taskew_sym', real=True)
    Th_Ta_sym = np.array([Th_Tax_sym, Th_Tay_sym, Th_Taz_sym, Th_Taskew_sym])

    Tw_Tax_sym, Tw_Tay_sym, Tw_Taz_sym, Tw_Taskew_sym = sympy.symbols('Tw_Tax_sym, Tw_Tay_sym, Tw_Taz_sym, Tw_Taskew_sym ', real=True)
    Tw_Ta_sym = np.array([Tw_Tax_sym, Tw_Tay_sym, Tw_Taz_sym, Tw_Taskew_sym ])

    x_sym = np.concatenate((q_sym, omega_B_sym, omega_w_sym, i_sym, Th_Ta_sym, Tw_Ta_sym))

    # Symbolic inputs
    pwmx_sym, pwmy_sym, pwmz_sym, pwmskew_sym = sympy.symbols('pwmx_sym, pwmy_sym, pwmz_sym, pwmskew_sym', real=True)
    Vin_sym = np.array([pwmx_sym, pwmy_sym, pwmz_sym, pwmskew_sym])

    tau_ex_sym, tau_ey_sym, tau_ez_sym, taueskew_sym = sympy.symbols('tau_ex_sym, tau_ey_sym, tau_ez_sym, taueskew_sym', real=True)
    tau_e_sym = np.array([tau_ex_sym, tau_ey_sym, tau_ez_sym, taueskew_sym])

    u_sym = np.concatenate((Vin_sym, tau_e_sym))

    t_sym = 0 # This is just a placeholder, t_sym not actually used but need it for function call

    f_sym = f(t_sym,x_sym,u_sym,params)

    return x_sym, u_sym, f_sym

def linearize(x_sym, u_sym, f_sym, params):
    # Initialize Dx and Du with row of zeros
    Dx = np.zeros(len(x_sym))
    Du = np.zeros(len(u_sym))

    # Loop through each function and state and generate Dx
    for i in range(0,len(f_sym)):
        # initialize row of Dx and Du
        row_x = np.array([0])
        row_u = np.array([0])

        # Loop through each state taking partial with respect to that state
        for j in range(0,len(x_sym)):
            partial_x = np.array([sympy.diff(f_sym[i],x_sym[j])])
            row_x = np.concatenate((row_x, partial_x))

            if j < len(u_sym):
                partial_u = np.array([sympy.diff(f_sym[i],u_sym[j])])
                row_u = np.concatenate((row_u, partial_u))

        # Get rid of first 0 in row that was used to initialize
        row_x = row_x[1:]
        row_u = row_u[1:]

        Dx = np.vstack((Dx,row_x))
        Du = np.vstack((Du,row_u))
    
    # Delete first row of zeros used to initialize
    Dx = Dx[1:,:]
    Du = Du[1:,:]

    return sympy.Matrix(Dx), sympy.Matrix(Du)

# Creat symbolic variables used to get jacobian
x_sym, u_sym, f_sym = symbolic()

# Generate jacobians for linearization
Dx, Du = linearize(x_sym,u_sym,f_sym, params)

# Convert sympy arrays to functions for speed
substitute_Dx = sympy.lambdify([x_sym, u_sym], Dx,"numpy")
substitute_Du = sympy.lambdify([x_sym, u_sym], Du,"numpy")

# Get number of states and number of inputs
x_len = len(sol.y[:,0])
u_len = len(u_traj[:,0])

# Define output, assume we only have measurements for q, omega_B, omega_w (first 11 states)
y_real = np.vstack([q_noisy, omega_B_noisy, omega_w_noisy])

# Initialize all the matrices for the linearized discrete Kalman filter
def initialize(x_len, u_len, y_real, n, P_gain, Q_gain, R_gain):
    # Initialize covariance matrix for sensors - assume they all have the same variance for now
    R = R_gain*np.eye(len(y_real[:,0]))
    #R = np.diag([.05, .05, .05, .05, .07, .07, .07, 10, 10, 10, 10]) # this is constant

    # Initialize covariance of model ("Gaussian zero mean white noise" - Optimal State Estimation pg 107)
    Q = Q_gain*np.eye(x_len) # this is constant 

    # Initialize covariance of estimation error matrix
    P_minus = np.zeros([x_len,x_len,n+1]) # this changes for each time step (that is why it is setup as 3D array)
    P_plus = np.zeros([x_len,x_len,n+1])

    # Set up P of initial state (this is basically just a guess of how far off our first estimate is)
    # just use initial variance of sensors for states we have sensor for, and 0 for everything else 
    P_plus[:,:,0] = P_gain*np.eye(x_len)
    #P_plus[:,:,0] = np.diag([.05, .05, .05, .05, .5, .5, .5, .5, .5, .5, .5, .01, .01, .01, .01, .01, .01, .01, .01,.01, .01, .01, .01])

    # initialize F,G,H 3D arrays
    # note we are switching notation to Optimal State Estimation (Simon) book
    F = np.zeros([x_len, x_len, n+1]) 
    G = np.zeros([x_len, u_len, n+1])
    H = np.zeros([len(y_real[:,0]), x_len, n+1])

    # Initialize F, G, H
    # Solve for A and B matrices of linear form using Jacobians
    A = substitute_Dx(x0, u_traj[:,0])
    B = substitute_Du(x0, u_traj[:,0])
    C = np.eye(len(y_real[:,0]))
    C = np.hstack([C,np.zeros([len(y_real[:,0]), x_len-11])])  # assume we only have measurements for q, omega_B, omega_w (first 11 states)
    D = 0 

    # Convert to discrete time 
    sys_c = ct.StateSpace(A,B,C,D)
    sys_d = ct.c2d(sys_c,dt)

    F[:,:,0] = sys_d.A
    G[:,:,0] = sys_d.B
    H[:,:,0] = sys_d.C

    # Initialize posteri observer state
    xh_plus = np.zeros([x_len,n+1])
    xh_plus[:,0] = x0

    return R,Q,P_minus,P_plus,C,D,F,G,H,xh_plus

# Set initial values of covariance matrices
P_gain = 1e-3 # total estimate
Q_gain = 5e-4 # model
R_gain = 1e-1 # sensor

# Initialize system
R,Q,P_minus,P_plus,C,D,F,G,H,xh_plus = initialize(x_len,u_len,y_real,n,P_gain,Q_gain,R_gain)

def KalmanFilter(y, H, K, F_minus1, G_minus1, delta_x_minus1, delta_u_minus1):
    # A priori state estimate
    xh_minus = F_minus1 @ delta_x_minus1 #+ G_minus1 @ delta_u_minus1 

    # Recursive Least-squares
    xh_plus = xh_minus + K @ (y - H @ xh_minus) 

    return xh_plus

# Initialize control input
#delta_u = 1e-2*np.ones([u_len,n+1])
u_out = np.zeros([u_len,n+1])
u_out[:,0] = u_traj[:,0]

kalman_begin = time.time()

# Generate linearized dynamics matrix for each time step
for i in range(1,n+1):
    # Linearize system using the jacobian
    A = substitute_Dx(xh_plus[:,i-1], u_out[:,i-1])
    B = substitute_Du(xh_plus[:,i-1], u_out[:,i-1])

    # Convert to discrete time
    sys_c = ct.StateSpace(A,B,C,D)
    sys_d = ct.c2d(sys_c,dt)

    # Generate discrete time matrices in OSE (Simon) notation
    F[:,:,i] = sys_d.A
    G[:,:,i] = sys_d.B
    H[:,:,i] = sys_d.C

    # P, K, delta_x are all calculated here as given in Eq 5.19 on pg 128-129 in Optimal State Estimation (Simon)
    # Update covariance matrix 
    P_minus[:,:,i] = F[:,:,i-1] @ P_plus[:,:,i-1] @ F[:,:,i-1].T + Q 

    # Gain for recursive least squares (same as L in closed loop observer)
    K = (P_minus[:,:,i] @ H[:,:,i].T) @ np.linalg.inv(H[:,:,i] @ P_minus[:,:,i] @ H[:,:,i].T + R)
    
    # Get posteri state estimate
    xh_plus[:,i] = KalmanFilter(y_real[:,i], H[:,:,i], K, F[:,:,i-1], G[:,:,i-1], xh_plus[:,i-1], u_out[:,i-1])
    
    # Posteri covariance
    P_plus[:,:,i] = (np.eye(x_len) - K @ H[:,:,i]) @ P_minus[:,:,i] 


print("Kalman time = ", time.time() - kalman_begin)

# #______________Least squares - Signal smoothing and higher order derivatives_____________#
def smoothCVX(sig, alp):
    '''
    Smoothing function written by Prof. Edgar Bolivar

    Takes in noisy signal and smoothing level alpha and outputs the smoothed version of the 
    input and its first 2 derivatives

    '''
    # Find a set of points that are close to sig but minimize jerk

    #-Create derivative operator Matrix (Center difference)
    n = len(sig)
    e = np.ones(n)
    # First derivative
    D1 = scipy.sparse.spdiags([-e, 0*e, e], [-1, 0, 1], n, n).toarray()
    D1[0,:] = D1[1,:]
    D1[-1,:] = D1[-2,:]

    # Second derivative
    D2 = scipy.sparse.spdiags([-e, 0*e, e], [-1, 0, 1], n, n).toarray()
    D2[0,:] = D2[1,:]
    D2[-1,:] = D2[-2,:]

    # Third derivative (Jerk)
    D3 = D1@D2

    # Minimize the norm of signal fit and jerk
    m = D3.shape[0]
    A   = np.vstack((np.eye(n), alp*D3))
    y   = np.hstack((sig, np.zeros(m)))
    
    #Adjust A for shape
    A_psuedo_inv = np.linalg.pinv(A)
    fil = A_psuedo_inv @ y

    return fil, D1, D2

#______________Least squares - Parameter estimation_____________#
#use smoothing function to obtain di_dt
[smo_omega_w0, D1, D2] = smoothCVX(omega_w_noisy[0,:], 3)

def estimate_inductance(smo_omega_w0, D1, D2, tim, t,y):
    #Scale derivative matrices to account for time
    samTim = tim[1] - tim[0]
    D1 = D1*(1/(2*samTim))
    D2 = D2*(1/(samTim**2))

    di_dt = D1 @ smo_omega_w0

    #define variables
    Vin = (9/65535)*u(t[0],y[0,:])[0:4]
    i = sol.y[11,:] # Current to each motor
    Tw_Ta = sol.y[19,:]
    Rw = params.Rwa*(1+params.alpha_Cu*Tw_Ta)

    #Define Ground truth 
    g = (Vin[0]-i*Rw - params.Kv*smo_omega_w0)
    #Adjust R matrix shape
    regressor_matrix = di_dt[:, np.newaxis]
                                                
    #use lstsq function to solve least squares solution to identify L (inductance) in electrical circuit for DC motor in rxn wheels
    x, residuals, rank, singular_values = np.linalg.lstsq(regressor_matrix, g, rcond=None)
    L = x
    return L

L = estimate_inductance(smo_omega_w0, D1, D2, tim, sol.t,sol.y)
print("L: ",L)

#______________Plotting__________________#

# Plot components of quaternion
q_plot = plt.figure()
plt.plot(sol.t, q_noisy[0,:], sol.t, q_noisy[1,:], sol.t, q_noisy[2,:], sol.t, q_noisy[3,:])
plt.plot(sol.t, q[0,:], sol.t, q[1,:], sol.t, q[2,:], sol.t, q[3,:])
plt.plot(sol.t,xh_plus[0,:],sol.t,xh_plus[1,:],sol.t,xh_plus[2,:],sol.t,xh_plus[3,:])
plt.xlabel('t (s)')
plt.ylabel('$q$')
plt.legend(['$q_0$', '$q_1$', '$q_2$','$q_3$'])

# Plot angular velocities of cubesat body
omegaB_plot = plt.figure()
plt.plot(sol.t, omega_B[0,:], sol.t, omega_B[1,:], sol.t, omega_B[2,:])
plt.xlabel('t (s)')
plt.ylabel('$\omega$ $_B^{BI}$ [rad/s]')
plt.legend(['$\omega_x^{BI}$', '$\omega_y^{BI}$','$\omega_z^{BI}$'])

# Plot angular velocities of rxn wheels
omegaw_plot = plt.figure()
plt.plot(sol.t,omega_w[0,:],sol.t,omega_w[1,:],sol.t,omega_w[2,:],sol.t,omega_w[3,:])
plt.xlabel('t (s)')
plt.ylabel('$\omega$ $_B^w$ [rad/s]')
plt.legend(['$\omega_x^w$', '$\omega_y^w$','$\omega_z^w$','$\omega_{skew}^w$'])

# Plot input
pwm_plot = plt.figure()
plt.plot(sol.t,pwm[:,0],sol.t,pwm[:,1],sol.t,pwm[:,2],sol.t,pwm[:,3])
plt.xlabel('t (s)')
plt.ylabel('pwm signal')
plt.legend(['$pwm_x^w$', '$pwm_y^w$','$pwm_z^w$','$pwm_{skew}^w$'])


# Plot observer vs actual
q0_plot_obs = plt.figure()
plt.plot(sol.t, q_noisy[0,:],alpha=.5)
plt.plot(sol.t, q[0,:])
plt.plot(sol.t,xh_plus[0,:])
plt.xlabel('t (s)')
plt.ylabel('$q_0$')
plt.legend(['Nonlinear Simulated Sensor Input','Actual Values','Observer'])

q1_plot_obs = plt.figure()
plt.plot(sol.t, q_noisy[1,:],alpha=.5)
plt.plot(sol.t, q[1,:])
plt.plot(sol.t,xh_plus[1,:]) 
plt.xlabel('t (s)')
plt.ylabel('$q_1$')
plt.legend(['Nonlinear Simulated Sensor Input','Actual Values','Observer'])

q2_plot_obs = plt.figure()
plt.plot(sol.t, q_noisy[2,:]) 
plt.plot(sol.t, q[2,:])
plt.plot(sol.t,xh_plus[2,:]) 
plt.xlabel('t (s)')
plt.ylabel('$q_2$')
plt.legend(['Nonlinear Simulated Sensor Input','Actual Values','Observer'])

q3_plot_obs = plt.figure()
plt.plot(sol.t, q_noisy[3,:])
plt.plot(sol.t, q[3,:])
plt.plot(sol.t,xh_plus[3,:])
plt.xlabel('t (s)')
plt.ylabel('$q_3$')
plt.legend(['Nonlinear Simulated Sensor Input','Actual Values','Observer'])

omega_plot_obs = plt.figure()
plt.plot(sol.t, omega_B_noisy[0,:])
plt.plot(sol.t, omega_B[0,:])
plt.plot(sol.t,xh_plus[4,:])
plt.xlabel('t (s)')
plt.ylabel('$\omega_{Bx}$')
plt.legend(['Nonlinear Simulated Sensor Input','Actual Values','Observer'])

current_plot_obs = plt.figure()
plt.plot(sol.t, sol.y[11,:])
plt.plot(sol.t,xh_plus[11,:])
plt.xlabel('t (s)')
plt.ylabel('$i_x$')
plt.legend(['Actual Values','Observer'])

print("Total time = ",time.time()-begin_time)

plt.show()