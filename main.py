'''
main.py
Author: Andrew Gaylord

uses fake models to simulate and compare different kalman filters

'''

from irishsat_ukf.PySOL.wmm import *
from irishsat_ukf.simulator import *
from irishsat_ukf.UKF_algorithm import *
from irishsat_ukf.hfunc import *


def initalize ():

    # number of steps to calculate
    n = 1000
    dt = 0.1
    # dimensionality of state space
    dim = 7
    # dimensionality of measurement space
    dimMes = dim - 1
    speed = 1
    # starting quaternion for propogator
    initQ = np.array([1, 0, 0, 0])
    # starting state estimate (should match initQ and w)
    start = [1, 0, 0, 0, 0, speed, speed/2]
    # angular velocity
    w = np.array([0, speed, speed/2])
    # starting covariance
    cov = np.identity(dim) * 5e-10
    # constant B field
    B_true = np.array([0, 0, 1])
    # reaction wheel speeds (0 for this test)
    reaction_speeds = np.zeros(3)
    old_reaction_speeds = np.zeros(3)

    # note: if model is less reliable/changes quickly, then q > r
    # r: measurement noise (m x m)
    noiseMagnitude = 0.01
    r = np.diag([noiseMagnitude] * dimMes)

    # q: process noise (n x n)
    noiseMagnitude = 0.005
    q = np.diag([noiseMagnitude] * dim)

    t0 = 0
    tf = 100
    i = 0

    # uncomment to store results and visualize after calculating
    # results = [[]]

    # initialize propogator object with inital quaternion and angular velocity
    propagator = AttitudePropagator(q_init=initQ, w_init=w)

    # use attitude propagator to find actual ideal quaternion for n steps
    states = propagator.propagate_states(t0, tf, n)

    # calculate sensor b field for every time step
    # rotation matrix(q) * true B field + noise
    # first value, then all the otheres
    B_sens = np.array([np.matmul(hfunc.quaternion_rotation_matrix(states[0]), B_true)])
    for a in range(1, n):
        B_sens = np.append(B_sens, np.array([np.matmul(hfunc.quaternion_rotation_matrix(states[a]), B_true)]), axis=0)

    # need to add small sensor noise
    
    while i < 1000:

        # create sensor data matrix of magnetomer reading and angular velocity
        data = [0] * dimMes
        data[0] = B_sens[i][0]
        data[1] = B_sens[i][1]
        data[2] = B_sens[i][2]
        data[3] = w[0]
        data[4] = w[1]
        data[5] = w[2]

        # run ukf algorithm for each iteration
        # note: for this test, b field is passed as gps_data instead of gps data
        start, cov = UKF_algorithm.UKF(start, cov, q, r, list(B_true), reaction_speeds, old_reaction_speeds, data)

        # uncomment to run fully and visualize after
        # results.append(list(start[:4]))

        # debug print statements
        # print("Data: ", data)
        print("Data: ", ["{:0.4f}".format(x) for x in data])
        # print("State: ", start[:4])
        print("State quaternion: ", ["{:0.4f}".format(x) for x in start[:4]])
        # print("Ideal quaternion: ", states[i])
        print("Ideal quaternion: ", ["{:0.4f}".format(x) for x in states[i]])

        print("")

        # draw our estimate's quaternion
        game_visualize(np.array([start[:4]]), i)

        # draw ideal state quaternion
        # game_visualize(np.array([states[i]]), i)
        i += 1


if __name__ == "__main__":



    # initialize starting state and covariance
    # starting angle, starting speed
    state = np.array([0, 0]) 
    cov = np.identity(dim) * 5e-10

    # r: measurement noise (m x m)
    # smaller value = trust it more = source is less noisy
    noise_mag = 5
    # 50 fine, best results with 5
    noise_gyro = 1
    # .5 fine, 1 best

    r = np.array([[noise_mag, 0, 0],
                [0, noise_mag, 0],
                [0, 0, noise_gyro]])

    # q: process noise (n x n)
    # Should depend on dt
    noise_mag = .5
    # small (.05) = rocky. 0.5 best
    q = np.array([[noise_mag*dt, 0],
                [0, noise_mag* (dt**2)]
    ])
