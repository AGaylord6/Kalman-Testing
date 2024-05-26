'''
tests.py
Author: Andrew Gaylord

Contains statistical, efficiency, and correctness for kalman filter objects

'''


'''
one of the ways to check Kalman filters performance is to check for error covariance matrix P 
to be converging. If it converges to + or - standard deviation of the estimated value, 
it can be considered as a stable point. 
calculate square of difference between estimated and real
You can verify that the estimated state converges to the actual state. 
The error covariance, P, must decrease.

innovation (V) or residual: difference between a measurement and its prediction at time k
    measures new info provided by adding another measurement in estimation
    can be used to validate a measurement before adding to observation sequence

    innovation tests: test that innovation has zero mean and white with cov S_k
        1) check that it is consistent with its cov/within bounds--checks filter consistency
        2) chi square test for unbiased 
        3) whiteness (autocorrelation) test

'''

import numpy as np
from graphing import *
from filter import *


# function that returns true if 95% of the innovations are within 2 standard deviations of the mean
# test #1 for filter consistency
# parameters: innovation arrays, innovation covariance arrays
def innovationTest(innovations, innovationCovs, dim_mes):
    innovationMags = np.array([np.linalg.norm(x) for x in innovations])
    innovationCovMags = innovationCovToStd(innovationCovs, dim_mes)

    upper = innovationMags + 2 * innovationCovMags
    lower = innovationMags - 2 * innovationCovMags

    # only returns true if 95% of innovations are within 2 standard deviations
    return np.sum((innovationMags < upper) & (innovationMags > lower)) / len(innovationMags) > .95


def plotInnovations(innovations, innovationCovs):
    # method for plotting each innovation and their respective standard deviations on separate graphs
    # for i in range(len(innovations[0])):
    #     sds =  np.array([np.sqrt(np.diag(x)) for x in innovationCovs])
    #     plot_multiple_lines(np.array([innovations[:, i], innovations[:, i] + 2 * sds[:, i], innovations[:, i] - 2 * sds[:, i]]), ["innovation magnitude", "upper sd", "lower sd"], "innovation " + str(i+1), 100 + i*50, 100 + i*50)

    # method using magnitude of whole innovation array
    # find magnitudes of innovation arrays
    # innovationMags = np.array([np.linalg.norm(x) for x in ukf.innovations])

    # to get standard deviation, take sqrt of diagonal
    # divide by number of observations to get standard error of mean
    # get magnitude afterwards
    # innovationCovMags = np.array([(np.linalg.norm(y)/ ukf.dim_mes) for y in np.array([np.sqrt(np.diag(x)) for x in ukf.innovationCovs])])

    # find upper and lower bounds of 2 * standard deviation
    # upper = innovationMags + 2 * innovationCovMags
    # lower = innovationMags - 2 * innovationCovMags

    # plot to check whether innovation is centered on 0 and 95% of measurements are consistent with standard deviation
    # plot_multiple_lines(np.array([innovationMags, upper, lower]), ["innovation magnitude", "upper sd", "lower sd"], "innovation", 300, 200)


    # plot orientation and velocity innovations on separate graphs
    plotVelocityInnovations(innovations, innovationCovs)
    plotOrientationInnovations(innovations, innovationCovs)

# plot the last 3 innovations and their respective standard deviations on 1 graph
def plotVelocityInnovations(innovations, innovationCovs):
    innovations = innovations[1:]
    innovationCovs = innovationCovs[1:]

    sds =  np.array([np.sqrt(np.diag(x)) for x in innovationCovs])
    plot_multiple_lines(np.array([innovations[:, 3], innovations[:, 3] + 2 * sds[:, 3], innovations[:, 3] - 2 * sds[:, 3], innovations[:, 4], innovations[:, 4] + 2 * sds[:, 4], innovations[:, 4] - 2 * sds[:, 4], innovations[:, 5], innovations[:, 5] + 2 * sds[:, 5], innovations[:, 5] - 2 * sds[:, 5]]), ["velocity 1", "upper 1", "lower 1", "velocity 2", "upper 2", "lower 2", "velocity 3", "upper 3", "lower 3"], "innovation magnitudes", 900, 100)
    
# plot the first 3 innovations on their respective standard deviations on 1 graph
def plotOrientationInnovations(innovations, innovationCovs):
    innovations = innovations[1:]
    innovationCovs = innovationCovs[1:]

    sds =  np.array([np.sqrt(np.diag(x)) for x in innovationCovs])
    plot_multiple_lines(np.array([innovations[:, 0], innovations[:, 0] + 2 * sds[:, 0], innovations[:, 0] - 2 * sds[:, 0], innovations[:, 1], innovations[:, 1] + 2 * sds[:, 1], innovations[:, 1] - 2 * sds[:, 1], innovations[:, 2], innovations[:, 2] + 2 * sds[:, 2], innovations[:, 2] - 2 * sds[:, 2]]), ["orientation 1", "upper 1", "lower 1", "orientation 2", "upper 2", "lower 2", "orientation 3", "upper 3", "lower 3"], "innovation magnitudes", 100, 100)




# test #2 for unbiasedness
# function that returns true if the innovations are unbiased
def unbiasedTest(innovations):
    return np.mean(innovations) == 0



# function that returns true if the error covariance matrix P is converging
def covarianceConvergence(P, threshold):
    return np.allclose(P, threshold, atol=1e-2)

# function that takes in an array of innovation covariances and returns an array of the standard deviations of the innovations
def innovationCovToStd(innovationCovs, dim_mes):
    return np.array([(np.linalg.norm(y)/ dim_mes) for y in np.array([np.sqrt(np.diag(x)) for x in innovationCovs])])



# function that finds normalised autocorrelation for each innovation sequence in a 2D array
def autocorrelation2D(innovations):
    return np.array([autocorrelation(x) for x in innovations])
# 
# function that plots a 2D array of autocorrelations
def plotAutocorrelation2D(innovations):
    plot_multiple_lines(autocorrelation2D(innovations), ["autocorrelation"], "autocorrelation", 300, 200)
# 
def autocorrelation(innovations):
    return np.correlate(innovations, innovations, mode='full') / np.linalg.norm(innovations)

# function that returns true if the innovations are white
def whiteTest(innovations):
    return np.allclose(autocorrelation(innovations), 0, atol=1e-2)



# function that returns true if the innovations are consistent with their covariance
def consistentTest(innovations, innovationCovs):
    return innovationTest(innovations, innovationCovs, 6) and whiteTest(innovations) and unbiasedTest(innovations)

# function that finds normalised innovation and moving average
def movingAverage(innovations, window):
    return np.convolve(innovations, np.ones(window), 'valid') / window

# function that performs normalised innovations chi square test 
def chiSquareTest(innovations, innovationCovs):
    return np.sum(innovations ** 2 / innovationCovs) / len(innovations) < 1.5