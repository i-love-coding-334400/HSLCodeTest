#  Statistical Learning Method Chapter 9 Exercise 9.3
#  Question description: Try to estimate 5 parameters of Gaussian Mixture Model with 2 components based on observed data: -67,-48,6,8,14,16,23,24,28,29,41,49,56,60,75

import random
import numpy as np


def calculate_norm(y, mu, sigma):
    """
    here is not need to be Gaussian distribution, but thr gaussian is simple.
    because the EM Step can be calculated easily bu simple formula.
    But if you use other distribution this is same for calculate the joint distribution of mu and sigma.
    """
    return (np.sqrt(2 * np.pi * sigma * sigma) ** -1) * np.e ** (-((y - mu) ** 2) / (2 * sigma * sigma))


def initialize_parameter():
    mu1 = random.randint(-100, 100)
    sigma1 = random.randint(100, 1000)
    mu2 = random.randint(-100, 100)
    sigma2 = random.randint(100, 1000)
    # only two gaussian components, one's weight is weight1,and the other's weight is 1-weight1
    weight1 = random.randint(1, 9)/10
    print(mu1, mu2, sigma1, sigma2, weight1)
    return mu1, sigma1, mu2, sigma2, weight1


def E_step(y, mu1, sigma1, mu2, sigma2, weight1):
    size = len(y)
    gussian_distribution1 = np.zeros(size)
    gussian_distribution2 = np.zeros(size)
    gamma_j1 = np.zeros(size)
    gamma_j2 = np.zeros(size)
    for i in range(size):
        gussian_distribution1[i] = calculate_norm(y[i], mu1, sigma1)
        gussian_distribution2[i] = calculate_norm(y[i], mu2, sigma2)
        gamma_j1[i] = (weight1 * gussian_distribution1[i]) / (
                weight1 * gussian_distribution1[i] + (1 - weight1) * gussian_distribution2[i])
        gamma_j2[i] = ((1 - weight1) * gussian_distribution2[i]) / (
                weight1 * gussian_distribution1[i] + (1 - weight1) * gussian_distribution2[i])
    # print("gussian_distribution1", gussian_distribution1)
    # print("gussian_distribution2", gussian_distribution2)
    # print("gamma_j1", gamma_j1)
    # print("gamma_j2", gamma_j2)
    return gamma_j1, gamma_j2


def M_step(y, gamma_j1, gamma_j2, mu1, mu2):
    size = len(y)
    sum1 = np.zeros(2)
    sum2 = np.zeros(2)
    sum3 = np.zeros(2)
    for i in range(size):
        sum1[0] += gamma_j1[i] * y[i]
        sum2[0] += gamma_j1[i]
        sum3[0] += gamma_j1[i] * (y[i] - mu1) * (y[i] - mu1)
        ...
        sum1[1] += gamma_j2[i] * y[i]
        sum2[1] += gamma_j2[i]
        sum3[1] += gamma_j2[i] * (y[i] - mu2) * (y[i] - mu2)
    ...
    mu1_new = sum1[0] / sum2[0]
    sigma1_new = np.sqrt(sum3[0] / sum2[0])
    mu2_new = sum1[1] / sum2[1]
    sigma2_new = np.sqrt(sum3[1] / sum2[1])
    weight1_new = sum2[0] / size
    return mu1_new, sigma1_new, mu2_new, sigma2_new, weight1_new


if __name__ == '__main__':
    y = [-67, -48, 6, 8, 14, 16, 23, 24, 28, 29, 41, 49, 56, 60, 75]
    # if parameters are random initialized, the result will be NaN
    mu1, sigma1, mu2, sigma2, weight1 = initialize_parameter()
    # mu1, sigma1, mu2, sigma2, weight1 = [30, 500, -30, 500, 0.5]
    epoch = 100
    for _ in range(epoch):
        gamma_j1, gamma_j2 = E_step(y, mu1, sigma1, mu2, sigma2, weight1)
        mu1, mu2, sigma1, sigma2, weight1 = M_step(y, gamma_j1, gamma_j2, mu1, mu2)
        print(mu1, mu2, sigma1, sigma2, weight1)
