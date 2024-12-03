import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

import torch.nn.functional as F
import math
def generate_user_in_line_multipath(Nt, d, r_min, r_max, f, N_user, sigma_aod, L, kappa):
    c = 3e8
    H_multi_user = torch.complex(torch.zeros(N_user, Nt),torch.zeros(N_user, Nt))
    # uniform random in line
    theta = torch.rand(1, 1) * 2 * np.pi / 3 - np.pi / 3
    r_list = torch.rand(1, N_user) * (r_max - r_min) + r_min
    theta_list = theta * torch.ones(1, N_user)

    nn = torch.arange(-(Nt - 1) / 2, (Nt - 1) / 2 + 1)
    theta_aod = torch.sqrt(torch.tensor(sigma_aod)) * torch.randn(N_user, L)
    ssf = (torch.randn(N_user, L) + 1j * torch.randn(N_user, L)) / np.sqrt(2)
    # allocate a factor to fade the NLoS channel
    alpha = 1
    beta = torch.sqrt(torch.tensor(alpha / L))
    ssf = ssf * beta

    for i_user in range(N_user):
        for l in range(L + 1):
            if l != L:
                r0 = r_list[0, i_user]
                theta0 = theta_list[0, i_user] + theta_aod[i_user, l]
                r = torch.sqrt(r0 ** 2 + (nn * d) ** 2 - 2 * r0 * nn * d * torch.sin(theta0))
                at = torch.exp(-1j * 2 * np.pi * f * (r - r0) / c) / torch.sqrt(torch.tensor(Nt))
                H_multi_user[i_user, :] = H_multi_user[i_user, :] + ssf[i_user, l] * at * torch.sqrt(torch.tensor(1 / (1 + kappa)))
            else:
                r0 = r_list[0, i_user]
                theta0 = theta_list[0, i_user]
                r = torch.sqrt(r0 ** 2 + (nn * d) ** 2 - 2 * r0 * nn * d * torch.sin(theta0))
                at = torch.exp(-1j * 2 * np.pi * f * (r - r0) / c) / torch.sqrt(torch.tensor(Nt))
                H_multi_user[i_user, :] = H_multi_user[i_user, :] + at * torch.sqrt(torch.tensor(kappa / (1 + kappa)))

    if L == 0:
        H_multi_user = H_multi_user / torch.sqrt(kappa / (1 + kappa))

    return H_multi_user

def RZF(H, pow):
    K, M = H.shape
    weights = torch.ones(K, 1)
    wSLNRMAX = functionSLNRMAX(H, pow * torch.ones(K, 1))
    #print(wSLNRMAX)
    rhos = torch.diag(torch.abs(torch.matmul(H, wSLNRMAX)) ** 2).T.conj().reshape(1, -1)


    powerAllocationwSLNRMAX_sumrate = functionHeuristicPowerAllocation(rhos, pow, weights)

    P = torch.mul(torch.kron(torch.sqrt(powerAllocationwSLNRMAX_sumrate), torch.ones((M, 1))), wSLNRMAX)
    return P

def functionSLNRMAX(H, eta=None, D=None):
    Kr = H.shape[0]
    N = H.shape[1]
    # If eta vector is not provided, all values are set to unity
    if eta is None:
        eta = torch.ones(Kr)

    # If D matrix is not provided, all antennas can transmit to everyone
    if D is None:

        D_real = torch.eye(N).unsqueeze(2).repeat(1, 1, Kr)
        D_imag = torch.zeros(N,N).unsqueeze(2).repeat(1, 1, Kr)
        D=torch.complex(D_real, D_imag)
    # Pre-allocation of MRT beamforming
    wSLNRMAX = torch.complex(torch.zeros(H.T.shape),torch.zeros(H.T.shape))

    # Computation of SLNR-MAX, based on Definition 3.5
    #print(H)
    for k in range(Kr):
        effectivechannel = torch.matmul(H, D[:, :, k]).conj().T  # Effective channels

        projectedchannel = torch.linalg.inv(
            (torch.complex(torch.eye(N),torch.zeros(N,N))) / eta[k] + torch.matmul(effectivechannel, effectivechannel.T.conj())) @ effectivechannel[:,k]  # Compute zero-forcing based on channel inversion
        real_part = torch.real(projectedchannel ) / torch.norm(projectedchannel)
        imaginary_part = torch.imag(projectedchannel ) /  torch.norm(projectedchannel)

        # 重新合成复数矩阵
        wSLNRMAX[:, k] = torch.complex(real_part,imaginary_part)
    #print(wSLNRMAX)

    return wSLNRMAX

def functionHeuristicPowerAllocation(rhos, q, weights):
    Kt = rhos.shape[0] # Number of base stations (BSs)
    Kr = rhos.shape[1] # Number of users (in total)
    powerallocation = torch.zeros((Kt, Kr)) # Pre-allocation of matrix for power allocation coefficients
    # Iteration over base stations to perform power allocation
    for j in range(Kt):
        indicesOfNonzero = torch.nonzero(rhos[j] > 0).view(-1)  # Find which users that are served by BS j
        m=indicesOfNonzero.shape[0]
        # Case 1: Compute waterlevel if all of the users served by BS j are allocated non-zero power.
        nuAllActive = (q + torch.sum(1 / rhos[j, indicesOfNonzero])) / torch.sum(weights[indicesOfNonzero])

        # Case 2: Compute waterlevel if only a subset of the users served by BS j are allocated non-zero power.
        # The range of the waterlevel is achieved by checking when there is equality in (3.37); that is, when users are activated.
        # The minimize_scalar function finds the waterlevel that minimizes the difference between the allocated power and available power.
        nuRangeLower = torch.min(1 / (rhos[j, indicesOfNonzero] * weights[indicesOfNonzero]))
        nuRangeUpper = torch.max(1 / (rhos[j, indicesOfNonzero] * weights[indicesOfNonzero]))
        res = minimize_scalar(
            lambda x: functionAllocDiff(x, q, rhos[j, indicesOfNonzero], weights[indicesOfNonzero]),
            bounds=(nuRangeLower.item(), nuRangeUpper.item()), method='bounded')
        nu = res.x

        # Check if the difference between the allocated power and the available power is minimized by allocating power to all users or only a subset.
        if functionAllocDiff(nu, q, rhos[j, indicesOfNonzero], weights[indicesOfNonzero]) < functionAllocDiff(
                nuAllActive, q, rhos[j, indicesOfNonzero], weights[indicesOfNonzero]):
            # Compute power allocation with optimal waterlevel (only a subset of users are active)
            powerallocation[j, indicesOfNonzero] = torch.max(torch.cat((weights[indicesOfNonzero] * nu - (1. / rhos[j, indicesOfNonzero]).view(m, -1), torch.zeros(len(indicesOfNonzero), 1)),dim=1), dim=1).values
        else:
            # Compute power allocation with optimal waterlevel (all users are active)

            powerallocation[j, indicesOfNonzero] = torch.max(torch.cat((weights[indicesOfNonzero] * nuAllActive - (1. / rhos[j, indicesOfNonzero]).view(m, -1), torch.zeros(len(indicesOfNonzero), 1)),dim=1), dim=1).values

        # Scale the power allocation to use full power (to improve numerical accuracy)
        powerallocation[j, :] = q * powerallocation[j, :] / torch.sum(powerallocation[j, :])

    return powerallocation


def functionAllocDiff(nu, q, rhos, weights):
    zeros_mask = torch.zeros(weights.size())
    max_values = torch.max(torch.cat((nu * weights - 1. / rhos, zeros_mask), dim=1), dim=1).values
    difference = torch.abs(torch.sum(max_values) - q)
    return difference






# def function_LSR(PRZF, N_RF):
#     # Implementation of function_LSR function in Python
#     # Replace with the actual implementation
#     pass
#
#
# def MO_AltMin(PRZF, N_RF):
#     # Implementation of MO_AltMin function in Python
#     # Replace with the actual implementation
#     pass


def SumRate(H, P, sigma2):
    M, K = H.size()
    c = torch.zeros(1, K)
    for idx1 in range(K):
        ds = torch.abs(torch.matmul(H[:, idx1].T, P[:, idx1])) ** 2
        inter = 0

        for idx2 in range(K):
            if idx2 != idx1:
                inter += torch.abs(torch.matmul(H[:, idx1].T, P[:, idx2])) ** 2

        sinr_k = ds / (sigma2 + inter)
        c[0, idx1] = torch.log2(1 + sinr_k)

    C = torch.sum(c)
    return C



if __name__ == '__main__':
    Nt = 128
    N_user = 8
    N_RF = 4
    fc = 30e9  # Frequency
    lambda_val = 3e8 / fc  # Wavelength
    d = lambda_val / 2
    r_circle_max = 100
    r_circle_min = 4
    sigma_aod = np.pi / 180 * 5
    L = 5
    kappa = 8
    snr = np.arange(-15, 6, 5)
    Pow = 10000
    realization = 20
    Normalization = 1
    smax = len(snr)  # enable the parallel
    sumratedata_rzf = np.zeros(smax)
    sumratedata_LSR = np.zeros(smax)
    sumratedata_MO = np.zeros(smax)
    sumratedata_AIGC = np.zeros(smax)

    for s in range(smax):
        print(s)
        SR_rzf = 0
        SR_LSR = 0
        SR_MO = 0
        SR_AIGC = 0
        sigma2 = Pow / (10 ** (snr[s] / 10))
        for reali in range(realization):
            H = generate_user_in_line_multipath(Nt, d, r_circle_min, r_circle_max, fc, N_user, sigma_aod, L, kappa)
            # real_part =  torch.randn(N_user, Nt)
            # imaginary_part = torch.randn(N_user, Nt)
            # #
            # H = torch.complex(real_part, imaginary_part)
            PRZF = RZF(H, Pow)
            print(torch.norm(PRZF, p='fro'))
            ##########################AIGC强化学习优化#############################
            #利用AIGC求出优化后的 F_RF,F_BB
            #以下代码随机生成一组变量值
            angles_RF = torch.empty(Nt, N_RF).uniform_(0, 2 * 3.14159265359)
            F_RF = torch.cos(angles_RF) + 1j * torch.sin(angles_RF)
            angles_BB = torch.empty(N_RF, N_user)
            F_BB = torch.cos(angles_BB) + 1j * torch.sin(angles_BB)

            PRZF_AIGC=torch.matmul(F_RF,F_BB)
            #print(PRZF_AIGC.size())

            #################################################################

            SR_rzf += SumRate(H.T, PRZF, sigma2)
            #SR_AIGC += SumRate(H.T, PRZF_AIGC, sigma2)

        sumratedata_rzf[s] = SR_rzf / realization
        #sumratedata_AIGC[s] = SR_AIGC / realization


    plt.figure()
    # plt.plot(snr, sumratedata_MO, 'g-.', linewidth=1.5)
    plt.plot(snr, sumratedata_rzf, 'b-', linewidth=1.5)
    #plt.plot(snr, sumratedata_LSR, 'k-p', linewidth=1.5)
    plt.legend(['rzf'], loc='upper left')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Spectrum Efficiency (bps/Hz)')
    plt.grid(True)
    plt.box(True)
    plt.show()







