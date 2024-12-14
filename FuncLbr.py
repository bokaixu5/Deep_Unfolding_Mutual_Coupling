# -* coding: utf-8 -*-
import numpy as np
import torch
from Global_Vars import *
import math
import torch.nn as nn
import h5py
from scipy.spatial.distance import pdist, squareform
from scipy.io import loadmat
from scipy.linalg import sqrtm
from torch.utils.data import Dataset, DataLoader
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def Ga_e2(r, rhat):
    f = 10e9  # Frequency of operation
    c = 299792458  # Light speed in vacuum
    mu = 4 * np.pi * 1E-7  # Vacuum permeability
    epsilon = 1 / (c ** 2 * mu)  # Vacuum permittivity
    lmbda = c / f  # Wavelength
    k = 2 * np.pi / lmbda  # Wavenumber
    term1 = (np.linalg.norm(r - rhat) ** 2 - (r[2] - rhat[2]) ** 2) / np.linalg.norm(r - rhat) ** 2
    term2 = 1j * (np.linalg.norm(r - rhat) ** 2 - 3 * (r[2] - rhat[2]) ** 2) / (np.linalg.norm(r - rhat) ** 3 * k)
    term3 = (np.linalg.norm(r - rhat) ** 2 - 3 * (r[2] - rhat[2]) ** 2) / (np.linalg.norm(r - rhat) ** 4 * k ** 2)
    return (term1 - term2 - term3) * np.exp(-1j * k * np.linalg.norm(r - rhat)) / (4 * np.pi * np.linalg.norm(r - rhat))
def calculateY_tt(N, f, mu, xyz_dma, epsilon):
    k = 2 * np.pi * f * np.sqrt(epsilon * mu)
    Y_tt = np.zeros((N, N), dtype=np.complex_)
    for n1 in range(N):
        for n2 in range(N):
            if n1 != n2:
                Y_tt[n1, n2] = 1j * 2 * 2 * np.pi * f * epsilon * Ga_e2(xyz_dma[n1, :], xyz_dma[n2, :])
            else:
                Y_tt[n1, n2] = k * 2 * np.pi * f * epsilon / (3 * np.pi)

    return Y_tt

def Topologies_DMA(site_xyz, N, Lmu, wvg_spacing, elem_spacing, S_mu, a, b, Plot_topology):
    # Total number of antennas
    L = Lmu * N
    # Pre-allocating
    #print(site_xyz.shape[1])
    ant_xyz = np.zeros((L , 3))
    rf_xyz = np.zeros((N , 3))

    for ksite in range(1):
        # Coordinates of RF chains
        z_rf = np.arange(N) * wvg_spacing + a / 2
        y_rf = b / 2
        x_rf = 0

        # Coordinates of DMA elements
        z_dma = z_rf
        y_dma = b
        x_dma = np.arange(1, Lmu + 1) * elem_spacing
        x_dma = x_dma - np.mean(x_dma) + S_mu / 2

        # Store coordinates for antennas in site ksite
        Xant, Yant, Zant = np.meshgrid(x_dma + site_xyz[0],
                                       y_dma + site_xyz[1],
                                       z_dma + site_xyz[2])
        ant_xyz[(L * (ksite)):(L * (ksite + 1)), :] = np.column_stack([Xant.flatten(), Yant.flatten(), Zant.flatten()])

        # Store coordinates for RF chains in site ksite
        Xrf, Yrf, Zrf = np.meshgrid(x_rf + site_xyz[0],
                                    y_rf + site_xyz[1],
                                    z_rf + site_xyz[2])
        rf_xyz[(N * (ksite)):(N * (ksite + 1)), :] = np.column_stack([Xrf.flatten(), Yrf.flatten(), Zrf.flatten()])

    # Plotting deployment
    if not Plot_topology:
        return ant_xyz, rf_xyz

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    antplt = ax.scatter3D(ant_xyz[:, 0], ant_xyz[:, 1], ant_xyz[:, 2])
    rfplt = ax.scatter3D(rf_xyz[:, 0], rf_xyz[:, 1], rf_xyz[:, 2])

    xtot = np.zeros((4 * N, 4))
    ytot = np.zeros((0, 4))
    ztot = np.zeros((0, 4))
    xyz_aux = rf_xyz

    for idn in range(N):
        zt = np.array([[0, a, a, 0],
                       [0, a, a, 0],
                       [0, 0, 0, 0],
                       [a, a, a, a]]) + (xyz_aux[1, 2] - xyz_aux[0, 2]) * idn
        yt = np.array([[0, 0, 0, 0],
                       [b, b, b, b],
                       [0, b, b, 0],
                       [0, b, b, 0]])
        xt = np.array([[0, 0, S_mu, S_mu],
                       [0, 0, S_mu, S_mu],
                       [0, 0, S_mu, S_mu],
                       [0, 0, S_mu, S_mu]]) + (xyz_aux[1, 0] - xyz_aux[0, 0]) * idn

        xtot[4 * idn:4 * (idn + 1), :] = xt + site_xyz[0, 0]
        ytot = np.vstack([ytot, yt + site_xyz[0, 1]])
        ztot = np.vstack([ztot, zt + site_xyz[0, 2]])

    ax.plot_surface(xtot.T, ytot.T, ztot.T, alpha=0.2)

    ax.set_xlim([0, 2 * S_mu])
    ax.set_ylim([0, 0.2])
    ax.set_zlim([site_xyz[0, 2], site_xyz[0, 2] + (N + 1) * wvg_spacing])
    ax.legend([antplt, rfplt], ['DMA elements', 'RF chains'])
    ax.set_title('Deployment DMA')
    ax.view_init(135, 37)

    plt.show()

    return ant_xyz, rf_xyz

def GenChannel(channel_type, wavelength, ant_xyz, user_xyz):
    L = ant_xyz.shape[0] # Number of DMA elements
    M = user_xyz.shape[0] # Number of users
    k = 2 * np.pi / wavelength

    # Rayleigh fading
    if channel_type:

        # Distances between users and DMA elements
        D = np.zeros((M, L))
        for idn in range(L):
            for idm in range(M):
                D[idm, idn] = np.linalg.norm(ant_xyz[idn, :] - user_xyz[idm, :])

        # Pathloss
        PL = (wavelength / (4 * np.pi * D)) ** 2

        # Uncorrelated Complex Gaussian realizations
        #Yrs_uncorr = (np.random.randn(M, L) + 1j * np.random.randn(M, L)) * np.sqrt(PL / 2)
        Yrs_uncorr = np.random.randn(M, L) + 1j * np.random.randn(M, L)

        # Compute correlation coefficient.
        dZ = squareform(pdist(ant_xyz[:, 2].reshape(-1, 1)))
        dR = squareform(pdist(ant_xyz))
        #print(ant_xyz)
        Sigma = 3 / 2 * (
                (1 + (-k ** 2 * dZ ** 2 - 1) / (dR ** 2 * k ** 2) +
                 3 * dZ ** 2 / (dR ** 4 * k ** 2)) * np.sin(k * dR) / (k * dR) +
                np.cos(k * dR) * (1 / (k * dR) - 3 * dZ ** 2 / (dR ** 3 * k)) / (k * dR))
        #print(Sigma)
        Sigma[np.isnan(Sigma)] = 1
        sq_Sigma = np.real(sqrtm(Sigma))  # Real operator due to imaginary part
        # being a product of quantization
        #print(Sigma)
        Y_rs = Yrs_uncorr @ sq_Sigma

    # LoS channel
    else:

        # Distances between users and DMA elements
        D = np.zeros((M, L))
        dz = np.zeros((M, L))
        for idn in range(L):
            for idm in range(M):
                D[idm, idn] = np.linalg.norm(ant_xyz[idn, :] - user_xyz[idm, :])
                dz[idm, idn] = np.abs(ant_xyz[idn, 2] - ant_xyz[idm, 2])

        # Polar angle
        theta = np.pi / 2 - np.arcsin(dz / D)

        # Pathloss
        PL = (wavelength / (4 * np.pi * D)) ** 2 * (3 / 2 * np.sin(theta) ** 2) * (6 / 2 * np.sin(theta) ** 2)
        Y_rs = np.sqrt(PL) * np.exp(-1j * k * D)

    return Y_rs






def DMA_admittance(f, a, b, l, S_mu, xyz_dma, xyz_rf, mu, epsilon):
    # Physical constants
    k = 2 * np.pi * f * np.sqrt(epsilon * mu)

    kx = np.sqrt(k ** 2 - (np.pi / a) ** 2)  # only for TE_10

    # Y_tt calculation
    z_rf = a / 2  # Position of RF chain in the width of the waveguide (0 <= z_rf <= a)

    y_tt = -l ** 2 * 2j * kx * np.sin(np.pi / a * z_rf) ** 2 * np.cos(kx * S_mu) / (
                a * b * 2 * np.pi * f * mu * np.sin(kx * S_mu))

    Y_tt = np.diag(y_tt * np.ones(xyz_rf.shape[0]))

    # Y_st calculation
    #Y_st = np.zeros((xyz_dma.shape[0], xyz_rf.shape[0]))
    Y_st = np.zeros((xyz_dma.shape[0], xyz_rf.shape[0]), dtype=np.complex128)
    Gw_e2 = lambda r, rhat: -kx * np.sin(np.pi / a * rhat[2]) * np.sin(np.pi / a * r[2]) * \
                            (np.cos(kx * (rhat[0] + r[0] - S_mu)) + \
                             np.cos(kx * (S_mu - np.abs(rhat[0] - r[0])))) / (a * b * k ** 2 * np.sin(kx * S_mu))

    N_ant_wg = xyz_dma.shape[0] // xyz_rf.shape[0]

    temp = np.zeros(xyz_dma.shape[0])
    temp[::N_ant_wg] = xyz_rf[:, 0]
    #print(xyz_dma[:, 0].shape)
    #print(np.convolve(np.ones(N_ant_wg), temp, mode="full").shape)
    x_ant_norm = xyz_dma[:, 0] - np.convolve(np.ones(N_ant_wg), temp, mode="same")

    for row in range(N_ant_wg):
        y_val = l ** 2 * 1j * 2 * np.pi * f * epsilon * Gw_e2([x_ant_norm[row], 0, z_rf], [0, 0, z_rf])
        for k_rf in range(xyz_rf.shape[0]):
            Y_st[row + k_rf * N_ant_wg, k_rf] = y_val

    # Y_ss calculation
    Ga_e2 = lambda r, rhat: ((np.linalg.norm(r - rhat) ** 2 - (r[2] - rhat[2]) ** 2) / np.linalg.norm(r - rhat) ** 2 - \
                             1j * (np.linalg.norm(r - rhat) ** 2 - 3 * (r[2] - rhat[2]) ** 2) / (
                                         np.linalg.norm(r - rhat) ** 3 * k) - \
                             (np.linalg.norm(r - rhat) ** 2 - 3 * (r[2] - rhat[2]) ** 2) / (
                                         np.linalg.norm(r - rhat) ** 4 * k ** 2)) * \
                            np.exp(-1j * k * np.linalg.norm(r - rhat)) / (4 * np.pi * np.linalg.norm(r - rhat))

    Y_ss = np.zeros((xyz_dma.shape[0], xyz_dma.shape[0]), dtype=np.complex128)

    for row in range(xyz_dma.shape[0]):
        for col in range(xyz_dma.shape[0]):
            if row == col:
                Y_ss[row, col] = l ** 2 * k * 2 * np.pi * f * epsilon / (3 * np.pi) + \
                                 1j * 2 * np.pi * f * epsilon * Gw_e2([xyz_dma[row, 0], 0, z_rf],
                                                                      [xyz_dma[col, 0], 0, z_rf])
            elif (np.abs(xyz_dma[row, 0] - xyz_dma[col, 0]) >= S_mu or \
                  np.abs(xyz_dma[row, 2] - xyz_dma[col, 2]) >= b / 2):
                Y_ss[row, col] = l ** 2 * 1j * 2 * np.pi * f * epsilon * \
                                 2 * Ga_e2(xyz_dma[row, :], xyz_dma[col, :])
            else:
                Y_ss[row, col] = l ** 2 * 1j * 2 * np.pi * f * epsilon * \
                                 (2 * Ga_e2(xyz_dma[row, :], xyz_dma[col, :]) + \
                                  Gw_e2([xyz_dma[row, 0], 0, z_rf], [xyz_dma[col, 0], 0, z_rf]))

    return Y_tt, Y_st, Y_ss

def Coupling_Dipoles(f, l, xyz_user, mu, epsilon):
    # Physical constants
    k = 2 * np.pi * f * np.sqrt(epsilon * mu)

    # Y_rr calculation
    Ga_e2 = lambda r, rhat: ((np.linalg.norm(r - rhat)**2 - (r[2] - rhat[2])**2) / np.linalg.norm(r - rhat)**2 - \
                            1j * (np.linalg.norm(r - rhat)**2 - 3 * (r[2] - rhat[2])**2) / (np.linalg.norm(r - rhat)**3 * k) - \
                            (np.linalg.norm(r - rhat)**2 - 3 * (r[2] - rhat[2])**2) / (np.linalg.norm(r - rhat)**4 * k**2)) * \
                            np.exp(-1j * k * np.linalg.norm(r - rhat)) / (4 * np.pi * np.linalg.norm(r - rhat))

    Y_rr = np.zeros((xyz_user.shape[0], xyz_user.shape[0]), dtype=np.complex128)

    for row in range(xyz_user.shape[0]):
        for col in range(xyz_user.shape[0]):
            if row == col:
                Y_rr[row, col] = l**2 * k * 2 * np.pi * f * epsilon / (6 * np.pi)
            else:
                Y_rr[row, col] = l**2 * 1j * 2 * np.pi * f * epsilon * \
                    Ga_e2(xyz_user[row, :], xyz_user[col, :])

    return Y_rr

def calculateY_q(Gamma, Y_p):
    N = Gamma.shape[1]  # Determine the size of N, assuming Gamma is an N x M matrix
    I_N = np.eye(N)  # N x N identity matrix
    Y_q = np.linalg.inv(I_N - np.conjugate(Gamma).T@ Gamma) @ Y_p  # Calculate Y_q
    return Y_q

def ZF(Pmax_t, Heq_fd, sigma2_x, Y_tt):
    N_user, Nt = Heq_fd.shape
    #print(Heq_fd)
    H_fd_dagger = np.linalg.pinv(Heq_fd)  # 使用 numpy.linalg.pinv 求伪逆
    numerator = np.sqrt(Pmax_t) * H_fd_dagger  # 分子部分
    denominator = np.sqrt(np.trace(sigma2_x / 2 * np.real(np.conjugate(H_fd_dagger).T @ Y_tt @ H_fd_dagger)))  # 分母部分
    B_fd = numerator / denominator  # 结果
    return B_fd


def SE_calculation(Heq, B, sigma2_n, sigma2_x):
    K, M = Heq.shape
    c = np.zeros(K)

    for idx1 in range(K):
        ds = np.abs(np.dot(Heq[idx1, :], B[:, idx1])) ** 2
        inter = 0

        for idx2 in range(K):
            if idx2 != idx1:
                inter += np.abs(np.dot(Heq[idx1, :], B[:, idx2])) ** 2

        sinr_k = ds / (sigma2_n / sigma2_x + inter)
        c[idx1] = np.log2(1 + sinr_k)

    C = np.sum(c)
    return C
def array_dimension(Nt):
    '''
    :param Nt: number of total antennas
    :return: the configuration of UPA that minimizes beam squint effect
    '''
    n = math.ceil(Nt ** 0.5)
    for i in range(n+1,1,-1):
        if Nt%i==0:
            Nth = i
            Ntv = int(Nt/i)
            break
    return Nth, Ntv

def pulase_filter(t, Ts, beta):
    '''
    Raised cosine filter
    :param t: time slot
    :param Ts: sampling frequency
    :param beta: roll-off factor
    :return: filtered value
    '''
    if abs(t-Ts/2/beta)/abs(t) <1e-4 or abs(t+Ts/2/beta)/abs(t)<1e-4:
        p = np.pi/4 * np.sinc(1/2/beta)
    else:
        p = np.sinc(t/Ts) * np.cos(np.pi*beta*t/Ts)/(1-(2*beta*t/Ts)**2)
    return p


def array_response(Nh,Nv, Angle_H, Angle_V, f,fc, array_type = 'UPA', AtDs=0.5):
    '''
    This function defines a steering vector for a Nh*Nv uniform planar array (UPA).
    See paper 'Dynamic Hybrid Beamforming With Low-Resolution PSs for Wideband mmWave MIMO-OFDM Systems'
    :param Nh: number of antennas in horizontal direction
    :param Nv: number of antennas in vertical direction
    :param fc: carrier frequency
    :param f: actual frequency
    :param AtDs: normalized antenna spacing distance, set to 0.5 by default
    :return: steering a vector at frequency f with azimuth and elevation angles
    '''
    N = int(Nh*Nv)
    Np = Angle_H.shape[0]
    AtDs_h = AtDs
    AtDs_v = AtDs
    array_matrix = np.zeros([N,Np], dtype=np.complex_)
    if array_type == 'ULA':
        spatial_h = np.sin(Angle_H)
        factor_h = np.array(range(N))
        for n in range(Np):
            array_matrix[:, n] = 1/np.sqrt(N)*np.exp(1j*2*np.pi * AtDs_h* factor_h*f/fc*spatial_h[n])

    else:
        # Nh, Nv = array_dimension(N)
        spatial_h = np.sin(Angle_H) * np.sin(Angle_V)
        spatial_v = np.cos(Angle_V)
        factor_h = np.array(range(Nh))
        factor_v = np.array(range(Nv))
        for n in range(Np):
            steering_vector_h = 1/np.sqrt(Nh) * np.exp(1j*2*np.pi * AtDs_h* factor_h*f/fc*spatial_h[n])
            steering_vector_v = 1/np.sqrt(Nv) * np.exp(1j*2*np.pi* AtDs_v * factor_v*f/fc*spatial_v[n])
            array_matrix[:,n] = np.kron(steering_vector_h, steering_vector_v)
    ccc = 1
    return array_matrix


def OMP(Fopt, At):
    Frf = []
    # print(np.shape(Frf))
    Fbb = np.zeros([Nrf, Ns], dtype='complex_')
    Fres = Fopt
    for n in range(Nrf):
        Pu = np.matmul(At.conj().T, Fres)
        best_idx = np.argmax(np.sum(np.abs(Pu), 1))
        best_at = np.asmatrix(np.sqrt(Nt)*At[:, best_idx]).T

        if n == 0:
            Frf = best_at
            Frf_pinv = np.linalg.pinv(np.asmatrix(Frf))
        else:
            Frf = np.append(Frf, best_at, axis=1)
            Frf_pinv = np.linalg.pinv(Frf)

        Fbb = np.matmul(Frf_pinv, Fopt)
        # print(np.shape(Frf))
        # print(np.shape(Fbb))
        Fres = (Fopt - np.matmul(np.asmatrix(Frf), Fbb)) / (np.linalg.norm(Fopt - np.matmul(Frf, Fbb), 'fro'))
    return Frf, Fbb
def generate_user_in_line_multipath(Nt, N_user):
    # Channel of FD mMIMO
    # Parameters
    f = 10e9  # Frequency of operation
    c = 299792458  # Light speed in vacuum
    mu = 4 * np.pi * 1E-7  # Vacuum permeability
    epsilon = 1 / (c ** 2 * mu)  # Vacuum permittivity
    lmbda = c / f  # Wavelength
    k = 2 * np.pi / lmbda  # Wavenumber
    a = 0.73 * lmbda  # Width of waveguides (only TE_10 mode)
    b = 0.17 * lmbda  # Height of waveguides (only TE_10 mode)
    channel_type = 1  # Type of channel: 0 -> LoS, 1 -> Rayleigh
    N = 6  # Number of RF chains / waveguides
    Lmu = 20  # Number of elements per waveguide
    N_1 = 6  # Number of RF chains / waveguides
    Lmu_1 = 20  # Number of elements per waveguide
    NN_1 = N_1 * Lmu_1
    wvg_spacing = lmbda  # Spacing between waveguides
    elem_spacing = lmbda  # Spacing between the elements
    l = 1  # Length of dipoles -> just normalization
    M = 6  # Number of static users
    Plot_topology = 0  # Boolean to plot the chosen setup

    Y_s = np.diag(1j * np.random.randn(N * Lmu))  # Load admittances of DMA element

    Y_intrinsic_source = 35.3387  # Intrinsic impedance of source matched to waveguide of width a and height b
    sigma2_x = 1

    # DMA and users coordinates
    site_xyz = np.array([0, 0, 10])  # [x y z] coordinates of bottom right corner of DMA
    S_mu = (Lmu + 1) * elem_spacing  # Length of waveguides
    S_mu_1 = (Lmu_1 + 1) * elem_spacing

    # Coordinates of DMA elements and RF chains
    ant_xyz_fd, _ = Topologies_DMA(site_xyz, N_1, Lmu_1, wvg_spacing, elem_spacing, S_mu_1, a, b, Plot_topology)
    ant_xyz, rf_xyz = Topologies_DMA(site_xyz, N, Lmu, wvg_spacing, elem_spacing, S_mu, a, b, Plot_topology)

    # Users positions (In this example, they are set randomly)
    x_lim = [-20, 20]
    y_lim = [20, 60]

    x = x_lim[0] + (x_lim[1] - x_lim[0]) * np.random.rand(M, 1)
    y = y_lim[0] + (y_lim[1] - y_lim[0]) * np.random.rand(M, 1)
    ones = np.ones((M, 1)) * 1.5

    user_xyz = np.concatenate((x, y, ones), axis=1)

    # user_xyz = np.hstack((x_lim[0] + (x_lim[1] - x_lim[0]) * x1,
    #                       y_lim[0] + (y_lim[1] - y_lim[0]) * y1,
    #                       1.5 * np.ones((M, 1))))

    # Calculation of Admittances
    Y_tt, Y_st, Y_ss = DMA_admittance(f, a, b, l, S_mu, ant_xyz, rf_xyz, mu, epsilon)
    Y_rr = Coupling_Dipoles(f, l, user_xyz, mu, epsilon)
    Y_r = np.conj(Y_rr.T) * np.eye(M)

    # Calculation of Y_rs (Wireless channel)
    Y_rs_fd = GenChannel(channel_type, lmbda, ant_xyz_fd, user_xyz)
    Y_rs = GenChannel(channel_type, lmbda, ant_xyz, user_xyz)

    # Equivalent channel according to Eq. (60)
    Heq = np.linalg.inv(Y_r + Y_rr) @ (Y_rs @ np.linalg.inv(Y_s + Y_ss) @ Y_st)
    # print(type(Y_st.T))
    # print(type(Y_s + Y_ss))
    # Computing received, transmitted, and supplied power
    Y_p = Y_tt - (Y_st.T @ np.linalg.inv(Y_s + Y_ss)) @ Y_st
    Y_in = np.eye(N) * Y_p
    Gamma = (Y_in - np.eye(N) * Y_intrinsic_source) @ np.linalg.inv(Y_in + np.eye(N) * Y_intrinsic_source)

    # Channel of FD mMIMO
    tilde_Yr = np.sqrt(np.real(Y_r) / 2) @ np.linalg.inv(Y_r + Y_rr)
    Y_rt = Y_rs_fd
    # print(-tilde_Yr)
    Heq_fd = -tilde_Yr @ Y_rt



    return Heq_fd




def masking_dyn(H, sub_connected=True, sub_structure_type="fixed"):
    batch_size, Nrf, Nt = H.shape
    N = 2 * Nt * Nrf
    bin_mask_mat = np.ones([batch_size, Nt, Nrf], dtype='int_') + 1j * np.ones([batch_size, Nt, Nrf], dtype='int_')
    bin_mask_vec_real = np.zeros([batch_size, N])

    for ii in range(batch_size):
        if sub_connected:
            if sub_structure_type == "fixed":
                bin_mask_mat[ii, Nt // 2:Nt, 0] = 0
                bin_mask_mat[ii, 0:Nt // 2, 1] = 0
            else:  # dynamic
                # choose best channel
                #power_H = np.zeros([K], dtype='float')
                #for k in range(K):
                #    power_H[k] = np.linalg.norm(H[ii,:, :, k])

                #k_max = np.argmax(power_H)
                Hmax = H[ii, :, :]
                # print(Hmax)
                D = np.abs(Hmax.T)
                # print(np.shape(D))
                bin_mask_mat_k = np.ones([Nt, Nrf], dtype='int_') + 1j * np.ones([Nt, Nrf], dtype='int_')
                for m in range(Nt // Nrf):
                    for n in range(Nrf):
                        m_min = np.argmin(D[:, n], axis=0)
                        bin_mask_mat_k[m_min, n] = 0
                        D[m_min, :] = 1000
                # print(bin_mask_mat_k)

                bin_mask_mat[ii, :, :] = bin_mask_mat_k

            bin_mask_vec = bin_mask_mat[ii, :, :].flatten('F')
            bin_mask_vec_real[ii, :] = np.concatenate((bin_mask_vec.real, bin_mask_vec.imag),
                                                      axis=0)  # convert to real values
        # print(bin_mask_mat[ii, :, :])

        else:
            bin_mask_vec = bin_mask_mat[ii, :, :].flatten('F')
            bin_mask_vec_real[ii, :] = np.concatenate((bin_mask_vec.real, bin_mask_vec.imag),
                                                      axis=0)  # convert to real values
    return bin_mask_vec_real


def normalize(FRF,sub_connected=True, sub_structure_type="fixed"):
    Nt, Nrf = FRF.shape
    if sub_connected:
        if sub_structure_type == "fixed":
            FRF[0:Nt // 2, 0] = FRF[0:Nt // 2, 0] / np.abs(FRF[0:Nt // 2, 0])
            FRF[Nt // 2:, 1] = FRF[Nt // 2:, 1] / np.abs(FRF[Nt // 2:, 1])

        else:
            for tt in range(Nt):
                for nn in range(Nrf):
                    if np.abs(FRF[tt, nn]) > 0.0001:
                        FRF[tt, nn] = FRF[tt, nn] / np.abs(FRF[tt, nn])
    else:
        FRF = FRF / np.abs(FRF)
    ccc=1
    return FRF
def RZF(H, pow):
    sigma2_x=1
    f = 10e9  # Frequency of operation
    c = 299792458  # Light speed in vacuum
    mu = 4 * np.pi * 1E-7  # Vacuum permeability
    epsilon = 1 / (c ** 2 * mu)  # Vacuum permittivity
    lmbda = c / f  # Wavelength
    k = 2 * np.pi / lmbda  # Wavenumber
    a = 0.73 * lmbda  # Width of waveguides (only TE_10 mode)
    b = 0.17 * lmbda  # Height of waveguides (only TE_10 mode)
    channel_type = 1  # Type of channel: 0 -> LoS, 1 -> Rayleigh
    N = 6  # Number of RF chains / waveguides
    Lmu = 20  # Number of elements per waveguide


    wvg_spacing = lmbda  # Spacing between waveguides
    elem_spacing = lmbda  # Spacing between the elements
    l = 1  # Length of dipoles -> just normalization
    M = 6  # Number of static users
    Plot_topology = 0  # Boolean to plot the chosen setup



    # DMA and users coordinates
    site_xyz = np.array([0, 0, 10])  # [x y z] coordinates of bottom right corner of DMA
    S_mu = (Lmu + 1) * elem_spacing  # Length of waveguides

    ant_xyz, rf_xyz = Topologies_DMA(site_xyz, N, Lmu, wvg_spacing, elem_spacing, S_mu, a, b, Plot_topology)
    Y_tt = calculateY_tt(N*Lmu, f, mu, ant_xyz, epsilon)
    H_fd_dagger = np.linalg.pinv(H)  # 使用 numpy.linalg.pinv 求伪逆

    numerator = np.sqrt(pow) * H_fd_dagger  # 分子部分
    denominator = np.sqrt(np.trace(sigma2_x / 2 * np.real(np.conjugate(H_fd_dagger).T @ Y_tt @ H_fd_dagger)))  # 分母部分
    B_fd = numerator / denominator  # 结果
    # K, M = H.shape
    # pre = np.dot(np.conj(H).T, np.linalg.inv(np.dot(H, np.conj(H).T) + M/pow * np.eye(K)))
    # P = np.sqrt(pow/np.trace(np.dot(pre, np.conj(pre).T))) * pre
    return B_fd
def re_Y():


    # 读取MAT文件并存储在变量data中
    Y_tt_fd = loadmat('Y_tt_fd.mat')
    return Y_tt_fd['Y_tt_fd']
    # # Parameters
    # f = 10e9  # Frequency of operation
    # c = 299792458  # Light speed in vacuum
    # mu = 4 * np.pi * 1E-7  # Vacuum permeability
    # epsilon = 1 / (c ** 2 * mu)  # Vacuum permittivity
    # lmbda = c / f  # Wavelength
    # k = 2 * np.pi / lmbda  # Wavenumber
    # a = 0.73 * lmbda  # Width of waveguides (only TE_10 mode)
    # b = 0.17 * lmbda  # Height of waveguides (only TE_10 mode)
    # channel_type = 1  # Type of channel: 0 -> LoS, 1 -> Rayleigh
    # N = 6  # Number of RF chains / waveguides
    # Lmu = 20  # Number of elements per waveguide
    # N_1 = 6  # Number of RF chains / waveguides
    # Lmu_1 = 20  # Number of elements per waveguide
    # NN_1 = N_1 * Lmu_1
    # wvg_spacing = lmbda  # Spacing between waveguides
    # elem_spacing = lmbda  # Spacing between the elements
    # l = 1  # Length of dipoles -> just normalization
    # M = 3  # Number of static users
    # Plot_topology = 0  # Boolean to plot the chosen setup
    #
    #
    #
    # # DMA and users coordinates
    # site_xyz = np.array([0, 0, 10])  # [x y z] coordinates of bottom right corner of DMA
    # S_mu = (Lmu + 1) * elem_spacing  # Length of waveguides
    #
    #
    # # Coordinates of DMA elements and RF chains
    #
    # ant_xyz, rf_xyz = Topologies_DMA(site_xyz, N, Lmu, wvg_spacing, elem_spacing, S_mu, a, b, Plot_topology)
    #
    #
    #
    #
    #
    # # Calculation of Admittances
    # Y_tt = calculateY_tt(N*Lmu, f, mu, ant_xyz, epsilon)
    #return Y_tt
def gen_data_wideband(Nt, Nr, Nrf, Ns, batch_size=1,
                      Sub_Connected=False,
                      Sub_Structure_Type='fixed',
                      Pulse_Filter=False,
                      fc=fc, Ncl=Ncl, Nray=Nray, bandwidth=Bandwidth):
    # data to get
    N = 2 * Nt * Nrf  # true for Nrf = Ns
    batch_X = np.zeros([batch_size, N], dtype='float32')  # only use when employing supervised learning

    #batch_z = np.zeros([batch_size, N, K], dtype='float32')  # input to DNN for training
    batch_z = np.zeros([batch_size, N], dtype='float32')  # input to DNN for training
    batch_B = np.zeros([batch_size, N, N], dtype='float32')  # input to DNN to compute loss function
    batch_Bz = np.zeros([batch_size, N], dtype='float32')  # input to DNN for training
    batch_BB = np.zeros([batch_size, N, N], dtype='float32')  # input to DNN for training
    batch_AA = np.zeros([batch_size, N, N], dtype='float32')  # input to DNN for training

    batch_H = np.zeros([batch_size, Nr, Nt], dtype='complex_')  # use to save testing data, used latter in Matlab
    batch_Fopt = np.zeros([batch_size, Nt, Ns], dtype='complex_')  # use to save testing data, used latter in Matlab
    #batch_Wopt = np.zeros([batch_size, Nr, Ns, K], dtype='complex_')  # use to save testing data, used latter in Matlab
    batch_Fbb = np.zeros([batch_size, Nrf, Ns], dtype='complex_')  # use to save testing data, used latter in Matlab
    #batch_At = np.zeros([batch_size, Nt, Ncl*Nray, K], dtype='complex_')  # use to save testing data, used latter in Matlab

    for ii in range(batch_size):
        #if init_scheme == 0:
        FRF = np.exp(1j * np.random.uniform(0, 2 * np.pi, [Nt, Nrf]))  # frequency-flat
            # FRF = normalize(FRF, Nt, Nrf, sub_connected=Sub_Connected, sub_structure_type=Sub_Structure_Type)
        FRF_vec = FRF.flatten('F')
        batch_X[ii, :] = np.concatenate((FRF_vec.real, FRF_vec.imag), axis=0)


        #H_ii, At_ii = channel_model(Nt, Nr, Pulse_Filter=Pulse_Filter, fc=fc, Ncl=Ncl, Nray=Nray, bandwidth=bandwidth)
        H_ii=generate_user_in_line_multipath(Nt, Nr)
        batch_H[ii, :, :] = H_ii
        Pow=1
        #batch_At[ii, :, :, :] = At_ii
        Fopt = RZF(H_ii, Pow)
        #At = At_ii[:, :, k]
        # U, S, VH = np.linalg.svd(H_ii[:, :])
        # V = VH.T.conj()
        # Fopt = V[:, 0:Ns]  # np.sqrt(Ns) *
        #Fopt = torch.matmul(torch.conj(H_ii).T, torch.linalg.inv(torch.matmul(H_ii, torch.conj(H_ii).T)))
        #Wopt = U[:, 0:Ns]

        ## construct training data
        ztilde = Fopt.flatten('F')
        z = np.concatenate((ztilde.real, ztilde.imag), axis=0)  # convert to real values
        # z_vector = np.matrix(z)
        #if init_scheme == 0:  # random FRF, FBB = LS solution
        FBB = np.matmul(np.linalg.pinv(FRF), Fopt)
        Y_tt = re_Y()
        P = 1
        sigma2_x = 1
        numerator = np.sqrt(P) * FBB  # 分子部分
        print(type(Y_tt))
        denominator = np.sqrt(
            np.trace(sigma2_x / 2 * np.real(np.conjugate(FRF @ FBB).T @ Y_tt @ (FRF @ FBB))))  # 分母部分
        FBB = numerator / denominator  # 结果


        Btilde = np.kron(FBB.T, np.identity(Nt))
        B1 = np.concatenate((Btilde.real, -Btilde.imag), axis=1)
        B2 = np.concatenate((Btilde.imag, Btilde.real), axis=1)
        B = np.concatenate((B1, B2), axis=0)
        # print(np.shape(B))

        # new for array response
        #AtH = At.conj().T
        #Atilde = np.kron(np.identity(Nrf), AtH)
        #A1 = np.concatenate((Atilde.real, -Atilde.imag), axis=1)
        #A2 = np.concatenate((Atilde.imag, Atilde.real), axis=1)
        #A = np.concatenate((A1, A2), axis=0)
        # print(np.shape(A))

        # Assign data to the ii-th batch
        # err = z_vector.dot(B) -np.matmul(B.T, z)
        # err1 = np.matmul(z_vector,B) - z_vector.dot(B)

        batch_Bz[ii, :] = np.matmul(B.T, z)
        batch_BB[ii, :, :] = np.matmul(B.T, B)
        batch_z[ii, :] = z
        batch_B[ii, :, :] = B.T
        batch_Fopt[ii, :, :] = Fopt
        #batch_Wopt[ii, :, :, k] = Wopt
        batch_Fbb[ii, :, :] = FBB
        #batch_AA[ii, :, :, k] = np.matmul(A.T, A)



    # dis_sum = 0
    # for k in range(K):
    #     med = np.matmul(np.expand_dims(batch_X,1), batch_B[:,:,:, k]).squeeze()
    #     diff = batch_z[ :, :, k] - med
    #     dis_sum += np.sum(np.linalg.norm(diff, 'fro'))
    # print(f'{ii} error:{dis_sum}')
    # ccc = 1
    return batch_Bz, batch_BB, batch_X, batch_z, batch_B, batch_H, batch_Fopt,batch_Fbb


def gen_data_large(Nt, Nr, Nrf, Ns, Num_batch,batch_size=1, fc=fc, Ncl=Ncl, Nray=Nray, bandwidth=Bandwidth, Pulse_Filter=False, data='taining'):
    def append_data(data_set, num_data, new_data):
        dims = list(data_set.shape)
        num_sp = dims[0] + num_data
        dims_new = list(dims[1:])
        dims_new.insert(0, num_sp)
        data_set.resize(tuple(dims_new))
        data_set[dims[0]:num_sp] = new_data
        return data_set
    # Channel setup
    channel_type = 'geometry'
    # data to get
    data_name = train_data_name
    if data == 'testing':
        data_name = test_data_name
    data_path = dataset_file + data_name
    hf = h5py.File(data_path, 'a')
    batch_Bz_set = hf.get('batch_Bz')
    batch_BB_set = hf.get('batch_BB')
    batch_X_set = hf.get('batch_X')
    batch_Z_set = hf.get('batch_Z')
    batch_B_set = hf.get('batch_B')
    batch_H_real_set = hf.get('batch_H_real')
    batch_H_imag_set = hf.get('batch_H_imag')
    batch_Fopt_real_set = hf.get('batch_Fopt_real')
    batch_Fopt_imag_set = hf.get('batch_Fopt_imag')
    if data == 'testing':
        #batch_Wopt_real_set = hf.get('batch_Wopt_real')
        #batch_Wopt_imag_set = hf.get('batch_Wopt_imag')
        batch_Fbb_real_set = hf.get('batch_Fbb_real')
        batch_Fbb_imag_set = hf.get('batch_Fbb_imag')
        #batch_At_real_set = hf.get('batch_At_real')
        #batch_At_imag_set = hf.get('batch_At_imag')


    # data to get
    N = 2 * Nt * Nrf  # true for Nrf = Ns
    batch_X = np.zeros([batch_size, N], dtype='float32')  # only use when employing supervised learning
    batch_z = np.zeros([batch_size, N], dtype='float32')  # input to DNN for training
    batch_B = np.zeros([batch_size, N, N], dtype='float32')  # input to DNN to compute loss function
    batch_Bz = np.zeros([batch_size, N], dtype='float32')  # input to DNN for training
    batch_BB = np.zeros([batch_size, N, N], dtype='float32')  # input to DNN for training
    #batch_AA = np.zeros([batch_size, N, N], dtype='float32')  # input to DNN for training

    batch_H = np.zeros([batch_size, Nr, Nt], dtype='complex_')  # use to save testing data, used latter in Matlab
    batch_Fopt = np.zeros([batch_size, Nt, Ns], dtype='complex_')  # use to save testing data, used latter in Matlab
    #batch_Wopt = np.zeros([batch_size, Nr, Ns, K], dtype='complex_')  # use to save testing data, used latter in Matlab
    batch_Fbb = np.zeros([batch_size, Nrf, Ns], dtype='complex_')  # use to save testing data, used latter in Matlab
    #batch_At = np.zeros([batch_size, Nt, Ncl*Nray, K], dtype='complex_')  # use to save testing data, used latter in Matlab


    for n in range(Num_batch):
        print(f'Generating {n}th batch data', flush=True)
        for ii in range(batch_size):
            #if init_scheme == 0:
            FRF = np.exp(1j * np.random.uniform(0, 2 * np.pi, [Nt, Nrf]))  # frequency-flat
            FRF_vec = FRF.flatten('F')
            batch_X[ii, :] = np.concatenate((FRF_vec.real, FRF_vec.imag), axis=0)

            # generate channel matrix
            if channel_type == 'Rician':
                Hii = 1 / np.sqrt(2) * (np.random.randn(Nr, Nt, K) + 1j * np.random.randn(Nr, Nt, K))
                batch_H[ii, :, :, :] = Hii
            else:
                #H_ii, At_ii = channel_model(Nt, Nr, Pulse_Filter=Pulse_Filter, fc=fc, Ncl=Ncl, Nray=Nray, bandwidth=bandwidth)
                H_ii = generate_user_in_line_multipath(Nt, Nr)
                batch_H[ii, :, :] = H_ii
                #batch_At[ii, :, :, :] = At_ii
                #At = At_ii[:, :, k]
                #U, S, VH = np.linalg.svd(H_ii[:, :])
                #V = VH.T.conj()
                #Fopt = V[:, 0:Ns]  # np.sqrt(Ns) *
                #Fopt=np.dot(np.conj(H_ii).T), np.linalg.inv(np.dot(H_ii, np.conj(H_ii).T))
                Pow = 1
                # batch_At[ii, :, :, :] = At_ii
                Fopt = RZF(H_ii, Pow)
                #Wopt = U[:, 0:Ns]

                ## construct training data
                ztilde = Fopt.flatten('F')
                z = np.concatenate((ztilde.real, ztilde.imag), axis=0)  # convert to real values
                # z_vector = np.matrix(z)

                #if init_scheme == 0:  # random FRF, FBB = LS solution
                FBB = np.matmul(np.linalg.pinv(FRF), Fopt)
                Y_tt = re_Y()
                P = 1
                sigma2_x = 1
                numerator = np.sqrt(P) * FBB  # 分子部分

                denominator = np.sqrt(
                    np.trace(sigma2_x / 2 * np.real(np.conjugate(FRF @ FBB).T @ Y_tt @ (FRF @ FBB))))  # 分母部分
                FBB = numerator / denominator  # 结果
                #else:  # obtain FRF and FBB based on OMP for all frequencies ==> better
                    #FRF, FBB = OMP(Fopt, At)

                Btilde = np.kron(FBB.T, np.identity(Nt))
                B1 = np.concatenate((Btilde.real, -Btilde.imag), axis=1)
                B2 = np.concatenate((Btilde.imag, Btilde.real), axis=1)
                B = np.concatenate((B1, B2), axis=0)
                # print(np.shape(B))

                # new for array response
                #AtH = At.conj().T
                #Atilde = np.kron(np.identity(Nrf), AtH)
                #A1 = np.concatenate((Atilde.real, -Atilde.imag), axis=1)
                #A2 = np.concatenate((Atilde.imag, Atilde.real), axis=1)
                #A = np.concatenate((A1, A2), axis=0)
                # print(np.shape(A))

                # Assign data to the ii-th batch
                # err = z_vector.dot(B) -np.matmul(B.T, z)
                # err1 = np.matmul(z_vector,B) - z_vector.dot(B)

                batch_Bz[ii, :] = np.matmul(B.T, z)
                batch_BB[ii, :, :] = np.matmul(B.T, B)
                batch_z[ii, :] = z
                batch_B[ii, :, :] = B.T
                batch_Fopt[ii, :, :] = Fopt
                #batch_Wopt[ii, :, :] = Wopt
                batch_Fbb[ii, :, :] = FBB
                #batch_AA[ii, :, :] = np.matmul(A.T, A)


            # Hgap = np.linalg.norm(H,ord='fro')/np.sqrt(Nt*Nr)
            # print(f'HQ is: {Hgap:.4f}')
            # Compute optimal digital precoder


        batch_Bz_set = append_data(batch_Bz_set, batch_size, batch_Bz)  # add new data into set
        batch_BB_set = append_data(batch_BB_set, batch_size, batch_BB)
        batch_X_set = append_data(batch_X_set, batch_size, batch_X)
        batch_Z_set = append_data(batch_Z_set, batch_size, batch_z)
        batch_B_set = append_data(batch_B_set, batch_size, batch_B)

        batch_H_real_set = append_data(batch_H_real_set, batch_size, batch_H.real)
        batch_H_imag_set = append_data(batch_H_imag_set, batch_size, batch_H.imag)

        batch_Fopt_real_set = append_data(batch_Fopt_real_set, batch_size, batch_Fopt.real)
        batch_Fopt_imag_set = append_data(batch_Fopt_imag_set, batch_size, batch_Fopt.imag)
        if data == 'testing':
            #batch_Wopt_real_set = append_data(batch_Wopt_real_set, batch_size, batch_Wopt.real)
            #batch_Wopt_imag_set = append_data(batch_Wopt_imag_set, batch_size, batch_Wopt.imag)

            batch_Fbb_real_set = append_data(batch_Fbb_real_set, batch_size, batch_Fbb.real)
            batch_Fbb_imag_set = append_data(batch_Fbb_imag_set, batch_size, batch_Fbb.imag)

            #batch_At_real_set = append_data(batch_At_real_set, batch_size, batch_At.real)
            #batch_At_imag_set = append_data(batch_At_imag_set, batch_size, batch_At.imag)



    ccc = 1


def exponentially_decay_lr(lr_ini, lr_lb, decay_factor, learning_steps, decay_steps, staircase=1):
    '''
    The latex formular is given as
        $\alpha = \max(\alpha_0 \beta^{\left \lfloor \frac{t}{{\Delta t}^I}\right \rfloor},\alpha_e)$

    :param lr_ini(\alpha_0): initial learning rate
    :param lr_lb(\alpha_e): learning rate lower bound
    :param decay_factor(\beta): decay factor of learning rate
    :param learning_steps(t): number of learning steps
    :param decay_steps(\Delta t): the number of steps that the learning rate keeps the same
    :param staircase(I): whether the staircase decrease of learning rate is adopted. 1 indicates True by default. If it is
    False, then the decay_steps doesn't function anymore.
    :return: decayed learning rate (\alpha)
    '''
    import math
    if staircase:
        med_steps = decay_steps
    else:
        med_steps = 1
    lr_decayed = lr_ini*decay_factor**(math.floor(learning_steps/med_steps))
    lr = max(lr_decayed,lr_lb)
    return lr




'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                        The architecture of ScNet
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

class VectorLinear(nn.Module):
    def __init__(self, N, keep_bias=True):
        super(VectorLinear, self).__init__()
        self.keep_bias = keep_bias
        # print(f'mask is {self.mask}')
        self.weight = nn.Parameter(torch.randn([1, N]))  # initialize weight
        # print(f'0 weight is {self.weight}')
        if self.keep_bias:
            self.bias = nn.Parameter(torch.randn([1, N]))  # initialize bias
        # print(f'0 bias is {self.bias}')
        self.reset_parameters()  # self-defined initialization

    def forward(self, input):
        if self.keep_bias:
            return input*self.weight + self.bias
        else:
            return input * self.weight

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                # nn.init.xavier_normal_(p)
                nn.init.normal_(p, std=0.01)
            else:
                # nn.init.xavier_normal_(p)
                nn.init.normal_(p, std=0.01)


class FcNet(nn.Module):
    def __init__(self, dim, num_subcarriers, Loss_scalar=1, training_method='unsupervised', device=MainDevice):
        super(FcNet, self).__init__()
        self.in_dim = dim*(dim+1)
        self.training_method = training_method
        self.device = device
        self.scalar = Loss_scalar
        self.num_subcarriers = num_subcarriers

        self.layer1 = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim),
            nn.BatchNorm1d(self.in_dim, momentum=0.2),
            nn.PReLU(),
            # nn.ReLU(),
            nn.Dropout(p=0.3),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(self.in_dim, dim ** 2),
            nn.BatchNorm1d(dim ** 2, momentum=0.2),
            nn.PReLU(),
            # nn.ReLU(),
            nn.Dropout(p=0.3),
        )

        self.layer3 = nn.Sequential(
            nn.Linear(dim ** 2, dim ** 2),
            nn.BatchNorm1d(dim ** 2, momentum=0.2),
            nn.PReLU(),
            # nn.ReLU(),
            nn.Dropout(p=0.3),
        )

        self.layer_end = nn.Sequential(
            nn.Linear(dim ** 2, dim),
            nn.BatchNorm1d(dim, momentum=0.2),
            nn.PReLU(),
            # nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(dim, dim)
        )


    def forward(self, input, x, z, B, Mask):
        # x_est = torch.zeros_like(x, requires_grad=True)
        # t = torch.tensor(data=[0.5], device=self.device0)
        x_tmp = self.layer1(input)
        x_tmp = self.layer2(x_tmp)
        # x_tmp = self.layer3(x_tmp)
        x_tmp = self.layer_end(x_tmp)
        x_tmp = x_tmp*Mask
        # x_est = torch.tanh(x_tmp)
        x_est = -1 + torch.nn.functional.relu(x_tmp + 0.5) / 0.5 - torch.nn.functional.relu(
            x_tmp - 0.5) / 0.5
        if self.training_method == 'supervised':

            dis = torch.mean(torch.square(x - x_est))

        else:
            dis_sum = 0
            for k in range(self.num_subcarriers):
                diff = z[:, :, k] - torch.matmul(x_est.unsqueeze(1), B[:, :, :, k]).squeeze()
                dis_sum += torch.mean(torch.square(diff))

        LOSS = self.scalar * dis_sum

        return x_est, LOSS


class Cnn_Net(nn.Module):
    def __init__(self, dim, num_subcarriers, Loss_scalar=10, residule=False, training_method='unsupervised', device=MainDevice):
        super(Cnn_Net, self).__init__()
        self.in_dim = dim
        self.device = device
        self.training_method = training_method
        self.Rsdl = residule
        self.scalar = Loss_scalar
        self.num_subcarriers = num_subcarriers
        sqrt_dim = int(np.sqrt(dim))

        self.conv2d = nn.Sequential(         # input shape (1, N, N)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=sqrt_dim,       # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (sqrt_dim, N, N)
            nn.BatchNorm2d(sqrt_dim),
            nn.ReLU(),                      # activation
            # nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (sqrt_dim, N, N)

            nn.Conv2d(sqrt_dim, 2 * sqrt_dim, 3, 1, 1),
            nn.BatchNorm2d(2 * sqrt_dim),
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (2*sqrt_dim, N/2, N/2)

            nn.Conv2d(2 * sqrt_dim, 4 * sqrt_dim, 3, 1, 1),
            nn.BatchNorm2d(4 * sqrt_dim),
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (4 * sqrt_dim, N/4, N/4)
        )

        self.conv1d = nn.Sequential(         # input shape (1, N)
            nn.Conv1d(1, sqrt_dim, 3, 1, 1),  # output shape (sqrt_dim, N, N)
            nn.BatchNorm1d(sqrt_dim),
            nn.ReLU(),                      # activation
            # nn.MaxPool1d(kernel_size=2),    # choose max value in 2x2 area, output shape (sqrt_dim, N)

            nn.Conv1d(sqrt_dim, 2 * sqrt_dim, 3, 1, 1),
            nn.BatchNorm1d(2 * sqrt_dim),
            nn.ReLU(),  # activation
            nn.MaxPool1d(kernel_size=2),  # choose max value in 2x2 area, output shape (2*sqrt_dim, N/2)

            nn.Conv1d(2 * sqrt_dim, 4 * sqrt_dim, 3, 1, 1),
            nn.BatchNorm1d(4 * sqrt_dim),
            nn.ReLU(),  # activation
            nn.MaxPool1d(kernel_size=2),  # choose max value in 2x2 area, output shape (4 * sqrt_dim, N/4)
        )
        self.flatten_dim = 4 * sqrt_dim * (self.in_dim/4 + (self.in_dim/4) ** 2)

        self.Fc2D = nn.Sequential(
            nn.Linear(int(self.flatten_dim), int(self.in_dim**2)),
            nn.BatchNorm1d(self.in_dim**2, momentum=0.2),
            nn.PReLU(),
            # nn.ReLU(),
            # nn.Dropout(p=0.2),

            nn.Linear(self.in_dim**2, self.in_dim),
            nn.BatchNorm1d(self.in_dim, momentum=0.2),
            nn.PReLU(),
            # nn.ReLU(),
            # nn.Dropout(p=0.2),
        )


        self.layer_end = nn.Linear(self.in_dim, self.in_dim)



    def forward(self, BB, Bz, x, z, B):
        # x_est = torch.zeros_like(x, requires_grad=True)
        # t = torch.tensor(data=[0.5], device=self.device0)
        BB_med = BB.unsqueeze(1)
        Bz_med = Bz.unsqueeze(1)
        x_tmp1 = self.conv2d(BB.unsqueeze(1))
        x_tmp1 = x_tmp1.view(x_tmp1.size(0), -1)
        x_tmp2 = self.conv1d(Bz.unsqueeze(1))
        x_tmp2 = x_tmp2.view(x_tmp2.size(0), -1)

        x_temp = torch.cat((x_tmp1, x_tmp2), 1)

        x_tmp = self.Fc2D(x_temp)
        x_tmp = self.layer_end(x_tmp)

        # x_est = torch.tanh(x_tmp)
        x_est = -1 + torch.nn.functional.relu(x_tmp + 0.5) / 0.5 - torch.nn.functional.relu(
            x_tmp - 0.5) / 0.5
        if self.training_method == 'supervised':

            dis = torch.mean(torch.square(x - x_est))

        else:
            dis_sum = 0
            for k in range(self.num_subcarriers):
                diff = z[:, :, k] - torch.matmul(x_est.unsqueeze(1), B[:, :, :, k]).squeeze()
                dis_sum += torch.mean(torch.square(diff))

        LOSS = self.scalar * dis_sum

        return x_est, LOSS


class ScNet(nn.Module):
    def __init__(self, in_dim, num_layer, Loss_scalar=1, IL=False, Keep_Bias=True, BN = True, Sub_Connected=False, training_method='unsupervised'):
        super(ScNet, self).__init__()
        self.in_dim = in_dim
        self.training_method = training_method
        self.dobn = BN
        self.IL = IL
        self.Sub_Connected = Sub_Connected
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers1K = nn.ModuleList()
        self.layers2K = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.bnsK = nn.ModuleList()
        self.num_layer = num_layer
        self.scalar = Loss_scalar
        #self.num_subcarriers = num_subcarriers
        # self.t = torch.tensor(data=[0.5])

        for i in range(num_layer):  # define all layers

            # self.layers1.append(VectorLinear(N, keep_bias=Keep_Bias))
            self.layers1.append(VectorLinear(N, keep_bias=Keep_Bias))
            self.layers2.append(VectorLinear(N, keep_bias=Keep_Bias))
            # for k in range(self.num_subcarriers):
            self.layers1K.append(VectorLinear(N, keep_bias=Keep_Bias))
            self.layers2K.append(VectorLinear(N, keep_bias=Keep_Bias))
            if self.dobn:
                bn_layer = nn.BatchNorm1d(self.in_dim, momentum=0.2)
                #setattr(self, 'bn_layers%i'%i, bn_layer)
                self.bnsK.append(nn.BatchNorm1d(self.in_dim, momentum=0.2))

            if self.dobn:
                # bn_layer = nn.BatchNorm1d(self.in_dim, momentum=0.2)
                # setattr(self, 'bn_layers%i'%i, bn_layer)
                self.bns.append(nn.BatchNorm1d(self.in_dim, momentum=0.2))


    def forward(self, BB, zB, x, z, B, Mask):
        # batch_size = zB.size()[0]
        LOSS = []
        x_est = torch.zeros_like(x, requires_grad=True)

        for l in range(self.num_layer):
            batch_Bz_sum = 0
            batch_BB_sum = 0
            # for k in range(self.num_subcarriers):
            index = l
            batch_Bz_sum = batch_Bz_sum + zB[:, :]
            batch_BB_sum = batch_BB_sum + BB[:, :, :]
            if self.IL:
                aux_term = torch.bmm(x_est.unsqueeze(1), batch_BB_sum).squeeze() - batch_Bz_sum
                out = self.layers1K[index](aux_term) + self.layers2K[index](x_est)
                # out = self.layers1K[index](aux_term + x_est)
                if self.dobn:
                    x_est = self.bnsK[index](out)
                x_est = x_est * Mask
                x_est = -1 + torch.nn.functional.relu(x_est + 0.5) / 0.5 - torch.nn.functional.relu(
                    x_est - 0.5) / 0.5



            if not self.IL:
                aux_term = torch.bmm(x_est.unsqueeze(1), batch_BB_sum).squeeze() - batch_Bz_sum
                if self.Sub_Connected:
                    out = self.layers1[l](aux_term * Mask) + self.layers2[l](x_est)
                else:
                    out = self.layers1[l](aux_term) + self.layers2[l](x_est)
                # out = self.layers1[l](aux_term + x_est)
                if self.dobn:
                    x_est = self.bns[l](out)
                x_est = x_est * Mask
                # if l<self.num_layer-1:
                #     x_est = torch.nn.functional.relu(x_est)
                #     # x_est = torch.nn.functional.leaky_relu(x_est)
                # else:
                #     # x_est = -1 + tf.nn.relu(x_tmp + t) / tf.abs(t) - tf.nn.relu(x_tmp - t) / tf.abs(t)
                #     # x_est = -1 + torch.nn.functional.relu(x_est + t) / torch.abs(t) - torch.nn.functional.relu(x_est - t) / torch.abs(t)
                #     x_est = torch.tanh(x_est)
                x_est = -1 + torch.nn.functional.relu(x_est + 0.5) / 0.5 - torch.nn.functional.relu(
                    x_est - 0.5) / 0.5

            if self.training_method == 'supervised':

                dis = torch.mean(torch.square(x - x_est))

            else:
                dis_sum = 0
                #for k in range(self.num_subcarriers):
                diff = z[:, :] - torch.matmul(x_est.unsqueeze(1), B[:, :, :]).squeeze()
                dis_sum += torch.mean(torch.square(diff))

            LOSS.append(self.scalar*np.log(l+1) * dis_sum)

        return x_est, LOSS


class ScNet_Wideband(nn.Module):
    def __init__(self, in_dim, num_subcarriers, num_layer, Loss_scalar=10, Residule=False, Keep_Bias=False, BN = True, training_method='unsupervised', device=MainDevice):
        super(ScNet_Wideband, self).__init__()
        self.in_dim = in_dim
        self.training_method = training_method
        self.device = device
        self.Rsdl = Residule
        self.dobn = BN
        self.layers_x = nn.ModuleList()
        self.layers_KL = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.num_layer = num_layer
        self.scalar = Loss_scalar
        self.num_subcarriers = num_subcarriers

        for i in range(num_layer):  # define all layers
            # layer = VectorLinear(N, keep_bias=Keep_Bias)
            self.layers_x.append(VectorLinear(N, keep_bias=Keep_Bias))
            # setattr(self, 'layer_x_%i' % i, layer)
            for k in range(num_subcarriers):
                # layerk = VectorLinear(N, keep_bias=Keep_Bias)
                # layer_id = str(i)+str(k)
                # setattr(self, 'layer_bzx_'+ layer_id, layerk)  ## another method is to use nn.ModuleList
                self.layers_KL.append(VectorLinear(N, keep_bias=Keep_Bias))

            if self.dobn:
                # bn_layer = nn.BatchNorm1d(self.in_dim, momentum=0.2)
                # setattr(self, 'bn_layers%i'%i, bn_layer)
                self.bns.append(nn.BatchNorm1d(self.in_dim, momentum=0.2))


    def forward(self, BB, zB, x, z, B):
        # batch_size = zB.size()[0]
        LOSS = []
        x_est = torch.randn_like(x, requires_grad=True)

        for l in range(self.num_layer):
            out_x = self.layers_x[l](x_est)
            # Bzx_sum = torch.zeros_like(x, device=self.device)
            for k in range(self.num_subcarriers):
                index = l*self.num_subcarriers + k
                aux_term = torch.bmm(x_est.unsqueeze(1), BB[:, :, :, k]).squeeze() - zB[:, :, k]
                out_x += self.layers_KL[index](aux_term)

            x_est = out_x
            # xxx = x_est @ BB
            # err = torch.bmm(x_est.unsqueeze(1), BB) - torch.matmul(x_est.unsqueeze(1), BB)
            # print(f'BN layers:{self.bns[l]}')
            if self.dobn:
                x_est = self.bns[l](x_est)

            if l<self.num_layer-1:
                x_est = torch.nn.functional.relu(x_est)
                # x_est = torch.nn.functional.leaky_relu(x_est)
            else:
                # x_est = torch.tanh(x_est)
                x_est = -1 + torch.nn.functional.relu(x_est + 0.5) / 0.5 - torch.nn.functional.relu(
                    x_est - 0.5) / 0.5
            if self.training_method == 'supervised':

                dis = torch.mean(torch.square(x - x_est))

            else:
                dis_sum = 0
                for k in range(self.num_subcarriers):
                    diff = z[:, :, k] - torch.matmul(x_est.unsqueeze(1), B[:, :, :, k]).squeeze()
                    dis_sum += torch.mean(torch.square(diff))

            LOSS.append(self.scalar*np.log(l+1) * dis_sum)

        return x_est, LOSS


class SaveParameters():
    def __init__(self, directory_model, para_file_name='Logs_Info.txt'):
        self.dir_para_file = os.path.join(directory_model, para_file_name)

        pass

    def first_write(self):
        file_para = open(self.dir_para_file, 'w')
        file_para.write('System parameters:\n')
        file_para.write('Nt= ' + str(Nt) + '\n')
        file_para.write('Nr= ' + str(Nr) + '\n')
        file_para.write('Nrf= ' + str(Nrf) + '\n')
        file_para.write('Ns= ' + str(Ns) + '\n')
        file_para.write('K= ' + str(K) + '\n')
        file_para.write('Ncl= ' + str(Ncl) + '\n')
        file_para.write('Nray= ' + str(Nray) + '\n')
        file_para.write('Array type= ' + str(Array_Type) + '\n')
        file_para.write('Sub_Connected= ' + str(Sub_Connected) + '\n')
        file_para.write('Sub_Structure_Type= ' + str(Sub_Structure_Type) + '\n\n')

        file_para.write('Training setup:\n')
        file_para.write('Training method:' + str(training_method) + '\n')
        file_para.write('Device= ' + str(MainDevice) + '\n')
        file_para.write('Use GPU= ' + str(bool(use_gpu)) + '\n')

        file_para.write('Black_box Net= ' + str(Black_box) + '\n')
        file_para.write('Keep_Bias= ' + str(Keep_Bias) + '\n')
        file_para.write('Wideband Net= ' + str(Wideband_Net) + '\n')
        file_para.write('Residule= ' + str(Residule_NN) + '\n')
        file_para.write('Init_scheme= ' + str(init_scheme) + '\n')
        file_para.write('Iterative_Training= ' + str(Iterative_Training) + '\n')
        file_para.write('Iterations_training= ' + str(Iterations_train) + '\n')
        file_para.write('Increamental_Learning= ' + str(Increamental_Learning) + '\n\n')

        file_para.write('Loss_coef= ' + str(Loss_coef) + '\n')
        file_para.write('Number of layers = ' + str(Num_layers) + '\n')
        file_para.write('Training seed = ' + str(Seed_train) + '\n')
        file_para.write('Testing seed = ' + str(Seed_test) + '\n')
        file_para.write('Traning Epoches= ' + str(Ntrain_Epoch) + '\n')
        file_para.write('Traning dataset size= ' + str(training_set_size_truncated) + '\n')
        file_para.write('Traning batch size= ' + str(train_batch_size) + '\n')
        file_para.write('Total number of training batches= ' + str(Ntrain_batch_total) + '\n')
        file_para.write('Number of training steps per Epoch= ' + str(Ntrain_Batch_perEpoch) + '\n\n')

        file_para.write('Optimizer info:\n')
        file_para.write('Optimizer type: Adam \n')
        file_para.write('Decayed learning rate: ' + str(bool(set_Lr_decay)) + '\n')
        file_para.write('Start learning rate= ' + str(start_learning_rate) + '\n')
        file_para.write('Decay factor= ' + str(Lr_decay_factor) + '\n')
        file_para.write('Learning keep steps= ' + str(Lr_keep_steps) + '\n')
        file_para.write('Learning rate lower bound= ' + str(Lr_min) + '\n')
        file_para.write('Weight decay= ' + str(Weight_decay) + '\n\n')
        # file_para.write('This learning is to improve the accuracy of phase, keep other NN fixed\n')

        file_para.write('Testing setup: \n')
        file_para.write('Testing batch size: ' + str(test_batch_size) + '\n')
        file_para.write('Iterations_testing: ' + str(Iterations_test) + '\n')

        # file_para.write('The training starts at the time= ' + str(time_now_start) + '\n')
        # file_para.write('The training ends at the time= ' + str(time_now_end) + '\n')
        # file_para.write('Training time cost =' + str(time_cost) + 'h\n\n')

        file_para.close()

    def add_logs(self, str):

        file_para = open(self.dir_para_file, 'a')
        file_para.write('\n')
        file_para.write(str + '\n')
        file_para.close()
        pass



class Data_Fetch():
    def __init__(self, file_dir, file_name, batch_size, training_set_size, training_set_size_truncated=training_set_size, data_str='training'):
        self.data_path = file_dir + file_name
        self.batch_size = batch_size
        self.data_str = data_str
        self.len = training_set_size+1
        self.len_truncated = training_set_size_truncated +1
        self.reset()

    def reset(self):
        self.pointer = np.random.randint(self.len_truncated)  # initialize the start position
        self.start_idx = self.pointer

    def get_item(self):
        data_all = h5py.File(self.data_path, 'r')

        self.end_idx = self.start_idx + self.batch_size
        if self.end_idx <= self.len_truncated-1:

            Bz = data_all['batch_Bz'][self.start_idx:self.end_idx, :]
            BB = data_all['batch_BB'][self.start_idx:self.end_idx, :, :]
            X = data_all['batch_X'][self.start_idx:self.end_idx, :]
            Z = data_all['batch_Z'][self.start_idx:self.end_idx, :]
            B = data_all['batch_B'][self.start_idx:self.end_idx, :, :]
            batch_H_real = data_all['batch_H_real'][self.start_idx:self.end_idx, :, :]
            batch_H_imag = data_all['batch_H_imag'][self.start_idx:self.end_idx, :, :]
            batch_Fopt_real = data_all['batch_Fopt_real'][self.start_idx:self.end_idx, :, :]
            batch_Fopt_imag = data_all['batch_Fopt_imag'][self.start_idx:self.end_idx, :, :]


            if self.data_str== 'testing':
                #batch_Wopt_real = data_all['batch_Wopt_real'][self.start_idx:self.end_idx, :, :, :]
                #batch_Wopt_imag = data_all['batch_Wopt_imag'][self.start_idx:self.end_idx, :, :, :]
                batch_Fbb_real = data_all['batch_Fbb_real'][self.start_idx:self.end_idx, :, :]
                batch_Fbb_imag = data_all['batch_Fbb_imag'][self.start_idx:self.end_idx, :, :]
                #batch_At_real = data_all['batch_At_real'][self.start_idx:self.end_idx, :, :, :]
                #batch_At_imag = data_all['batch_At_imag'][self.start_idx:self.end_idx, :, :, :]

                #batch_Wopt = batch_Wopt_real + 1j * batch_Wopt_imag
                batch_Fbb = batch_Fbb_real + 1j * batch_Fbb_imag
                #batch_At = batch_At_real + 1j * batch_At_imag

            data_all.close()
            self.start_idx = self.end_idx

        else:
            remain_num = self.end_idx - self.len_truncated

            Bz1 = data_all['batch_Bz'][self.start_idx:self.len_truncated, :]
            BB1 = data_all['batch_BB'][self.start_idx:self.len_truncated, :, :]
            X1 = data_all['batch_X'][self.start_idx:self.len_truncated, :]
            Z1 = data_all['batch_Z'][self.start_idx:self.len_truncated, :]
            B1 = data_all['batch_B'][self.start_idx:self.len_truncated, :, :]
            batch_H_real1 = data_all['batch_H_real'][self.start_idx:self.len_truncated, :, :]
            batch_H_imag1 = data_all['batch_H_imag'][self.start_idx:self.len_truncated, :, :]
            batch_Fopt_real1 = data_all['batch_Fopt_real'][self.start_idx:self.len_truncated, :, :]
            batch_Fopt_imag1 = data_all['batch_Fopt_imag'][self.start_idx:self.len_truncated, :, :]

            Bz2 = data_all['batch_Bz'][:remain_num, :]
            BB2 = data_all['batch_BB'][:remain_num, :, :]
            X2 = data_all['batch_X'][:remain_num, :]
            Z2 = data_all['batch_Z'][:remain_num, :]
            B2 = data_all['batch_B'][:remain_num, :, :]
            batch_H_real2 = data_all['batch_H_real'][:remain_num, :, :]
            batch_H_imag2 = data_all['batch_H_imag'][:remain_num, :, :]
            batch_Fopt_real2 = data_all['batch_Fopt_real'][:remain_num, :, :]
            batch_Fopt_imag2 = data_all['batch_Fopt_imag'][:remain_num, :, :]

            Bz = np.concatenate((Bz1, Bz2), axis=0)
            BB = np.concatenate((BB1, BB2), axis=0)
            X = np.concatenate((X1, X2), axis=0)
            Z = np.concatenate((Z1, Z2), axis=0)
            B = np.concatenate((B1, B2), axis=0)
            batch_H_real = np.concatenate((batch_H_real1, batch_H_real2), axis=0)
            batch_H_imag = np.concatenate((batch_H_imag1, batch_H_imag2), axis=0)
            batch_Fopt_real = np.concatenate((batch_Fopt_real1, batch_Fopt_real2), axis=0)
            batch_Fopt_imag = np.concatenate((batch_Fopt_imag1, batch_Fopt_imag2), axis=0)


            data_all.close()
            self.start_idx = remain_num

        batch_H = batch_H_real + 1j * batch_H_imag
        batch_Fopt = batch_Fopt_real + 1j * batch_Fopt_imag
        if self.data_str == 'testing':
            return Bz, BB, X, Z, B, batch_H, batch_Fopt,batch_Fbb
        else:
            return Bz, BB, X, Z, B, batch_H, batch_Fopt



if __name__ == '__main__':
    import matplotlib.pyplot as plt


    def generate_training_data():
        # training_set_size = 70
        print('----------------------training data-------------------------')
        batch_Bz, batch_BB, batch_X, batch_Z, batch_B, batch_H, batch_Fopt, batch_Fbb = gen_data_wideband(
            Nt, Nr, Nrf, Ns, batch_size=1, fc=fc, Ncl=Ncl, Nray=Nray, bandwidth=Bandwidth) # batch_size=Gen_Batch_size
        data_all = {'batch_Bz': batch_Bz, 'batch_BB': batch_BB, 'batch_X': batch_X, 'batch_Z': batch_Z,
                    'batch_B': batch_B,
                    'batch_H_real': batch_H.real,
                    'batch_H_imag': batch_H.imag,
                    'batch_Fopt_real': batch_Fopt.real,
                    'batch_Fopt_imag': batch_Fopt.imag,
                    }

        train_data_path = dataset_file + train_data_name
        file_handle = h5py.File(train_data_path, 'w')
        # for name in data_all.keys():
        #     file_handle.attrs[name]=data_all[name]
        # file_handle.close()
        for name in data_all:
            dshp = data_all[name].shape
            dims = list(dshp[1:])
            dims.insert(0, None)
            # print(f'dshp shape:{dshp}, dims shape:{dims}')
            file_handle.create_dataset(name, data=data_all[name], maxshape=dims, chunks=True, compression='gzip',
                                       compression_opts=9)
        # hf = h5py.File(train_data_path, 'r')
        # print('----------------------training data-------------------------')
        # for key in hf.keys():
        #     print(key, hf[key])


    def generate_testing_data(Pulse_Filter=False):
        # testing_set_size = 30
        print('----------------------testing data-------------------------')
        batch_Bz, batch_BB, batch_X, batch_Z, batch_B, batch_H, batch_Fopt, batch_Fbb= gen_data_wideband(
            Nt, Nr, Nrf, Ns, batch_size=1, Pulse_Filter=Pulse_Filter, fc=fc, Ncl=Ncl, Nray=Nray, bandwidth=Bandwidth)  # batch_size=Gen_Batch_size
        data_all = {'batch_Bz': batch_Bz, 'batch_BB': batch_BB, 'batch_X': batch_X, 'batch_Z': batch_Z, 'batch_B': batch_B,
                    'batch_H_real': batch_H.real,
                    'batch_H_imag': batch_H.imag,
                    'batch_Fopt_real': batch_Fopt.real,
                    'batch_Fopt_imag': batch_Fopt.imag,
                    'batch_Fbb_real': batch_Fbb.real,
                    'batch_Fbb_imag': batch_Fbb.imag,
                    }

        # data_all = {'batch_Bz': batch_Bz, 'batch_BB': batch_BB, 'batch_X': batch_X, 'batch_Z': batch_Z, 'batch_B': batch_B,
        #             'batch_H_real': batch_H.real, 'batch_H_imag': batch_H.imag,
        #             'batch_Fopt_real': batch_Fopt.real, 'batch_Fopt_imag': batch_Fopt.imag,
        #             'batch_Wopt_real': batch_Wopt.real, 'batch_Wopt_imag':batch_Wopt.imag,
        #             'batch_Fbb_real': batch_Fbb.real, 'batch_Fbb_imag': batch_Fbb.imag,
        #             'batch_At_real': batch_At.real,'batch_At_imag': batch_At.imag}
        # print(f'H:{batch_Hb}')
        # test_data_name = 'test_set.hdf5'
        test_data_path = dataset_file + test_data_name
        file_handle = h5py.File(test_data_path, 'w')
        # for name in data_all.keys():
        #     file_handle.attrs[name]=data_all[name]
        # file_handle.close()
        for name in data_all:
            dshp = data_all[name].shape
            dims = list(dshp[1:])
            dims.insert(0, None)
            file_handle.create_dataset(name, data=data_all[name], maxshape=dims, chunks=True, compression='gzip',
                                       compression_opts=9)
            # file_handle.create_dataset(name, data=data_all[name], chunks=True, compression='gzip',
            #                            compression_opts=9)
        # print(name)

        hf = h5py.File(test_data_path, 'r')
        print('----------------------testing data-------------------------')
        for key in hf.keys():
            print(key, hf[key])


    generate_testing_data()
    gen_data_large(Nt, Nr, Nrf, Ns, Num_batch=GenNum_Batch_te, batch_size=Gen_Batch_size_te, fc=fc, Ncl=Ncl, Nray=Nray,
                   bandwidth=Bandwidth, data='testing')

    generate_training_data()
    gen_data_large(Nt, Nr, Nrf, Ns, Num_batch=GenNum_Batch_tr, batch_size=Gen_Batch_size_tr, fc=fc, Ncl=Ncl, Nray=Nray, bandwidth=Bandwidth)

    # train_data_path = dataset_file + train_data_name
    # hf = h5py.File(train_data_path, 'r')
    # print('----------------------training data-------------------------')
    # for key in hf.keys():
    #     print(key, hf[key])

    test_data_path = dataset_file + test_data_name
    hf = h5py.File(test_data_path, 'r')
    print('----------------------testing data-------------------------')
    for key in hf.keys():
        print(key, hf[key])
    # for idx, data in enumerate(batch_Bz):
    #     print(f'idx={idx},data is : {data}')

    # print(hf.attrs.keys() )
    # s1 = hf.attrs['batch_Bz']
    # print(f'batch_Bz is : {s1.dtpye}')
    ccc=1
    pass
    def draw_lrfunc():
        lr_ini = 0.001
        lr_lb = 1e-4
        decay_factor = 0.8
        decay_steps = 10
        staircase = 1
        num_learning_steps = 400
        Lr_all = []
        for step in range(num_learning_steps):
            lr = exponentially_decay_lr(lr_ini, lr_lb, decay_factor=decay_factor, learning_steps=step,
                                        decay_steps=decay_steps, staircase=staircase)
            Lr_all.append(lr)

        plt.figure(dpi=100)
        plt.plot(Lr_all, label=r'$\psi(x)$')

        plt.legend(loc='center right')
        # plt.xticks(x)
        plt.xlabel('steps')
        plt.ylabel('lr value')
        plt.grid(True)
        plt.show()
        # print(batch_Z)
