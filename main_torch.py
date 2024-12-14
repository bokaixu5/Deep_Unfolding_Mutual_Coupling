# -*- oding:utf-8 -*-
# -* coding: utf-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import torch

import logging
import time
import datetime
import matplotlib.pyplot as plt
import scipy.io as spio
from FuncLbr import *
from Global_Vars import *
from scipy.io import loadmat

train_ManNet = 0

# os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable INFO\WARNING prompt
#logging.disable(logging.WARNING)  # forbidden all log info

save_paras = SaveParameters(directory_model, para_file_name='Logs_Info.txt')
save_paras.first_write()
time_now_start = datetime.datetime.now().strftime('%Y-%m-%d %H-%M%S')
print('The trainings starts at the time:', time_now_start,flush=True)  # 当前时间
save_paras.add_logs('The training starts at the time= ' + str(time_now_start))
t_start = time.time()

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
def re_Y():
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
    # return Y_tt


# train the model
def train_model(BB_beamformer = 'LS'):
    torch.manual_seed(Seed_train)
    np.random.seed(Seed_train)
    myModel.train()  # training mode
    print('start training',flush=True)
    Lr_list = []
    Loss_cache = []

    batch_count = 0
    for epoch in range(Ntrain_Epoch):
        dataloader_tr.reset()
        print('-----------------------------------------------')
        for batch_idx in range(Ntrain_Batch_perEpoch):
            batch_count += 1
            batch_Bz, batch_BB, batch_X, batch_Z, batch_B, batch_H, batch_Fopt = dataloader_tr.get_item()
            batch_Mask = masking_dyn(batch_H, sub_connected=Sub_Connected, sub_structure_type=Sub_Structure_Type)
            batch_Mask = torch.from_numpy(batch_Mask).float()
            batch_Bz = torch.from_numpy(batch_Bz).float()
            batch_BB = torch.from_numpy(batch_BB).float()
            batch_X = torch.from_numpy(batch_X).float()
            batch_Z = torch.from_numpy(batch_Z).float()
            batch_B = torch.from_numpy(batch_B).float()

            batch_Bz_sum = 0
            batch_BB_sum = 0
            #for k in range(K):
            batch_Bz_sum += batch_Bz[:, :]
            batch_BB_sum += batch_BB[:, :, :]

            BB_sum_vec = torch.reshape(batch_BB_sum, [-1, N**2])
            FcNet_input = torch.cat((batch_Bz_sum, BB_sum_vec), axis=1)

            # for s in range(train_batch_size):
            #     BBs_vec = BB_sum_vec[s,:]
            #     stmp = batch_BB_sum[s,:,:]
            #     err = BBs_vec - torch.flatten(stmp)
            #     print(f'err is {err}')

            # dis_sum = 0
            # for k in range(K):
            #     med = np.matmul(np.expand_dims(batch_X, 1), batch_B[:, :, :, k]).squeeze()
            #     diff = batch_Z[:, :, k] - med
            #     dis_sum += np.sum(np.linalg.norm(diff, 'fro'))
            # print(f'{batch_idx} error:{dis_sum}')


            if Black_box:
                x_est, loss = myModel(FcNet_input.to(MainDevice), batch_X.to(MainDevice), batch_Z.to(MainDevice), batch_B.to(MainDevice), batch_Mask.to(MainDevice))
                # x_est, loss = myModel(batch_BB_sum.to(MainDevice), batch_Bz_sum.to(MainDevice), batch_X.to(MainDevice), batch_Z.to(MainDevice),
                #                       batch_B.to(MainDevice))
            else:

                if Wideband_Net:
                    s_hat, loss_list = myModel(batch_BB.to(MainDevice), batch_Bz.to(MainDevice), batch_X.to(MainDevice),
                                               batch_Z.to(MainDevice), batch_B.to(MainDevice))
                else:
                    s_hat, loss_list = myModel(batch_BB.to(MainDevice), batch_Bz.to(MainDevice), batch_X.to(MainDevice), batch_Z.to(MainDevice),
                                               batch_B.to(MainDevice), batch_Mask.to(MainDevice))
                if SUM_LOSS==1:
                    loss = sum(loss_list)
                else:
                    loss = loss_list[-1]

            Loss_cache.append(loss.item())

            if set_Lr_decay:
                for g in optimizer.param_groups:
                    g['lr'] = exponentially_decay_lr(lr_ini=start_learning_rate, lr_lb=Lr_min, decay_factor=Lr_decay_factor,
                                                     learning_steps=batch_count, decay_steps=Lr_keep_steps, staircase=1)
            loss.requires_grad_(True)
            torch.cuda.empty_cache()
            optimizer.zero_grad()  # zero gradient
            loss.backward()  # backpropagation
            optimizer.step()  # update training prapameters
            torch.cuda.empty_cache()
            Lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
            # for name, params in myModule.named_parameters():
            #     if 'layer_axz_0.weight' in name:
            #         print(f'epoch {epoch} after update: name {name}, params {params}')
            if (batch_idx) % Log_interval == 0:
                len_loss = len(Loss_cache)
                if len_loss > 2 * Log_interval:
                    avr_loss = np.mean(Loss_cache[len_loss-Log_interval:])  # 取倒数Log_interval个loss做平均
                    print(f'Epoch:{epoch}, batch_id:{batch_idx}, learning rate: {Lr_list[-1]:.5f}, average loss:{avr_loss:.6f}',flush=True)

            if not Black_box and Iterative_Training:  # start iterative training
                s_hat = s_hat.detach().cpu().numpy()
                batch_Bz = batch_Bz.numpy()
                batch_BB = batch_BB.numpy()
                batch_X = batch_X.numpy()
                batch_Z = batch_Z.numpy()
                batch_B = batch_B.numpy()

                for jj in range(Iterations_train):
                    # s_dim = s_hat.shape[0]
                    # 1. Update input to the network: only update data related to Frf, not change the channels
                    for ii in range(s_hat.shape[0]):
                        ff = s_hat[ii, :]  # prepare testing data
                        FF = np.reshape(ff, [Nt * Nrf, 2], 'F')
                        ff_complex = FF[:, 0] + 1j * FF[:, 1]  # convert to complex vector
                        FRF = np.reshape(ff_complex, [Nt, Nrf], 'F')  # convert to RF precoding matrix
                        FRF = normalize(FRF, sub_connected=Sub_Connected, sub_structure_type=Sub_Structure_Type)

                        #for k in range(K):
                        Fopt_ii = batch_Fopt[ii, :, :]  # recall optimal fully digital precoder
                        Hii = batch_H[ii, :, :]
                        Uo, So, VoH = np.linalg.svd(Hii)

                        # solution to Fbb
                        if BB_beamformer == 'LS':  # LS
                            FBB = np.matmul(np.linalg.pinv(FRF), Fopt_ii)  # compute BB precoder
                        else:  # equal-weight water-filling
                            Heff = np.matmul(Hii, FRF)
                            Q = np.matmul(FRF.conj().T, FRF)
                            Qrank = np.linalg.matrix_rank(Q, tol=1e-4)
                            Uq, S, UqH = np.linalg.svd(Q)
                            Uqnew = Uq[:, 0:Qrank]
                            # print(f'S:{S}')
                            # Snew = S[0:Qrank]
                            Snew = 1 / (np.sqrt(S[0:Qrank]))
                            Qinvsqrt = np.dot(Uqnew * Snew, Uqnew.conj().T)
                            # term1 = Qnew @ Qnew
                            # term2 = term1 @ Q

                            # err= np.linalg.norm(term2-np.eye(Nrf))
                            # print(f'err:{err}')

                            U, S, VH = np.linalg.svd(Heff * Qinvsqrt)

                            V = VH.T.conj()
                            FBB = np.matmul(Qinvsqrt, V[:, 0:Ns])
                        Y_tt = re_Y()
                        P = 1
                        sigma2_x = 1
                        numerator = np.sqrt(P) * FBB  # 分子部分

                        denominator = np.sqrt(
                            np.trace(sigma2_x / 2 * np.real(np.conjugate(FRF @ FBB).T @ Y_tt @ (FRF @ FBB))))  # 分母部分
                        FBB = numerator / denominator  # 结果

                        #FBB =FBB / np.linalg.norm(np.matmul(FRF, FBB), 'fro')

                        Btilde = np.kron(FBB.T, np.identity(Nt))

                        # convert to real values
                        z_ii = batch_Z[ii, :]
                        B1 = np.concatenate((Btilde.real, -Btilde.imag), axis=1)
                        B2 = np.concatenate((Btilde.imag, Btilde.real), axis=1)
                        B_ii = np.concatenate((B1, B2), axis=0)

                        # B1 = B_ii.T
                        batch_Bz[ii, :] = np.matmul(B_ii.T, z_ii)  # update values
                        batch_BB[ii, :, :] = np.matmul(B_ii.T, B_ii)
                        batch_B[ii, :, :] = B_ii.T



                    # Update training data

                    batch_Bz = torch.from_numpy(batch_Bz)
                    batch_BB = torch.from_numpy(batch_BB)
                    batch_X = torch.from_numpy(batch_X)
                    batch_Z = torch.from_numpy(batch_Z)
                    batch_B = torch.from_numpy(batch_B)

                    if Wideband_Net:
                        s_hat, loss_list = myModel(batch_BB.to(MainDevice), batch_Bz.to(MainDevice), batch_X.to(MainDevice),
                                                   batch_Z.to(MainDevice), batch_B.to(MainDevice))
                    else:
                        batch_Bz_sum = 0
                        batch_BB_sum = 0
                        #for k in range(K):
                        batch_Bz_sum += batch_Bz[:, :]
                        batch_BB_sum += batch_BB[:, :, :]

                        s_hat, loss_list = myModel(batch_BB.to(MainDevice), batch_Bz.to(MainDevice), batch_X.to(MainDevice), batch_Z.to(MainDevice),
                                                   batch_B.to(MainDevice), batch_Mask.to(MainDevice))
                    if SUM_LOSS == 1:
                        loss = sum(loss_list)
                    else:
                        loss = loss_list[-1]

                    torch.cuda.empty_cache()
                    optimizer.zero_grad()  # zero gradient
                    loss.backward()  # backpropagation
                    optimizer.step()  # update training prapameters
                    torch.cuda.empty_cache()


                    s_hat = s_hat.detach().cpu().numpy()
                    batch_X = batch_X.numpy()
                    batch_Z = batch_Z.numpy()
                    batch_B = batch_B.numpy()
                    batch_Bz = batch_Bz.numpy()
                    batch_BB = batch_BB.numpy()


            if batch_idx >= Ntrain_Batch_perEpoch:
                break


    return Loss_cache, Lr_list

def SumRate(H, P, sigma2):
    K = 6
    c = np.zeros((1, K))
    for idx1 in range(K):
        ds = np.abs(np.matmul(H[:, idx1].T, P[:, idx1])) ** 2
        inter = 0

        for idx2 in range(K):
            if idx2 != idx1:
                inter += np.abs(np.matmul(H[:, idx1].T, P[:, idx2])) ** 2

        sinr_k = ds / (sigma2 + inter)
        c[0, idx1] = np.log2(1 + sinr_k)

    C = np.sum(c)
    return C

def tst_model(BB_beamformer ='LS'):
    torch.manual_seed(Seed_test)
    np.random.seed(Seed_test)
    save_paras.add_logs('\n Test:')
    myModel.eval()  # testing mode
    myModel.to('cpu')  # test on CPU
    dataloader_te.start_idx = 0
    f_all = []

    for batch_idx in range(1):
        print(f'batch_id:{batch_idx}')
        batch_Bz, batch_BB, batch_X, batch_Z, batch_B, batch_H, batch_Fopt, batch_Fbb = dataloader_te.get_item()

        batch_Mask = masking_dyn(batch_H, sub_connected=Sub_Connected, sub_structure_type=Sub_Structure_Type)
        batch_Mask = torch.from_numpy(batch_Mask).float()
        batch_Bz = torch.from_numpy(batch_Bz).float()
        batch_BB = torch.from_numpy(batch_BB).float()
        batch_X = torch.from_numpy(batch_X).float()
        batch_Z = torch.from_numpy(batch_Z).float()
        batch_B = torch.from_numpy(batch_B).float()



        # batch_Bz = data['batch_Bz'].float()
        # batch_BB = data['batch_BB'].float()
        # batch_X = data['batch_X'].float()
        # batch_Z = data['batch_Z'].float()
        # batch_B = data['batch_B'].float()
        # batch_H = data['batch_H'].float()
        # batch_Fopt = data['batch_Fopt'].float()
        # batch_Wopt = data['batch_Wopt'].float()
        # batch_Fbb = data['batch_Fbb'].float()
        # At = data['batch_At'].float()

        # dis_sum = 0
        # for k in range(K):
        #     med = np.matmul(np.expand_dims(batch_X, 1), batch_B[:, :, :, k]).squeeze()
        #     diff = batch_Z[:, :, k] - med
        #     dis_sum += np.sum(np.linalg.norm(diff, 'fro'))
        # print(f'{batch_idx} error:{dis_sum}')

        batch_Bz_sum = 0
        batch_BB_sum = 0
        #for k in range(K):
        batch_Bz_sum += batch_Bz[:, :]
        batch_BB_sum += batch_BB[:, :, :]

        BB_sum_vec = torch.reshape(batch_BB_sum, [-1, N ** 2])
        FcNet_input = torch.cat((batch_Bz_sum, BB_sum_vec), axis=1)

        if Black_box:
            s_hat, loss = myModel(FcNet_input, batch_X, batch_Z, batch_B,batch_Mask)
            # s_hat, loss = myModel(batch_BB_sum, batch_Bz_sum, batch_X,batch_Z, batch_B)
            s_hat = s_hat.detach().numpy()
        else:

            if Wideband_Net:
                s_hat, loss = myModel(batch_BB, batch_Bz, batch_X, batch_Z, batch_B)
            else:
                s_hat, loss = myModel(batch_BB, batch_Bz, batch_X, batch_Z, batch_B, batch_Mask)




            # batch_Bz_sum = torch.from_numpy(batch_Bz_sum).float()
            # batch_BB_sum = torch.from_numpy(batch_BB_sum).float()

            # batch_Fopt = torch.from_numpy(batch_Fopt).float()

            s_hat = s_hat.detach().numpy()
            batch_Bz = batch_Bz.numpy()
            batch_BB = batch_BB.numpy()
            batch_X = batch_X.numpy()
            batch_Z = batch_Z.numpy()
            batch_B = batch_B.numpy()

            # batch_H = batch_H.numpy()
            # batch_Fopt = batch_Fopt.numpy()
            # batch_Wopt = batch_Wopt.numpy()
            # batch_Fbb = batch_Fbb.numpy()
            # At = At.numpy()

            # batch_Fopt_real = batch_Fopt[:, 0, :, :, :]
            # batch_Fopt_imag = batch_Fopt[:, 1, :, :, :]
            # batch_Wopt_real = batch_Wopt[:, 0, :, :, :]
            # batch_Wopt_imag = batch_Wopt[:, 1, :, :, :]

            # batch_H = batch_H[:, 0, :, :, :] + 1j * batch_H[:, 1, :, :, :]
            # batch_Fopt = batch_Fopt[:, 0, :, :, :] + 1j * batch_Fopt[:, 1, :, :, :]
            # batch_Wopt = batch_Wopt[:, 0, :, :, :] + 1j * batch_Wopt[:, 1, :, :, :]
            # batch_Fbb = batch_Fbb[:, 0, :, :, :] + 1j * batch_Fbb[:, 1, :, :, :]
            # At = At[:, 0, :, :, :] + 1j * At[:, 1, :, :, :]

            # At = np.transpose(At, (1, 2, 0))

            for jj in range(Iterations_test):
                # 1. Update input to the network: only update data related to Frf, not change the channels
                for ii in range(test_batch_size):
                    ff = s_hat[ii, :]# prepare testing data
                    FF = np.reshape(ff, [Nt * Nrf, 2], 'F')
                    ff_complex = FF[:, 0] + 1j * FF[:, 1]  # convert to complex vector
                    FRF = np.reshape(ff_complex, [Nt, Nrf], 'F')  # convert to RF precoding matrix
                    FRF = normalize(FRF, sub_connected=Sub_Connected, sub_structure_type=Sub_Structure_Type)
                    FRF_vec = FRF.flatten('F')
                    batch_X[ii, :] = np.concatenate((FRF_vec.real, FRF_vec.imag), axis=0)
                    #for k in range(K):
                    Fopt_ii = batch_Fopt[ii, :, :]  # recall optimal fully digital precoder
                    Hii = batch_H[ii, :, :]
                    Uo, So, VoH = np.linalg.svd(Hii)
                    Wopt = Uo[:, 0:Ns]
                    # solution to Fbb
                    if BB_beamformer == 'LS':  # LS
                        FBB = np.matmul(np.linalg.pinv(FRF), Fopt_ii)  # compute BB precoder
                    else:  # equal-weight water-filling
                        Heff = np.matmul(Hii, FRF)
                        Q = np.matmul(FRF.conj().T, FRF)
                        Qrank = np.linalg.matrix_rank(Q, tol=1e-4)
                        Uq, S, UqH = np.linalg.svd(Q)
                        Uqnew = Uq[:, 0:Qrank]
                        # print(f'S:{S}')
                        # Snew = S[0:Qrank]
                        Snew = 1 / (np.sqrt(S[0:Qrank]))
                        Qinvsqrt = np.dot(Uqnew * Snew, Uqnew.conj().T)
                        # term1 = Qnew @ Qnew
                        # term2 = term1 @ Q

                        # err= np.linalg.norm(term2-np.eye(Nrf))
                        # print(f'err:{err}')

                        U, S, VH = np.linalg.svd(Heff * Qinvsqrt)

                        V = VH.T.conj()
                        FBB = np.matmul(Qinvsqrt, V[:, 0:Ns])
                    Y_tt=re_Y()
                    P=1
                    sigma2_x=1
                    numerator = np.sqrt(P) * FBB # 分子部分

                    denominator = np.sqrt(
                        np.trace(sigma2_x / 2 * np.real(np.conjugate(FRF@FBB).T @ Y_tt @ (FRF@FBB))))  # 分母部分
                    FBB = numerator / denominator  # 结果


                    Btilde = np.kron(FBB.T, np.identity(Nt))

                    # convert to real values
                    z_ii = batch_Z[ii, :]
                    B1 = np.concatenate((Btilde.real, -Btilde.imag), axis=1)
                    B2 = np.concatenate((Btilde.imag, Btilde.real), axis=1)
                    B_ii = np.concatenate((B1, B2), axis=0)

                    # B1 = B_ii.T
                    batch_Bz[ii, :] = np.matmul(B_ii.T, z_ii)  # update values
                    batch_BB[ii, :, :] = np.matmul(B_ii.T, B_ii)
                    batch_B[ii, :, :] = B_ii.T
                    batch_Fbb[ii, :, :] = FBB
                    ff1 = s_hat[ii, :]  # prepare testing data
                    FF = np.reshape(ff1, [Nt * Nrf, 2], 'F')
                    ff_complex1 = FF[:, 0] + 1j * FF[:, 1]  # convert to complex vector
                    FRF = np.reshape(ff_complex1, [Nt, Nrf], 'F')
                    snr = np.arange(-15, 21, 5)
                    Pmax_t = 1
                    for idx1 in range(len(snr)):
                        sigma2 = Pmax_t / (10 ** (snr[idx1] / 10))
                        sum1 = SE_calculation(Hii, FRF @ FBB, sigma2, sigma2_x)
                        sum2 = SE_calculation(Hii, Fopt_ii, sigma2, sigma2_x)
                        print('----------------')
                        print('Iteration',jj)
                        print('SNR:',snr[idx1])
                        print('HBF:',sum1)
                        print('FD:', sum2)
                        print('----------------')
                    #sum1 = SumRate(Hii, FRF @ FBB, sigma2)







                # Update training data

                batch_Bz = torch.from_numpy(batch_Bz)
                batch_BB = torch.from_numpy(batch_BB)
                batch_X = torch.from_numpy(batch_X)
                batch_Z = torch.from_numpy(batch_Z)
                batch_B = torch.from_numpy(batch_B)

                # batch_Bz = torch.from_numpy(batch_Bz).float()
                # batch_BB = torch.from_numpy(batch_BB).float()
                # batch_X = torch.from_numpy(batch_X).float()
                # batch_Z = torch.from_numpy(batch_Z).float()
                # batch_B = torch.from_numpy(batch_B).float()

                # dis_sum = 0
                # for k in range(K):
                #     med = np.matmul(np.expand_dims(batch_X, 1), batch_B[:, :, :, k]).squeeze()
                #     diff = batch_Z[:, :, k] - med
                #     dis_sum += np.sum(np.linalg.norm(diff, 'fro'))
                # print(f'{jj} error:{dis_sum}')

                if Wideband_Net:
                    s_hat, loss_list = myModel(batch_BB, batch_Bz, batch_X, batch_Z, batch_B)
                else:
                    batch_Bz_sum = 0
                    batch_BB_sum = 0
                    #for k in range(K):
                    batch_Bz_sum += batch_Bz[:, :]
                    batch_BB_sum += batch_BB[:, :, :]
                    s_hat, loss_list = myModel(batch_BB, batch_Bz, batch_X, batch_Z, batch_B, batch_Mask)

                if SUM_LOSS:
                    loss = sum(loss_list)
                else:
                    loss = loss_list[-1]
                s_hat = s_hat.detach().numpy()
                f_all.append(s_hat)
                batch_X = batch_X.numpy()
                batch_Z = batch_Z.numpy()
                batch_B = batch_B.numpy()
                batch_Bz = batch_Bz.numpy()
                batch_BB = batch_BB.numpy()
                print(f'Iteration:{jj}, loss:{loss:.4f}')

                save_paras.add_logs(' Iteration= ' + str(jj)+', loss=' +str(loss.item()))
    return s_hat, f_all, batch_H, batch_Fopt, batch_Fbb

if __name__== '__main__':
# load data
    dataloader_tr = Data_Fetch(file_dir=dataset_file,
                               file_name=train_data_name,
                               batch_size=train_batch_size,
                               training_set_size=training_set_size,
                               training_set_size_truncated=training_set_size_truncated,
                               data_str='training')
    dataloader_te = Data_Fetch(file_dir=dataset_file,
                               file_name=test_data_name,
                               batch_size=test_batch_size,
                               training_set_size=testing_set_size,
                               data_str='testing')

    # define the network
    if Black_box:
        with torch.no_grad():
            myModel = FcNet(N, K, Loss_scalar=Loss_coef, training_method='unsupervised')
            # myModel = Cnn_Net(N, K, Loss_scalar=Loss_coef, residule=Residule_NN, training_method='unsupervised', device=MainDevice)
    else:
        if Wideband_Net:
            myModel = ScNet_Wideband(N, K, Num_layers, Loss_coef, Residule=Residule_NN, Keep_Bias=Keep_Bias, BN=True)
        else:
            myModel = ScNet(N, Num_layers, Loss_coef, IL=Increamental_Learning, Keep_Bias=Keep_Bias, BN=True, Sub_Connected=Sub_Connected)

    myModel.to(MainDevice)


    optimizer = torch.optim.Adam(myModel.parameters(), lr=start_learning_rate, weight_decay=Weight_decay)
    Loss_cache, Lr_list = train_model()

    checkpoint = {'model_state_dict': myModel.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}
    torch.save(checkpoint, model_file_name)  # save model

    time_now_end = datetime.datetime.now().strftime('%Y-%m-%d %H-%M%S')
    print('The trainings ends at the time:', time_now_end,flush=True)  # 当前时间
    t_end = time.time()
    time_cost = (t_end - t_start)/3600
    print(f'---------End training------time cost: {time_cost:.4f}h',flush=True)
    save_paras.add_logs('The training ends at the time= ' + str(time_now_end))
    save_paras.add_logs('Training time cost =' + str(time_cost))

    # --------------------draw figure----------------------------
    fig, axs = plt.subplots(ncols=2, nrows=1)

    ax = axs.flatten()

    ax[0].plot(np.arange(len(Loss_cache )) , Loss_cache)

    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('loss value')
    ax[0].grid(True)

    ax[1].plot(np.arange(len(Lr_list)), Lr_list)
    ax[1].set_xlabel('Iteration')
    ax[1].set_ylabel('learning rate')
    ax[1].grid(True)

    fig.tight_layout()
    fig_name = 'loss_lr-Epoch.png'
    fig_path = directory_model + '/' + fig_name
    plt.savefig(fig_path) # save figure
    plt.show()
    # plt.plot(Loss_cache, label='loss')
    # plt.legend()
    # plt.xlabel('Iteration')
    # plt.ylabel('Loss')
    # plt.grid()
    # plt.show()
    var_dict = {'loss': Loss_cache, 'Lr': Lr_list}
    fullpath = directory_model + '/' + 'training_record.mat'
    spio.savemat(fullpath, var_dict)

    print('-----------------------------Start Test---------------------------------',flush=True)
    s_hat, f_all, batch_H, batch_Fopt,batch_Fbb = tst_model()
    spio.savemat(dat_file_name,
                 {"H": batch_H, "Fopt": batch_Fopt,  "Fbb": batch_Fbb, "f": s_hat,
                  'f_all': f_all})

    print('-----------------------------Test Finished---------------------------------')




