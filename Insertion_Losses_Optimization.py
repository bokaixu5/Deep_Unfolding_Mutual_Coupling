import torch
import scipy.io
import numpy as np
# read the mat
Data_Y_tt = scipy.io.loadmat('Y_tt.mat')
# convert to tensor
Y_tt = torch.tensor(Data_Y_tt['Y_tt'])
print('Y_tt:', Y_tt.shape, Y_tt[0][0])

# read the mat
Data_Y_st = scipy.io.loadmat('Y_st.mat')
# convert to tensor
Y_st = torch.tensor(Data_Y_st['Y_st'])
print('Y_st:', Y_st.shape, Y_st[0][0])

# read the mat
Data_Y_ss = scipy.io.loadmat('Y_ss_1.mat')
# convert to tensor
Y_ss = torch.tensor(Data_Y_ss['Y_ss_1'])
print('Y_ss:', Y_ss.shape, Y_ss[0][0])


# read the mat
Data_tilde_Yr = scipy.io.loadmat('tilde_Yr.mat')
# convert to tensor
tilde_Yr = torch.tensor(Data_tilde_Yr['tilde_Yr'])
print('tilde_Yr:', tilde_Yr.shape, tilde_Yr[0][0])

# read the mat (random)
Data_Y_rs = scipy.io.loadmat('Y_rs.mat')
# convert to tensor
Y_rs = torch.tensor(Data_Y_rs['Y_rs'])
print('Y_rs:', Y_rs.shape, Y_rs[0][0])

#paramter
N                   = 6
L_mu                = 6
Y_intrinsic_source  = 35.3387
Ys_im               = torch.diag(torch.ones(N*L_mu))
Ys_im.requires_grad = True
print(Ys_im[0][0], 'and', Ys_im[1][1])

##################################################MAIN
lr     = 0.755

#lr=0.75
epochs = 400000
#epochs = 100000
loss_res = []
n_features=N*L_mu
velocity = torch.zeros(n_features)
momentum=0.9
for i in range(epochs):

  # Y_p
  Y_p   = Y_tt - torch.transpose(Y_st, 0, 1) @ torch.linalg.inv(1j*Ys_im + Y_ss) @ Y_st
  Y_in  = Y_p * torch.eye(N)
  Gamma = (Y_in - Y_intrinsic_source * torch.eye(N)) @ torch.linalg.inv(Y_in + Y_intrinsic_source * torch.eye(N))
  #Gamma=torch.zeros(N,N)+1j*torch.zeros(N,N)
  #Gamma=Gamma.to(torch.complex128)
  #print(Gamma)
  Y_q   = torch.linalg.inv( torch.eye(N) - torch.conj(torch.transpose(Gamma, 0, 1)) @ Gamma ) @ Y_p

  # H_dma
  H_dma     = tilde_Yr @ ( Y_rs @ torch.linalg.inv(1j*Ys_im + Y_ss) @ Y_st )
  H_dma_p   = torch.conj( torch.transpose(H_dma, 0, 1)) @ torch.linalg.inv(H_dma @ torch.conj(torch.transpose(H_dma, 0, 1)) )
  H_dma_p_c = torch.conj( torch.transpose(H_dma_p, 0, 1) )

  #Loss      = torch.trace( ((Y_q).real+1j*0) @ H_dma_p @ H_dma_p_c )
  #Loss      = torch.trace(  (H_dma_p @ Y_q @ H_dma_p_c).real )
  Loss = torch.trace((H_dma_p_c  @ Y_q @H_dma_p ).real)
  # Loss      = (Loss).real
  runingloss = Loss.item()
  loss_res.append(runingloss)
  if i % 2000 == 0:
    print(f"Epoch {i}: Loss = {Loss.item()}")

    #lr = lr * 0.99


  Loss.backward(retain_graph=True)
  # Ys_im.data = Ys_im.data - lr * Ys_im.grad

  velocity = momentum * velocity +   lr* torch.diag(Ys_im.grad, 0)
  Ys_im_1 =torch.diag(Ys_im.data, 0)-velocity

  #Ys_im_1 = torch.diag(Ys_im.data, 0)- lr * torch.diag(Ys_im.grad, 0)

  Ys_im.data = torch.diag(Ys_im_1)
  Ys_im.grad.zero_()


def ZF(Pmax_t, Heq_fd, sigma2_x, Y_tt):
  H_fd_dagger = Heq_fd.T.conj() @ torch.linalg.inv(Heq_fd @ Heq_fd.T.conj())
  num         = H_fd_dagger
  # de          = torch.sqrt( torch.trace(sigma2_x / 2 * H_fd_dagger.T.conj() @ ((Y_tt).real + 1j*0) @  H_fd_dagger) )
  de          = torch.sqrt( torch.trace(sigma2_x / 2 * (H_fd_dagger.T.conj() @ Y_tt @  H_fd_dagger).real) )
  B_fd        = num / de
  return B_fd

def SE_calculation(Heq, B, sigma2_n, sigma2_x):
  K, M = Heq.shape
  #print(K)
  c    = torch.zeros(K)
  for idx1 in range(K):
    ds = torch.abs(Heq[idx1, :] @ (B[:, idx1].unsqueeze(1)))**2
    interference = 0
    for idx2 in range(K):
      if idx2 != idx1:
        interference += torch.abs(Heq[idx1, :] @ (B[:, idx2].unsqueeze(1)))**2
        sinr_k = ds / (sigma2_n/sigma2_x + interference)
        c[idx1] = torch.log2(1 + sinr_k)

  C = torch.sum(c)
  return C

Pmax_t    = 1
sigma2_x  = 1
#sigma2_n  = Pmax_t / (10 ** (15/10) )
H_dma     = tilde_Yr @ ( Y_rs @ torch.linalg.inv(1j*Ys_im + Y_ss) @ Y_st )


B_DMA    = ZF(Pmax_t, H_dma, sigma2_x, Y_q)

#result   = SE_calculation(H_dma, B_DMA, sigma2_n, sigma2_x)
snr = np.arange(-10, 21, 5)
Pmax_t = 1
for idx1 in range(len(snr)):
    sigma2 = Pmax_t / (10 ** (snr[idx1] / 10))
    sum1 = SE_calculation(H_dma, B_DMA, sigma2, sigma2_x)

    print('----------------')
    print('SNR:',snr[idx1])
    print('DMA:',sum1)
    print('----------------')