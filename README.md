# Deep Unfolding Beamforming and Power Control Designs for Multi-Port Matching Networks


This simulation code package is mainly used to reproduce the results of the following paper [1]:

[1] Bokai Xu, Jiayi Zhang, Qingfeng Lin, Huahua Xiao, Yik-Chung Wu, and Bo Ai, “Deep Unfolding Beamforming and Power Control Designs for Multi-Port Matching Networks”, IEEE Transactions on Wireless Communications, to appear, 2024.

https://ieeexplore.ieee.org/document/10791449

### Abstract

The key technologies of sixth generation (6G), such as ultra-massive multiple-input multiple-output (MIMO), enable intricate interactions between antennas and wireless propagation environments. As a result, it becomes necessary to develop joint models that encompass both antennas and wireless propagation channels. To achieve this, we utilize the multi-port communication theory, which considers impedance matching among the source, transmission medium, and load to facilitate efficient power transfer. Specifically, we first investigate the impact of insertion loss, mutual coupling, and other factors on the performance of multi-port matching networks. Next, to further improve system performance, we explore two important deep unfolding designs for the multi-port matching networks: beamforming and power control, respectively. For the hybrid beamforming, we develop a deep unfolding framework, i.e., projected gradient descent (PGD)-Net based on unfolding projected gradient descent. For the power control, we design a deep unfolding network, graph neural network (GNN) aided alternating optimization (AO)Net, which considers the interaction between different ports in optimizing power allocation. Numerical results verify the necessity of considering insertion loss in the dynamic metasurface antenna (DMA) performance analysis. Besides, the proposed PGD-Net based hybrid beamforming approaches approximate the conventional model-based algorithm with very low complexity. Moreover, our proposed power control scheme has a fast run time compared to the traditional weighted minimum mean squared error (WMMSE) method.
