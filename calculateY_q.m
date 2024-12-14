function Y_q = calculateY_q(Gamma, Y_p)
    N = size(Gamma, 2);  % 确定N的大小，假设Gamma是N x M的矩阵
    I_N = eye(N);  % N x N的单位矩阵
    Y_q = inv(I_N - Gamma' * Gamma) * Y_p;  % 计算Y_q
end