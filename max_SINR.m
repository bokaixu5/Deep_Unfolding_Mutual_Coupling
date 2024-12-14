

function SE = max_SINR(H,sigma2,Pow,sigma2_x, Y_tt_fd)



[K,M] = size(H);
pre = H' * inv(H* H');
%p=H';
for k=1:K
    numerator = pre(:,k);  % 分子部分
    denominator = sqrt(trace(sigma2_x/ 2 *pre(:,k)' *  real(Y_tt_fd) * pre(:,k)));  % 分母部分
    P(:,k) = numerator / denominator;  % 结果
    P_FD_1=sigma2_x/2 * trace(real( P(:,k)' * Y_tt_fd *  P(:,k)))

end



L=1;

signal_RZF = zeros(K,L);
interf_RZF = zeros(K,L,K,L);
for idx1 = 1:1:K
    signal_RZF(idx1,L) = abs(H(idx1,:)*P(:,idx1))^2;
    int = 0;
    for idx2 = 1:1:K
        if idx2 ~= idx1
          interf_RZF(idx2,L,idx1,L)=abs(H(idx1,:)*P(:,idx2))^2;
        end
    end
   
    interf_RZF(idx1,L,idx1,L) = 0;
  
   
end

Pmax=Pow;
SE = functionPowerOptimization_prodSINR(signal_RZF, interf_RZF,Pmax,sigma2);