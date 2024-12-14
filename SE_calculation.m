function C = SE_calculation(Heq,B,sigma2_n,sigma2_x, Y_tt)
[K,M] = size(Heq);
c = zeros(1,K);
for idx1 = 1:1:K
    ds = abs(Heq(idx1,:)*B(:,idx1))^2;
    int = 0;
    for idx2 = 1:1:K
        if idx2 ~= idx1
            int = int + abs(Heq(idx1,:)*B(:,idx2))^2;
        end
    end
    sinr_k = ds/(sigma2_n/sigma2_x+int);
    %sinr_k
    c(idx1) = log2(1+sinr_k);
    
end
%H_fd_dagger = Heq' * inv(Heq * Heq');
%sigma2_x/(sigma2_n*trace(sigma2_x/2*real(H_fd_dagger'*Y_tt*H_fd_dagger)))

C = sum(c);