function [FRF,FBB, count] = SDR_AltMin(Pmax_t,Fopt, NRF,sigma2_x,Y_tt_fd)

% randomly generate FRF
[Nt, Ns, K] = size(Fopt);
FRF = [];
for i = 1:NRF
    FRF = blkdiag(FRF, exp(sqrt(-1) * unifrnd (0,2*pi,[Nt/NRF,1])));
end
% FRF = FRF;

count = 0;
y = [];
FBB = zeros(NRF, Ns, K);
Niter=2;
%while(isempty(y) || abs(y(1)-y(2))>1e-3)
 for i=1:Niter   
    % fix FRF, optimize FBB
    y = [0,0];
    temp1 = zeros(Nt, NRF);
    for k = 1:K
        A1 = diag([ones(1,Ns*NRF),0]);
        A2(Ns*NRF+1,Ns*NRF+1) = 1;
        
        temp = kron(eye(Ns),FRF);
        C = [temp'*temp,-temp'*vec(Fopt(:,:,k));-vec(Fopt(:,:,k))'*temp,vec(Fopt(:,:,k))'*vec(Fopt(:,:,k))];
        
        cvx_begin quiet
        variable X(Ns*NRF+1,Ns*NRF+1) hermitian
        minimize(real(trace(C*X)));
        subject to
        trace(A1*X) == NRF*Ns;
        trace(A2*X) == 1;
        X == hermitian_semidefinite(Ns*NRF+1);
        cvx_end
        
        [V,D] = eig(X);
        [value,num] = max(diag(D));
        x = sqrt(value)*V(:,num);
        FBB(:,:,k) = reshape(x(1:Ns*NRF),NRF,Ns);
        
        y(1) = y(1) + norm(Fopt(:,:,k)-FRF*FBB(:,:,k),'fro')^2;
        temp1 = temp1 + Fopt(:,:,k) * FBB(:,:,k)';
    end
    
    % fix FBB, optimize FRF
    %FRF = exp(1i * angle(temp1));
    
    for i = 1:Nt
        m = ceil(i*NRF/Nt);
        FRF(i,m) = exp( sqrt(-1) * angle(temp1(i,m)) );
    end
    
    for k = 1:K
        y(2) = y(2) + norm(Fopt(:,:,k) - FRF*FBB(:,:,k),'fro')^2;
    end
    
    count = count + 1;
end

for k = 1:K
    FBB(:,:,k) = sqrt(Ns) * FBB(:,:,k) / norm(FRF * FBB(:,:,k),'fro');
    if abs(norm(FRF * FBB(:,:,k),'fro')^2 - Ns) > 1e-4
        error('check power constraint !!!!!!!!!!!!')
    end
end
  numerator = sqrt(Pmax_t) * FBB;  % 分子部分
    denominator = sqrt(trace(real(sigma2_x / 2 * (FRF*FBB)' * Y_tt_fd * (FRF*FBB))));  % 分母部分
    FBB = numerator / denominator;  % 结果
end