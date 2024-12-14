function [ FRF,FBB ] = MO_AltMin( Pmax_t,Fopt, NRF,sigma2_x,Y_tt_fd )

[Nt, Ns] = size(Fopt);
y = [];
Niter=2;
FRF = exp( 1i*unifrnd(0,2*pi,Nt,NRF) );
%while(isempty(y) || abs(y(1)-y(2))>1e-3)
for i=1:Niter
    FBB = pinv(FRF) * Fopt;
    y(1) = norm(Fopt - FRF * FBB,'fro')^2;
    [FRF, y(2)] = sig_manif(Fopt, FRF, FBB);
end



  numerator = sqrt(Pmax_t) * FBB;  % 分子部分
    denominator = sqrt(trace(real(sigma2_x / 2 * (FRF*FBB)' * Y_tt_fd * (FRF*FBB))));  % 分母部分
    FBB = numerator / denominator;  % 结果
end