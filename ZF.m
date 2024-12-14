function B_fd = ZF(Pmax_t, Heq_fd, sigma2_x, Y_tt)
[K,M] = size(Heq_fd);
% H_fd_dagger = Heq_fd' * inv(Heq_fd * Heq_fd');
% numerator = sqrt(Pmax_t) * H_fd_dagger;  % 分子部分
% denominator = sqrt(trace(sigma2_x/ 2 * real((H_fd_dagger)' * Y_tt * H_fd_dagger)));  % 分母部分
% B_fd = numerator / denominator;  % 结果
% B_fd=normalize(H_fd_dagger);
% end
H_fd_dagger = Heq_fd' * inv(Heq_fd * Heq_fd');
%H_fd_dagger = Heq_fd' ;
numerator = sqrt(Pmax_t) * H_fd_dagger;  % 分子部分
denominator = sqrt(trace(sigma2_x/ 2 * real((H_fd_dagger)' * Y_tt * H_fd_dagger)));  % 分母部分
B_fd = numerator / denominator;  % 结果
%B_fd=normalize(Heq_fd');
% for k=1:M
%     numerator =sqrt(Pmax_t) * H_fd_dagger(k,:);  % 分子部分
%     denominator = sqrt(trace(sigma2_x/ 2 * real(H_fd_dagger(k,:)' * Y_tt(k,k) * H_fd_dagger(k,:))));  % 分母部分
%     B_fd(k,:) = numerator / denominator;  % 结果
%     %P_FD_1=sigma2_x/2 * trace(real( p_norm(:,k)' * Y_tt_fd *  p_norm(:,k)))
% 
% end



end