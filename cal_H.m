function H_dagger = cal_H(tilde_Yr, Y_rs, Ys_im , Y_st,Y_ss)
% 定义 H 函数
H =  tilde_Yr *Y_rs * inv(1j * Ys_im + Y_ss) * Y_st ;
% 定义 H_dagger 函数
H_dagger =H' * inv(H * H') ;
  
end