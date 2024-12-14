clc;
clear all;
train_size = 1000;
valid_size = 100;
test_size = 100;
N = 8;     %Num of users
M = 36;     %Tx antennas
B=1;
H_train = zeros(B, train_size, N, M);
H_valid = zeros(B, valid_size, N, M);
H_test = zeros(B, test_size, N, M);
for train_s=1:train_size
H_train(B,train_s,:,:)=function_gen_data(N);
end
for valid_s=1:valid_size
    H_valid(B,valid_s,:,:)=function_gen_data(N);
end
for test_s=1:test_size
    H_test(B,test_s,:,:)=function_gen_data(N);
end
save('H_train.mat', 'H_train'); % 通过逗号分隔多个变量名称
save('H_valid.mat', 'H_valid'); % 通过逗号分隔多个变量名称
save('H_test.mat', 'H_test'); % 通过逗号分隔多个变量名称