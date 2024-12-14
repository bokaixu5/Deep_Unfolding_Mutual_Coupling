clc;
clear all;
%% Parameters
f = 10e9;                        % Frequency of operation
c = 299792458;                   % Light speed in vacuum
mu = 4*pi*1E-7;                  % Vacuum permeability
epsilon = 1/ (c^2 * mu);         % Vacuum permittivity
lambda = c/f;                    % Wavelength
k = 2*pi/lambda;                 % Wavenumber
a = 0.73*lambda;                 % Width of waveguides (only TE_10 mode)
b = 0.17*lambda;                 % Height of waveguides (only TE_10 mode)
channel_type = 1;                % Type of channel:
P_RF=0.2;

P_PS=0.03;
% 0 -> LoS
% 1 -> Rayleigh
N = 6;                           % Number of RF chains / waveguides
Lmu =6;                         % Number of elements per waveguide
wvg_spacing = lambda;            % Spacing between waveguides
elem_spacing = 0.5*lambda;           % Spacing between the elements
l = 1;                           % Length of dipoles -> just normalization
M = 6;                           % Number of static users
Plot_topology = 1;               % Boolean to plot the chosen setup

Y_s = diag(1i*randn(N*Lmu,1));   % Load admittances of DMA element
% Has to be a diagonal matrix of
% N*Lmu x N*Lmu (total number of elements)

Y_intrinsic_source = 35.3387;    % Intrinsic impedance of source
% matched to waveguide of width a  =
% 0.73*lambda and height b = 0.17*lambda
sigma2_x=1;

%% DMA and users coordinates
site_xyz = [0 0 10];             % [x y z] coordinates of bottom right
% corner of DMA
S_mu = (Lmu+1)*elem_spacing;     % Length of waveguides

% Coordinates of DMA elements and RF chains
[ant_xyz, rf_xyz] = Topologies_DMA(site_xyz,N, Lmu, wvg_spacing,...
    elem_spacing, S_mu, a, b, Plot_topology);
% Users positions (In this example, they are set randomly)
x_lim = [-20 20];
y_lim = [20 60];
% x_lim = [-60 60];
% y_lim = [60 180];
user_xyz = [x_lim(1)+(x_lim(2)-x_lim(1))*rand(M,1) ...
    y_lim(1)+(y_lim(2)-y_lim(1))*rand(M,1) 1.5*ones(M,1)];


%% Calculation of Admittances

% Calculating Y_tt, Y_st and Y_ss according to Eqs. (35)-(42)
[Y_tt, Y_st, Y_ss] = DMA_admittance(f, a, b, l, S_mu, ant_xyz, ...
    rf_xyz, mu, epsilon);
Y_ss_1=diag(Y_ss);
Y_ss_1=diag(Y_ss_1);
% Calculating Y_rr according to Eqs. (44)-(46)
Y_rr = Coupling_Dipoles(f, l, user_xyz, mu, epsilon);

% Choosing Y_r (load admittance of users) as conjugate of self-admittance
Y_r = Y_rr'.*eye(M);

% Calculation of Y_rs (Wireless channel)
Y_rs = GenChannel(channel_type, lambda, ant_xyz, user_xyz);

%% Equivalent channel according to Eq. (60)
Heq = eye(M)/(Y_r + Y_rr) * (Y_rs/(Y_s + Y_ss)*Y_st);
%Heq = eye(M)/(Y_r + Y_rr) * (Y_rs*Y_st);
%% Computing received, transmitted and supplied power:

% Computing approximate reflection coefficient assuming no cross-waveguide
% coupling
Y_p = Y_tt - (Y_st.' / (Y_s + Y_ss)) * Y_st;
Y_in = eye(N) .* Y_p;
Gamma = (Y_in - eye(N)*Y_intrinsic_source) / (Y_in + eye(N)*Y_intrinsic_source);

% Choosing random beam
%B = randn(N,M) + 1i*randn(N,M);

% Computing received, transmit, and supplied power
% x = randn(M,1);
% y = Heq * B * x;
%
% P_r = 1/2 * real(Y_r) * abs(y).^2;
% P_t = 1/2 * real(x' * B' * Y_p * B * x);
% P_s = 1/2 * real(x' * B' * ((eye(N) - Gamma' * Gamma) \ Y_p) * B * x);

%% Channel of FD mMIMO
tilde_Yr = sqrt(real(Y_r)/2)/ (Y_r + Y_rr);
%tilde_Yr = eye(M)/ (Y_r + Y_rr);
Y_rt=Y_rs;
Heq_fd=-tilde_Yr*Y_rt;
%% Channel of DMA
Heq_DMA=sqrt(real(Y_r)/2)*Heq;
N1=7;
%% Plot SE_SNR
Hnum = 1;
snr = -5:5:20;
Pmax_t = 1;
sumratedata_FD = zeros(1,length(snr));
sumratedata_DMA = zeros(1,length(snr));
sumratedata_HBF = zeros(1,length(snr));

Y_tt_fd=calculateY_tt(N*Lmu,f,mu, ant_xyz,epsilon);
Y_q = calculateY_q(Gamma, Y_p);
%Y_tt_fd=calculateY_tt(N_1,f,mu, ant_xyz,epsilon);
 R_rzf=Pmax_t+N*Lmu*P_RF;
 R_DMA=Pmax_t+N*P_RF;
 R_LSR=Pmax_t+N*P_RF+N*N*Lmu*P_PS;
for idx1 = 1:1:length(snr)
    disp(['SNR: ' num2str(snr(idx1))])
    sigma2_n=Pmax_t/(10^(snr(idx1)/10));
    SR_FD = 0;
    SR_DMA = 0;
    SR_HBF = 0;
    for idx2 = 1:1:Hnum
        % ZF
        B_FD = ZF(Pmax_t, Heq_fd, sigma2_x, Y_tt_fd);
        SR_FD = SR_FD + SE_calculation(Heq_fd,B_FD,sigma2_n,sigma2_x, Y_tt_fd);
        [FRF, FBB ]=function_LSR(Pmax_t,B_FD, N1,sigma2_x,Y_tt_fd);
        B_HBF=FRF*FBB;
        B_DMA = ZF(Pmax_t, Heq_DMA, sigma2_x, Y_q);
        %x = randn(M,1);
        P_DMA=sigma2_x/2 * trace(real(B_DMA' * Y_q * B_DMA ));

        P_HBF=sigma2_x/2 * trace(real(B_HBF' * Y_tt_fd * B_HBF));
        P_FD=sigma2_x/2 * trace(real(B_FD' * Y_tt_fd * B_FD));
        %P_r1 = 1/2*trace((real(B_FD'*Y_tt_fd*B_FD)))
        %P_r2 = abs(Heq_DMA*B_DMA*x).^2

        SR_HBF = SR_HBF + SE_calculation(Heq_fd,B_HBF,sigma2_n,sigma2_x);
        SR_DMA = SR_DMA + SE_calculation(Heq_DMA,B_DMA,sigma2_n,sigma2_x);
    end
    sumratedata_FD(idx1) = SR_FD/Hnum;
    sumratedata_DMA(idx1) = SR_DMA/Hnum;
    sumratedata_HBF(idx1) = SR_HBF/Hnum;
end

save('Y_tt.mat', 'Y_tt'); % 通过逗号分隔多个变量名称
save('Y_st.mat', 'Y_st'); % 通过逗号分隔多个变量名称
save('Y_ss.mat', 'Y_ss'); % 通过逗号分隔多个变量名称
save('tilde_Yr.mat', 'tilde_Yr'); % 通过逗号分隔多个变量名称
save('Y_rs.mat', 'Y_rs'); % 通过逗号分隔多个变量名称
 save('Y_tt_fd.mat', 'Y_tt_fd'); % 通过逗号分隔多个变量名称
save('Y_ss_1.mat', 'Y_ss_1'); % 通过逗号分隔多个变量名称

figure
plot(snr,sumratedata_FD,'r-*','LineWidth',2)
hold on
plot(snr,sumratedata_DMA,'b-*','LineWidth',2)
plot(snr,sumratedata_HBF,'k-*','LineWidth',2)
plot(snr,sumratedata_DMA_opt,'g-*','LineWidth',2)


plot(snr,sumratedata_FD,'r-*','LineWidth',2)
hold on
plot(snr,sumratedata_DMA,'b-*','LineWidth',2)
plot(snr,sumratedata_HBF,'k-*','LineWidth',2)
%plot(snr,sumratedata_DMA_opt,'g-*','LineWidth',2)

xlabel('SNR (dB)')
ylabel('Sum Rate (bps/Hz)')
legend('FD','DMA','HBF','Location','best')
grid on
