close all;
clear;
clc;
r=[3,5,10];
for ri=1:3
    for numm=0:1:10
        %% configuration
        wavelength=1;  %  *** wavelength
        k0=2*pi/wavelength;  % wavenumber
        Volume=1;
        delta=0.1;      % step ***
        size=1;         % xoy range ***
        rr=r(ri);
        theta(numm+1)=-pi/2+numm*(pi/10);       
        dz=rr*sin(theta(numm+1));
        dx=rr*cos(theta(numm+1));
        sigma = 1;  %noise
        %% sampling the source area xoy plane
        x00= 0;
        y00= 0;
        z00= -size:delta:size;
        z00=z00+delta/2;
        
        Nsz=length(z00);
        %  position of source point
        index_s=0;
        for i=1:Nsz   %  Z
            index_s=index_s+1;
            sou(1,index_s)=x00(1);
            sou(2,index_s)=y00(1);
            sou(3,index_s)=z00(i);
        end
        N=index_s;
        
        %% sampling the observation area xoy plane
        xo= dx;
        yo= 0;
        zo= -size+dz:delta:size+dz;  % observation z
        zo=zo+delta/2;
        
        Nz=length(zo);
        %  position of observation point
        index_f=0;
        for i=1:Nz   %  Z
            index_f=index_f+1;
            pos(1,index_f)=xo(1);
            pos(2,index_f)=yo(1);
            pos(3,index_f)=zo(i);
        end
        N_p=index_f;
        %% dyadic Green function
        G = Dyadic_Green(N_p,N,pos,sou,k0,Volume);
        R=G*G';
        R=R/norm(R);
        %% EDOF
        EDof(ri,numm+1)=(trace(R)/norm(R,'fro'))^2;
        %% Mutua information
        Rnoise=sigma^2*eye(3*N);
        I(ri,numm+1)=Mutua_information(R,Rnoise);
    end
end

figure(1)
plot(theta,I(1,:),'-o','Linewidth',1.5,'markersize',5,'color','blue');
hold on
plot(theta,I(2,:),'--','Linewidth',1.5,'markersize',5,'color','black');
hold on
plot(theta,I(3,:),'-.','Linewidth',1.5,'markersize',5,'color','red');
hold on
grid on
title('Mutual information','FontSize',14);
xlabel(' $\theta[rad]$','Interpreter','latex');
ylabel('Mutual information[Nats/Hz]','FontSize',14);
set(gca,'xlim',[-pi/2,pi/2]);%设置x轴坐标范围
set(gca,'xtick',-pi/2:pi/10:pi/2);%设置x轴坐标间隔
set(gca,'XTickLabel',{'-\pi/2','-2\pi/5','-3\pi/10','-\pi/5','-\pi/10','0','\pi/10','\pi/5','3\pi/10','2\pi/5','\pi/2'});%\pi=π
legend('\lambda = 1m,R = 3m','\lambda = 1m, R = 5m','\lambda = 1m, R = 10m');

figure(2)
plot(theta,EDof(1,:),'-o','Linewidth',1.5,'markersize',5,'color','blue');
hold on
plot(theta,EDof(2,:),'--','Linewidth',1.5,'markersize',5,'color','black');
hold on
plot(theta,EDof(3,:),'-.','Linewidth',1.5,'markersize',5,'color','red');
hold on
grid on
title('EM EDOF','Interpreter','latex','FontSize',14);
xlabel(' $\theta[rad]$','Interpreter','latex','FontSize',14);
ylabel('EM EDOF','Interpreter','latex','FontSize',14);
set(gca,'xlim',[-pi/2,pi/2]);%设置x轴坐标范围
set(gca,'xtick',-pi/2:pi/10:pi/2);%设置x轴坐标间隔
set(gca,'XTickLabel',{'-\pi/2','-2\pi/5','-3\pi/10','-\pi/5','-\pi/10','0','\pi/10','\pi/5','3\pi/10','2\pi/5','\pi/2'});%\pi=π
legend('\lambda = 1m,R = 3m','\lambda = 1m, R = 5m','\lambda = 1m, R = 10m');