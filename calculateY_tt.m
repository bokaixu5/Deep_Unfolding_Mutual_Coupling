function Y_tt = calculateY_tt(N,f,mu, xyz_dma,epsilon)
k = 2*pi*f*sqrt(epsilon*mu);
Ga_e2 = @(r,rhat) ((norm(r-rhat)^2 - (r(3)-rhat(3))^2)/norm(r-rhat)^2 - ...
                    1i*(norm(r-rhat)^2 - 3*(r(3)-rhat(3))^2)/(norm(r-rhat)^3*k)...
                    - (norm(r-rhat)^2 - 3*(r(3)-rhat(3))^2)/(norm(r-rhat)^4*k^2))...
                    *exp(-1i*k*norm(r-rhat))/(4*pi*norm(r-rhat));
for n1=1:N
    for n2=1:N
    if n1 ~= n2
        Y_tt(n1,n2) = 1i * 2 * 2*pi*f * epsilon *Ga_e2(xyz_dma(n1,:),xyz_dma(n2,:));
    else
        
        Y_tt(n1,n2) = k * 2*pi*f * epsilon / (3 * pi);
    end
    end
end
end