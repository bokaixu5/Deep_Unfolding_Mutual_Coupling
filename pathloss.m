function BETAA=pathloss(dist)


%Pathloss exponent
alpha = 3.67;

%Average channel gain in dB at a reference distance of 1 meter.
constantTerm = -30.5;
channelGaindB = constantTerm - alpha*10*log10(dist);
%Noise figure at the BS (in dB)
noiseFigure = 0;
%Communication bandwidth
B = 20e6;
%Compute noise power
noiseVariancedBm = -174 + 10*log10(B) + noiseFigure;
channelGainOverNoise = channelGaindB - noiseVariancedBm;
channelGain=10^(channelGainOverNoise/10);

BETAA=sqrt(channelGain);