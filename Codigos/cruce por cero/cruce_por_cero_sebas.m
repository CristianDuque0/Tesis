close all;clear all;clc;

t_min=54;
t_max=64;
traslape=0.1;  % tamaño de traslape en segundos
vent=[54 55];    % vector de tamaño de ventana en segundos
vent=vent-traslape;
tiempo=t_min:traslape:t_max;
t=length(tiempo);

[v,fs]=wavread('OjosDeYoNoSeQue.wav',[60 62]);
[v,fs]=wavread('OjosDeYoNoSeQue.wav', [t_min t_max]*fs);
v=v(:,1);
t1=t_min:1/fs:t_max;

Z=zeros(t,1);


for i=1:t
    vent=vent+traslape;
    [vector,fs]=wavread('OjosDeYoNoSeQue.wav', vent*fs);
    vector=vector(:,1);
    Z(i,1)= feature_zcr(vector);
end

figure
subplot(2,1,1), plot(tiempo,Z,'-;Bambuco;');title('Cruce por Cero');xlabel('Tiempo (s)'); ylabel('Amplitud')
subplot(2,1,2), plot(t1,v);title('Cancion Bambuco');xlabel('Tiempo (s)'); ylabel('Amplitud')