t=60:0.1:62;

[vector_bam,fs]=wavread('canciones/bambuco3/OjosDeYoNoSeQue.wav',[60 62]);
[vector_bam,fs]=wavread('canciones/bambuco3/OjosDeYoNoSeQue.wav',[60 62]*fs);
vector_bam=vector_bam(:,1);
t1=60:1/fs:62;

centVbam=tex_tim_4canciones_4(1:21,10);

[vector_cum,fs]=wavread('canciones/cumbia3/ColombiaTierraQuerida.wav',[60 62]);
[vector_cum,fs]=wavread('canciones/cumbia3/ColombiaTierraQuerida.wav',[60 62]*fs);
vector_cum=vector_cum(:,1);
t2=60:1/fs:62;

centVcum=tex_tim_4canciones_4(22:42,10);

[vector_por,fs]=wavread('canciones/porro3/Atlantico.wav',[60 62]);
[vector_por,fs]=wavread('canciones/porro3/Atlantico.wav',[60 62]*fs);
vector_por=vector_por(:,1);
t3=60:1/fs:62;

centVpor=tex_tim_4canciones_4(43:63,10);

[vector_va,fs]=wavread('canciones/vallenato3/LaGotaFria.wav',[60 62]);
[vector_va,fs]=wavread('canciones/vallenato3/LaGotaFria.wav',[60 62]*fs);
vector_va=vector_va(:,1);
t4=60:1/fs:62;

centVval=tex_tim_4canciones_4(64:84,10);

%figure, subplot(2,1,1), plot (t,centVbam), title('Centroide Espectral');xlabel('Tiempo (s)'); ylabel('Amplitud');
%subplot(2,1,2), plot(t1,vector_bam), title('Cancion Bambuco');xlabel('Tiempo (s)'); ylabel('Amplitud');
%
%figure, subplot(2,1,1), plot (t,centVcum), title('Centroide Espectral');xlabel('Tiempo (s)'); ylabel('Amplitud');
%subplot(2,1,2), plot(t2,vector_cum), title('Cancion Cumbia');xlabel('Tiempo (s)'); ylabel('Amplitud');
%
%figure, subplot(2,1,1), plot (t,centVpor), title('Centroide Espectral');xlabel('Tiempo (s)'); ylabel('Amplitud');
%subplot(2,1,2), plot(t3,vector_por), title('Cancion Porro');xlabel('Tiempo (s)'); ylabel('Amplitud');

figure
%subplot(2,1,1), 
plot (t,centVval,'m-.;Vallenato;h'), title('Centroide Espectral');xlabel('Tiempo (s)'); ylabel('Amplitud'); 
%subplot(2,1,2), plot(t4,vector_va), title('Cancion Vallenato');xlabel('Tiempo (s)'); ylabel('Amplitud');