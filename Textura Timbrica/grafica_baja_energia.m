t=38:0.1:41;

%[vector_bam,fs]=wavread('canciones/bambuco3/OjosDeYoNoSeQue.wav',[60 62]);
%[vector_bam,fs]=wavread('canciones/bambuco3/OjosDeYoNoSeQue.wav',[60 62]*fs);
%vector_bam=vector_bam(:,1);
%t1=60:1/fs:62;
%
%lowVbam=tex_tim_4canciones_4(1:21,19);

[vector_cum,fs]=wavread('canciones/cumbia3/ColombiaTierraQuerida.wav',[60 62]);
[vector_cum,fs]=wavread('canciones/cumbia3/ColombiaTierraQuerida.wav',[37.95 40.95]*fs);
vector_cum=vector_cum(:,1);
t2=38:1/fs:41;

lowVcum=tex_tim_cum_paraLow(1:31,19);

%[vector_por,fs]=wavread('canciones/porro3/Atlantico.wav',[60 62]);
%[vector_por,fs]=wavread('canciones/porro3/Atlantico.wav',[60 62]*fs);
%vector_por=vector_por(:,1);
%t3=60:1/fs:62;
%
%lowVpor=tex_tim_4canciones_4(43:63,19);
%
%[vector_va,fs]=wavread('canciones/vallenato3/LaGotaFria.wav',[60 62]);
%[vector_va,fs]=wavread('canciones/vallenato3/LaGotaFria.wav',[60 62]*fs);
%vector_va=vector_va(:,1);
%t4=60:1/fs:62;
%
%lowVval=tex_tim_4canciones_4(64:84,19);

%figure, subplot(2,1,1), plot (t,lowVbam), title('Baja Energia');xlabel('Tiempo (s)'); ylabel('Porcentaje (%)');
%subplot(2,1,2), plot(t1,vector_bam), title('Cancion Bambuco'); xlabel('Tiempo (s)'); ylabel('Amplitud');
%print Baja_energía_bam.pdf

figure, subplot(2,1,1), plot (t,lowVcum,'g--;Cumbia;h'), title('Baja Energia');xlabel('Tiempo (s)'); ylabel('Porcentaje (%)');
subplot(2,1,2), plot(t2,vector_cum), title('Cancion Cumbia'); xlabel('Tiempo (s)'); ylabel('Amplitud');
print Baja_energía_cum.pdf

%figure, subplot(2,1,1), plot (t,lowVpor), title('Baja Energia');xlabel('Tiempo (s)'); ylabel('Porcentaje (%)');
%subplot(2,1,2), plot(t3,vector_por), title('Cancion Porro'); xlabel('Tiempo (s)'); ylabel('Amplitud');
%print Baja_energía_por.pdf
%
%figure, subplot(2,1,1), plot (t,lowVval), title('Baja Energia');xlabel('Tiempo (s)'); ylabel('Porcentaje (%)');
%subplot(2,1,2), plot(t4,vector_va), title('Cancion Vallenato'); xlabel('Tiempo (s)'); ylabel('Amplitud');
%print Baja_energía_va.pdf