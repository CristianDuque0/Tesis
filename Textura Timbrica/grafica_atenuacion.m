t=54:0.2:64;

[vector_bam,fs]=wavread('canciones/bambuco3/OjosDeYoNoSeQue.wav',[60 62]);
[vector_bam,fs]=wavread('canciones/bambuco3/OjosDeYoNoSeQue.wav',[54 64]*fs);
vector_bam=vector_bam(:,1);
t1=54:1/fs:64;

atenVbam=(tex_tim_bam_paraAtenuacion(1:51,11))/1000000;

%[vector_cum,fs]=wavread('canciones/cumbia3/ColombiaTierraQuerida.wav',[60 62]);
%[vector_cum,fs]=wavread('canciones/cumbia3/ColombiaTierraQuerida.wav',[60 62]*fs);
%vector_cum=vector_cum(:,1);
%t2=60:1/fs:62;
%
%atenVcum=(tex_tim_4canciones_4(22:42,11))/1000000;
%
%[vector_por,fs]=wavread('canciones/porro3/Atlantico.wav',[60 62]);
%[vector_por,fs]=wavread('canciones/porro3/Atlantico.wav',[60 62]*fs);
%vector_por=vector_por(:,1);
%t3=60:1/fs:62;
%
%atenVpor=(tex_tim_4canciones_4(43:63,11))/1000000;
%
%[vector_va,fs]=wavread('canciones/vallenato3/LaGotaFria.wav',[60 62]);
%[vector_va,fs]=wavread('canciones/vallenato3/LaGotaFria.wav',[60 62]*fs);
%vector_va=vector_va(:,1);
%t4=60:1/fs:62;
%
%atenVval=(tex_tim_4canciones_4(64:84,11))/1000000;

figure, subplot(2,1,1), plot (t,atenVbam,'-;Bambuco;h'), title('Atenuacion Espectral');xlabel('Tiempo (s)'); ylabel('Frecuencia (MHz)');
%subplot(2,1,2), plot(t1,vector_bam), title('Cancion Bambuco');xlabel('Tiempo (s)'); ylabel('Amplitud');

%figure, subplot(2,1,1), plot (t,atenVcum), title('Atenuacion Espectral');xlabel('Tiempo (s)'); ylabel('Frecuencia (MHz)');
%subplot(2,1,2), plot(t2,vector_cum), title('Cancion Cumbia');xlabel('Tiempo (s)'); ylabel('Amplitud');
%
%figure, subplot(2,1,1), plot (t,atenVpor), title('Atenuacion Espectral');xlabel('Tiempo (s)'); ylabel('Frecuencia (MHz)');
%subplot(2,1,2), plot(t3,vector_por), title('Cancion Porro');xlabel('Tiempo (s)'); ylabel('Amplitud');
%
%figure, subplot(2,1,1), plot (t,atenVval), title('Atenuacion Espectral');xlabel('Tiempo (s)'); ylabel('Frecuencia (MHz)');
%subplot(2,1,2), plot(t4,vector_va), title('Cancion Vallenato');xlabel('Tiempo (s)'); ylabel('Amplitud');