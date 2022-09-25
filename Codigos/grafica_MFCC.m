t=152:0.1:158;

[vector_bam,fs]=wavread('canciones/bambuco3/OjosDeYoNoSeQue.wav',[60 62]);
[vector_bam,fs]=wavread('canciones/bambuco3/OjosDeYoNoSeQue.wav',[152 158]*fs);
vector_bam=vector_bam(:,1);
t1=152:1/fs:158;

COEF1=tex_tim_bam_paraMFCC(1:61,14);

COEF2=tex_tim_bam_paraMFCC(1:61,15);

COEF3=tex_tim_bam_paraMFCC(1:61,16);

COEF4=tex_tim_bam_paraMFCC(1:61,17);

COEF5=tex_tim_bam_paraMFCC(1:61,18);

%figure
%subplot(2,1,1), plot(t,COEF1), title('Coeficiente 1 MFCC');xlabel('Tiempo (s)'); ylabel('Amplitud');
%subplot(2,1,2), plot(t1,vector_bam), title('Cancion Bambuco');xlabel('Tiempo (s)'); ylabel('Amplitud');

figure
subplot(2,1,1), plot(t,COEF2,'-;Bambuco;h'), title('Coeficiente 2 MFCC'); xlabel('Tiempo (s)'); ylabel('Amplitud');
subplot(2,1,2), plot(t1,vector_bam), title('Cancion Bambuco');xlabel('Tiempo (s)'); ylabel('Amplitud');

%figure
%subplot(2,1,1), plot(t,COEF3), title('Coeficiente 3 MFCC');xlabel('Tiempo (s)'); ylabel('Amplitud');
%subplot(2,1,2), plot(t1,vector_bam), title('Cancion Bambuco');xlabel('Tiempo (s)'); ylabel('Amplitud');
%
%figure
%subplot(2,1,1), plot(t,COEF4), title('Coeficiente 4 MFCC');xlabel('Tiempo (s)'); ylabel('Amplitud');
%subplot(2,1,2), plot(t1,vector_bam), title('Cancion Bambuco');xlabel('Tiempo (s)'); ylabel('Amplitud');
%
%figure
%subplot(2,1,1), plot(t,COEF5), title('Coeficiente 5 MFCC');xlabel('Tiempo (s)'); ylabel('Amplitud');
%subplot(2,1,2), plot(t1,vector_bam), title('Cancion Bambuco');xlabel('Tiempo (s)'); ylabel('Amplitud');