function vec_caract = vector_tex_tim(nombre,t=[x1 x2],Tw,Ts)
if !ischar(nombre)
	error ('no es un nombre de archivo');
end


alpha = 0.97;           % coeficiente de pre-enfasis
M = 20;                 % Cantidad de canales de filtros
C = 12;                 % Numero de coeficientes cepstrales
L = 22;                 % Parametro cepstral sine lifter
LF = 300;               % Limite de frecuencia inferior (Hz)
HF = 3700;              % Limite de frecuencia superior (Hz)
        samples = [1, 100];
        [vector,fs]=audioread(nombre,samples); % lectura del audio para saber la frecuencia fs
        %samples= [1, t]
        [vector,fs]=audioread(nombre,t*fs);
        vector=vector(:,1);

%        [Ctm Ctv]=centroide(Tw,Ts,fs,vector);
%        [Aem Aev]=atenuacion(Tw,Ts,fs,vector);
%        [Ftm Ftv]=flujo(Tw,Ts,fs,vector);
%        [Ztm Ztv]=cruce(Tw,Ts,fs,vector);
%        [MFCCs,FBEs,frames]=mfcc(vector,fs,Tw,Ts,alpha,@hamming,[LF HF],M,C+1,L);
%        size(MFCCs)
%        x=MFCCs(1:5,:);
%        MFCCsm=mean(x');
%        MFCCsv=var(x');
%        L_E = low_e(Tw, Ts, fs, vector);
%	       vec_caract =[Ctm Aem Ftm Ztm MFCCsm Ctv Aev Ftv Ztv MFCCsv L_E];



        [Ct]=centroide(Tw,Ts,fs,vector);
        [Ae]=atenuacion(Tw,Ts,fs,vector);
        [Ft]=flujo(Tw,Ts,fs,vector);
        [Zt]=cruce(Tw,Ts,fs,vector);
        [MFCCs,FBEs,frames]=mfcc(vector,fs,Tw,Ts,alpha,@hamming,[LF HF],M,C+1,L);
        MFCCs5=MFCCs(1:5,:);
        L_E = low_e(Tw, Ts, fs, vector);
        vec_caract =[Ct;Ae;Ft;Zt;MFCCs5];

end



        

