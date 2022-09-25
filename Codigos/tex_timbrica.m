clear all; close all; clc;

% -----------------------------------------------------------------------
% Esta implementacion nos permite generar un vector de 19 elementos para
% la caracterstica musical "textura timbrica", donde se utilizan las 
% funciones creadas teniendo en cuenta el algoritmo de Tzatetakis y Cook.
% Finalmente estos vectores hallados para cada una de las canciones de
% los generos musicales, en este caso colombianos, ingresados en la 
% variable 'wav_file', por supuesto teniendo en cuenta su direccion de
% ubicacion, determinada en la variable 'extensiones', se guardan en un
% archivo .xlsx.
% -----------------------------------------------------------------------

% Autores: Sebastian Munoz Garcia <sebastianmg.info@gmail.com>
%          Leonardo Rodriguez Mujica <lrodriguez@mail.unicundi.edu.co>
% Creado: 2016

% tamano total en segundos del fragmento de la cancin a analizar
t_min=60;
t_max=65

Tw = 25;                % Duracion de la trama de analisis (ms)
Ts = 10;                % Cambio de trama de analisis (ms)
alpha = 0.97;           % coeficiente de pre-enfasis
M = 20;                 % Cantidad de canales de filtros
C = 12;                 % Numero de coeficientes cepstrales
L = 22;                 % Parametro cepstral sine lifter
LF = 300;               % Limite de frecuencia inferior (Hz)
HF = 3700;              % Limite de frecuencia superior (Hz)

extensiones={'canciones/bambuco/';'canciones/cumbia/';...
    'canciones/porro/';'canciones/vallenato/'};
e=size(extensiones);
s=e(1);

% creacin de cada vector de caracterstica (por cancin)
vec_caract_bam=zeros(21,19);
vec_caract_cum=zeros(21,19);
vec_caract_por=zeros(21,19);
vec_caract_va=zeros(21,19);

for i=1:s
    if (i==1)
        wav_file={'Candilejas'};% ingreso del archivo de audio de bambuco
%             {'ALaReina' 'BrisasDelPamplonita' 'ChaticaLinda'...
%             'CorazoncitoMio' 'ElBarcino' 'ElGuatecano' 'ElPifano'...
%             'ElRepublicano' 'ElSapiroco' 'ElSotareno' 'FlorDeCafe'...
%             'Humorisimo' 'LaGuanena' 'LosOpitas' 'ManzanitaColorada'...
%             'MiCasta' 'MuchachaDeRisaLoca' 'OjoAlToro'...
%             'OjosDeYoNoSeQue' 'Palonegro' 'Raza' 'Rosalinda'...
%             'Sanjuanero' 'SanPedroEnElEspinal' 'Surenita'...
%             'TaberneroAmigo' 'TaitaQuillacinga' 'TreceDeJunio'};  
        w=length(wav_file);
      elseif (i==2)
        wav_file={'OnTabas'};% ingreso del archivo de audio de cumbia
%             {'AguaceroEMayo' 'AnoNuevo' 'ColombiaTierraQuerida'...
%             'CumbiaBanquena' 'CumbiaCampesina' 'CumbiaDelCaribe'...
%             'CumbiaDePescadores' 'CumbiaFonsequera' 'CumbiaSabanera'...
%             'DosDeFebrero' 'ElMaranon' 'ElPescador' 'ElZabroson'...
%             'GaitaDeLasFlores' 'JulioMoreno' 'LaCumbiaCienaguera'...
%             'LaCumbiaDelAmor' 'LaPolleraColora' 'LaRebuscona'...
%             'LasMirlas' 'LaZenaida' 'MarthaLaReina' 'MiCumpleanos'...
%             'Minarete' 'Monalinda' 'PlinioGuzman'...
%             'TierraSanta'};  
        w=length(wav_file);
      elseif (i==3)
        wav_file={'prueba'};% ingreso del archivo de audio de porro
%             {'Atlantico' 'CarmenDeBolivar' 'ElSabroson'...
%             'GolfoDeMorrosquillo' 'LaLunaYLaPlaya' 'LaTelefonista'...
%             'LaVacaVieja' 'Macondo' 'Merecumbe' 'Monalinda'...
%             'PajaroPicon' 'Peluquero' 'SanCarlos' 'SanFernando'...
%             'Tina'};   
        w=length(wav_file);
      elseif (i==4)
        wav_file={'TalamoDeRosas'};% ingreso del archivo de audio de vallenato
%             {'AliciaAdorada' 'CuandoCasiTeOlvidaba' 'ElBozal'...
%             'ElParrandon' 'LaGotaFria' 'LaMuerteDeAbelAntonio'...
%             'LaMujerYLaPrimavera' 'LaPlata' 'LaPuyaRegional'...
%             'LaSanguijuela' 'LaVillaDelRosario' 'MeDejoSolito'...
%             'NinaBonita' 'PuyaInstrumental' 'VeniVeni'};  
        w=length(wav_file);
    end
    ext=strtrim(extensiones(i,:));
    t=[60 61]; % vector del tamao de la ventana de textura dentro del audio
    traslape=0.1;
    tam=t_min:traslape:t_max;
    tam=length(tam);
    t=t-traslape;
    for j=1:tam
%        mtiempos=[60 70;65 75;70 80;75 85;80 90;85 95;90 100;95 105;100 110;105 115;110 120];
%        tiempo=mtiempos(j,:);
        t=t+traslape;
        wav=wav_file;
        wav=char(wav);
        ext=char(ext);
        WAVFILE=[ext wav];
        [vector,fs,nbits]=wavread(WAVFILE,100); % lectura del audio para saber la frecuencia fs
        [vector,fs,nbits]=wavread(WAVFILE,t*fs);
        vector=vector(:,1);
        [Ctm Ctv]=centroide(Tw,Ts,fs,vector);
        [Aem Aev]=atenuacion(Tw,Ts,fs,vector);
        [Ftm Ftv]=flujo(Tw,Ts,fs,vector);
        [Ztm Ztv]=cruce(Tw,Ts,fs,vector);
        [MFCCs,FBEs,frames]=mfcc(vector,fs,Tw,Ts,alpha,@hamming,[LF HF],M,C+1,L);
        size(MFCCs)
        x=MFCCs(1:5,:);
        MFCCsm=mean(x');
        MFCCsv=var(x');
        L_E = low_e(Tw, Ts, fs, vector);
          if (i==1)
              vec_caract_bam(j,:)=[Ctm Aem Ftm Ztm MFCCsm Ctv Aev Ftv Ztv MFCCsv L_E];
              elseif (i==2)
              vec_caract_cum(j,:)=[Ctm Aem Ftm Ztm MFCCsm Ctv Aev Ftv Ztv MFCCsv L_E];
              elseif (i==3)
              vec_caract_por(j,:)=[Ctm Aem Ftm Ztm MFCCsm Ctv Aev Ftv Ztv MFCCsv L_E];
              elseif (i==4)
              vec_caract_va(j,:)=[Ctm Aem Ftm Ztm MFCCsm Ctv Aev Ftv Ztv MFCCsv L_E];
          end
    end            
end

vec_caract_bam;
vec_caract_cum;
vec_caract_por;
vec_caract_va;

csvwrite('textura_timbrica.xlsx',[vec_caract_bam vec_caract_cum vec_caract_por vec_caract_va]);
