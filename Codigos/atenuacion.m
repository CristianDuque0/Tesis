% Copyright (C) Ingenieria Electronica. 2016 UdeC
%  
% -*- texinfo -*-
% @deftypefn  {Function File} {@var{Aem} @var{Aev}} = atenuacion (@var{Tw},
% @var{Ts}, @var{fs}, @var{vector})
% Calcula la atenuación espectral de  @var{vector}.
% % @table @asis
% @item @var{Tw}:
% Tiempo de duración de los marcos.
% 
% @item @var{Ts}:
% Tiempo de superposición de los marcos.
% 
% @item @var{fs}:
% Muestras por segundo a la cual se muestreó @var{vector}.
% 
% @item @var{vector}:
% Señal de audio a analizar.
% @end table
%  
% La salida es:
% 
% @table @asis
% @item @var{Aem}:
% Valor medio de la atenuación espectral.
% 
% @item @var{Aev}:
% Varianza de la atenuación espectral.
% @end table
% 
% Ver: "Musical Genre Clasification of Audio Signals". Tzanetakis, 
% Cook. 2002
% 
% @seealso{centroide, flujo, cruce, low_e}
% @end deftypefn
% 
% Author: Sebastián Muñoz <sebastianmg.info@gmail.com>
% Created: 2016

%function[Aem Aev]=atenuacion(Tw,Ts,fs,vector)
function[Ae]=atenuacion(Tw,Ts,fs,vector)

Nw=round(1E-3*Tw*fs);    % duración del marco en muestras
Ns=round(1E-3*Ts*fs);    % duración del cambio de marco en muestras

nfft=2^nextpow2(Nw);     % Longitud del análisis FFT

% Enmarcado y ventanamiento (marcos como columnas)
frames=vec2frames(vector,Nw,Ns);

% Cálculo del espectro de magnitud (como vectores columna)
MAG=abs(fft(frames,nfft,1)); 

Ae=0.85.*(sum(MAG));     % Atenuación Espectral

%Aem=mean(Ae);
%Aev=var(Ae);

end
