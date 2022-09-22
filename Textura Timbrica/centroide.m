% Copyright (C) Ingenieria Electronica. 2016 UdeC
% 
% -*- texinfo -*-
% @deftypefn  {Function File} {@var{Ctm} @var{Ctv}} = centroide (@var{Tw}, 
% @var{Ts}, @var{fs}, @var{vector})
% Calcula el centroide espectral de  @var{vector} 
% 
% @table @asis
% @item @var{Tw}:
% Tiempo de duración de los marcos.
% 
% @item @var{Ts}:
% Tiempo de superposición de los marcos.
% 
% @item @var{fs}:
% Muestras por segundo a la cual se muestrea @var{vector}.
% 
% @item @var{vector}:
% Señal de audio a analizar.
% @end table
%  
% La salida es:
% 
% @table @asis
% @item @var{Ctm}:
% Valor medio del centroide espectral.
% 
% @item @var{Ctv}:
% Varianza del centroide espectral.
% @end table
% 
% Ver: "Musical Genre Clasification of Audio Signals". Tzanetakis,
% Cook. 2002
% 
% @seealso{atenuacion, flujo, cruce, low_e}
% @end deftypefn
% 
% Author: Sebastián Muñoz <sebastianmg.info@gmail.com>
% Created: 2016

%function[Ctm Ctv]=centroide(Tw,Ts,fs,vector)
function[Ct]=centroide(Tw,Ts,fs,vector)

%% verificar parametros

Nw=round(1E-3*Tw*fs);    % duración del marco en muestras
Ns=round(1E-3*Ts*fs);    % duración del cambio de marco en muestras

nfft=2^nextpow2(Nw);     % Longitud del análisis FFT

% Enmarcado y ventanamiento (marcos como columnas)
frames=vec2frames(vector,Nw,Ns);
% Calculo del espectro de magnitud (como vectores columna)
MAG=abs(fft(frames,nfft,1));

m=size(MAG);
t=1:m(2);
n=1:m(1);
n=n';
mag_n=(MAG).*(n);

num=sum(mag_n);
den=sum(MAG);
Ct=num./den; % Centroide Espectral

%Ctm=mean(Ct);
%Ctv=var(Ct);

end

%! test
%! Tw=25
%! Ts=10
%! fs=44000
%! vector=[4 8 6 2; 2 1 3 4; 1 1 1 1; 2 4 4 5]
%!assert (centroide (Tw,Ts,fs,vector),[3 4])
