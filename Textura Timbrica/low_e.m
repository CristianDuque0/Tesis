% Copyright (C) Ingenieria Electronica. 2016 UdeC
%  
% -*- texinfo -*-
% @deftypefn  {Function File} @var{L_E} = low_e (@var{Tw}, @var{Ts}, 
% @var{fs}, @var{vector})
% Calcula las caracteristicas de baja energia de  @var{vector}.
%  
% @table @asis
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
% @item @var{L_E}:
% Caracteristicas de baja energía.
% @end table
% 
% Ver: "Musical Genre Clasification of Audio Signals". Tzanetakis, 
% Cook. 2002
% 
% @seealso{centroide, cruce, atenuacion, flujo}
% @end deftypefn
% 
% Author: Sebastián Muñoz <sebastianmg.info@gmail.com>
% Created: 2016

function L_E = low_e(Tw, Ts, fs, vector)

Nw = round( 1E-3*Tw*fs );    % duración del marco en muestras
Ns = round( 1E-3*Ts*fs );    % duración del cambio de marco en muestras

nfft = 2^nextpow2( Nw );     % Longitud del análisis FFT 

% Enmarcado y ventanamiento (marcos como columnas)
frames = vec2frames(vector, Nw, Ns);

% Cálculo del espectro de magnitud (como vectores columna)
MAG = abs(fft(frames,nfft,1)); 

f=size(frames);
f=f(2);
Et=zeros(1,f);

for i=1:f
    d=(MAG(:,i)).^2;
    P=sum(d);         % Potencia
    RMS=sqrt(P);      % Valor RMS
    Et(i)=RMS*(Tw*1e-3);   % Energia para cada una de las ventanas 
%     de analisis
end

Ep=mean(Et);          % Energia promedio de la ventana de textura
unos=Et<Ep;
L_E=sum(unos)/f*100;   % Porcentaje de ventanas de baja energia
end
