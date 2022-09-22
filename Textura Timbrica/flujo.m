% Copyright (C) Ingenieria Electronica. 2016 UdeC
% 
% -*- texinfo -*-
% @deftypefn  {Function File} {@var{Ftm} @var{Ftv}} = flujo (@var{Tw},
% @var{Ts}, @var{fs}, @var{vector})
% Calcula el flujo espectral de  @var{vector}.
%  
% @table @asis
% @item @var{Tw}:
% Tiempo de duración de los marcos.
% 
% @item @var{Ts}:
% Tiempo de superposición de los marcos.
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
% @item @var{Ftm}:
% Valor medio del flujo espectral.
% 
% @item @var{Ftv}:
% Varianza del flujo espectral.
% @end table
% 
% Ver: "Musical Genre Clasification of Audio Signals". Tzanetakis, 
% Cook. 2002
% 
% @seealso{centroide, cruce, atenuacion, low_e}
% @end deftypefn
% 
% Author: Sebastián Muñoz <sebastianmg.info@gmail.com>
% Created: 2016
%function [Ftm Ftv] = flujo(Tw, Ts, fs, vector)
function [Ft_sum] = flujo(Tw, Ts, fs, vector)

Nw = round( 1E-3*Tw*fs );    % duración del marco en muestras
Ns = round( 1E-3*Ts*fs );    % duración del cambio de marco en muestras

nfft = 2^nextpow2( Nw );     % Longitud del análisis FFT

% Enmarcado y ventanamiento (marcos como columnas)
frames = vec2frames(vector, Nw, Ns);

% Cálculo del espectro de magnitud (como vectores columna)
MAG = abs(fft(frames,nfft,1)); 

m=size(MAG);
t=1:m(2);
n=1:m(1);
n=n';

Ft=zeros(m);
mayor=max(MAG);

for i=2:m(2)
    Nt_1=(MAG(:,i))/mayor(i);       % se normaliza con el mismo numero???
    Nt_2=(MAG(:,i-1))/mayor(i-1);
    Ft(:,i)=(Nt_1-Nt_2).^2;      % Flujo Espectral
end

Ft_sum=sum(Ft);

%Ftm=mean(Ft_sum);
%Ftv=var(Ft_sum);

end
