% ## Copyright (C) Ingenieria Electronica. 2016 UdeC
% ##
% 
% ## -*- texinfo -*-
% ## @deftypefn  {Function File} {@var{Ztm} @var{Ztv}} = cruce (@var{Tw}, @var{Ts}, @var{fs}, @var{vector})
% ## Calcula el cruce por cero en el dominio del tiempo de  @var{vector}.
% ## 
% ## @table @asis
% ## @item @var{Tw}:
% ## Tiempo de duración de los marcos.
% ##
% ## @item @var{Ts}:
% ## Tiempo de superposición de los marcos.
% ##
% ## @item @var{fs}:
% ## Muestras por segundo a la cual se muestreó @var{vector}.
% ##
% ## @item @var{vector}:
% ## Señal de audio a analizar.
% ## @end table
% ## 
% ## La salida es:
% ##
% ## @table @asis
% ## @item @var{Ztm}:
% ## Valor medio del cruce por cero.
% ##
% ## @item @var{Ztv}:
% ## Varianza del cruce por cero.
% ## @end table
% ##
% ## Ver: "Musical Genre Clasification of Audio Signals". Tzanetakis, Cook. 2002
% ##
% ## @seealso{centroide, flujo, atenuacion, low_e}
% ## @end deftypefn
% 
% ## Author: Sebastián Muñoz <sebasmg.reggae@gmail.com>
% ## Created: 2016
%function [Ztm Ztv] = cruce(Tw, Ts, fs, vector)
function [Zt_sum] = cruce(Tw, Ts, fs, vector)

Nw = round( 1E-3*Tw*fs );    % frame duration (samples)
Ns = round( 1E-3*Ts*fs );    % frame shift (samples)

nfft = 2^nextpow2( Nw );     % length of FFT analysis 

% Framing and windowing (frames as columns)
frames = vec2frames(vector, Nw, Ns);

% Magnitude spectrum computation (as column vectors)
%MAG = abs(fft(frames,nfft,1)); 

m=size(frames);
t=1:m(2);
n=1:m(1);
n=n';

Zt=zeros(1,m(2));

for i=2:m(1)
    m1=frames(i,:);
    m2=frames(i-1,:);
    s1=signo(m1);
    s2=signo(m2);
    Zt(i,:)=abs(s1-s2);
end

Zt_sum=1/2*(sum(Zt));
%Ztm=mean(Zt_sum);
%Ztv=var(Zt_sum);

end

function s = signo(x)
s=x>0;
end
