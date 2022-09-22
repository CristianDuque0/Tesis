#! /usr/bin/octave

%
%  Documentación
%
%  Calular el 98 en función de Tw,Ts,t1,t2
%
% agregar Tw,... a los parametros de la función

% Realizado por:
function vector_dir(dir, genero)
	
	% verificación de número y tipo de parámetros

	Tw = 25;                % Duracion de la trama de analisis (ms)
	Ts = 10;                % Cambio de trama de analisis (ms)
	t1= 90;
	t2= 91;
	archivos=[dir,"*.wav"];
	wav_file =glob(archivos);
	file1 =glob(archivos);
	w=length(wav_file);
	vec_caract_bam=zeros(10,w*98);   %% acá

	for canciones=1:w
		wav=wav_file(canciones,1);
		wav=char(wav);
		vec_caract = vector_tex_tim(wav,t=[t1 t2],Tw,Ts); %%% cálculo del vector de características
		vec_caract_cur=[vec_caract];
		vec_caract_cur=vec_caract_cur';
		tam=size(vec_caract_cur)(1);
		for canc= 1:tam
			printf("%0.10f,",vec_caract_cur(canc,:));   
			%printf("%s",char(wav_file(canciones,1)));
			%printf("%s", ",",genero); 
			printf("\n");

		end
		printf("0,0,0,0,0,0,0,0,0,\n")
		%		

	end	
