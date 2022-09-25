%Definicion de variables
Tw = 25;                % Duracion de la trama de analisis (ms)
Ts = 10;                % Cambio de trama de analisis (ms)
t1= 90;
t2= 91;
mat=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];

%Reconocimiento de las canciones
for i=1:4
	if (i==1)
		wav_file =glob('Canciones/Bambuco/*.wav');
		file1 =glob('Canciones/Bambuco/*.wav');
		w=length(wav_file);
		vec_caract_pf=zeros(10,w*98);
	elseif (i==2)
		wav_file =glob('Canciones/Carranga/*.wav');
		file2 =glob   ('Canciones/Carranga/*.wav');
		w=length(wav_file);
		vec_caract_pv=zeros(10,w*98);
	elseif(i==3)
		wav_file =glob('Canciones/Pasillo/*.wav');
		file3 =glob('Canciones/Pasillo/*.wav');
		w=length(wav_file);
		vec_caract_ot=zeros(10,w*98);
	end

  %Calculo del vector de caracteristicas
	for canciones=1:w
		wav=wav_file(canciones,1);
		wav=char(wav);
		vec_caract = vector_tex_tim(wav,t=[t1 t2],Tw,Ts);
    if (i==1)
				vec_caract_pf =[vec_caract];
				vec_caract_pf=vec_caract_pf';
        vec_caract_pf=[mean(vec_caract_pf),std(vec_caract_pf)];
        mat_pf(canciones,1:18)=[vec_caract_pf];
        mat(canciones+1,1:18)=[vec_caract_pf];
				long=size(vec_caract_pf)(1);
        filename_bam=sprintf('Graficas/Bambuco/textura_timbrica_b.csv');
        csvwrite(filename_bam,[mat_pf]);
			for canc= 1:long
				printf("%0.10f,",vec_caract_pf(canc,:));
				printf("%s",char(wav_file(canciones,1)));
				printf("%s", ",bambuco");
				printf("\n");
			end
		elseif (i==2)
			vec_caract_pv=[vec_caract];
			vec_caract_pv=vec_caract_pv';
      vec_caract_pv=[mean(vec_caract_pv),std(vec_caract_pv)]
      mat_pv(canciones,1:18)=[vec_caract_pv];
      mat(canciones+21,1:18)=[vec_caract_pv];
			long=size(vec_caract_pv)(1);
      filename_pv=sprintf('Graficas/Carranga/textura_timbrica_c.csv');
      csvwrite(filename_pv,[mat_pv]);
			for canc= 1:long
				printf("%0.10f,",vec_caract_pv(canc,:));
				printf("%s",char(wav_file(canciones,1)));
				printf("%s", ",carranga");
				printf("\n");
			end
		elseif (i==3)
			vec_caract_ot=[vec_caract];
			vec_caract_ot=vec_caract_ot';
      vec_caract_ot=[mean(vec_caract_ot),std(vec_caract_ot)]
      mat_ot(canciones,1:18)=[vec_caract_ot];
      mat(canciones+41,1:18)=[vec_caract_ot];
			long=size(vec_caract_ot)(1);
      filename_ot=sprintf('Graficas/Pasillo/textura_timbrica_p.csv');
      csvwrite(filename_ot,[mat_ot]);
			for canc= 1:long
				printf("%0.10f,",vec_caract_ot(canc,:));
				printf("%s",char(wav_file(canciones,1)));
				printf("%s", ",pasillo");
				printf("\n");
			end
		end
	end
end

%Generacion de los csv
csvwrite('Clasificadores/vector.csv',[mat]);
cell2csv('graficas/nombres.csv',[file1;file2;file3]);
