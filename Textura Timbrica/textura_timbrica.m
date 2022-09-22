Tw = 25;                % Duracion de la trama de analisis (ms)
Ts = 10;                % Cambio de trama de analisis (ms)
t1= 90;
t2= 91;

for i=1:4
	if (i==1)
	    wav_file =glob('/home/jestebanza/Aprendizaje/audicion computacional/canciones/bambuco/*.wav');
	    file1 =glob('/home/jestebanza/Aprendizaje/audicion computacional/canciones/bambuco/*.wav');
	    w=length(wav_file);
            vec_caract_bam=zeros(10,w*98);
	elseif (i==2)
	    wav_file =glob('/home/jestebanza/Aprendizaje/audicion computacional/canciones/currulao/*.wav');
	    file2 =glob('/home/jestebanza/Aprendizaje/audicion computacional/canciones/currulao/*.wav');
	    w=length(wav_file);
            vec_caract_cur=zeros(10,w*98);
	elseif(i==3)
	    wav_file =glob('/home/jestebanza/Aprendizaje/audicion computacional/canciones/joropo/*.wav');
	    file3 =glob('/home/jestebanza/Aprendizaje/audicion computacional/canciones/joropo/*.wav');
	    w=length(wav_file);
            vec_caract_jor=zeros(10,w*98);
	elseif (i==4)
	    wav_file =glob('/home/jestebanza/Aprendizaje/audicion computacional/canciones/vallenato/*.wav');
	    file4 =glob('/home/jestebanza/Aprendizaje/audicion computacional/canciones/vallenato/*.wav');
	    w=length(wav_file);
            vec_caract_va=zeros(10,w*98);
	end
	
       for canciones=1:w
            wav=wav_file(canciones,1);
            wav=char(wav);
	    vec_caract = vector_tex_tim(wav,t=[t1 t2],Tw,Ts);

            if    (i==1)
                 vec_caract_bam =[vec_caract];
		 vec_caract_bam=vec_caract_bam';
		 long=size(vec_caract_bam)(1);
 		for canc= 1:long
 			printf("%0.10f,",vec_caract_bam(canc,:));   
			printf("%s",char(wav_file(canciones,1)));
			printf("%s", ",bambuco"); 
			printf("\n");
		end
            elseif (i==2)
                 vec_caract_cur=[vec_caract];
		 vec_caract_cur=vec_caract_cur';
		 long=size(vec_caract_cur)(1);
 		for canc= 1:long
 			printf("%0.10f,",vec_caract_cur(canc,:));   
			printf("%s",char(wav_file(canciones,1)));
			printf("%s", ",currulao");
			printf("\n");
		end
            elseif (i==3)
                 vec_caract_jor=[vec_caract];
		 vec_caract_jor=vec_caract_jor';
		 long=size(vec_caract_jor)(1);
 		for canc= 1:long
 			printf("%0.10f,",vec_caract_jor(canc,:));   
			printf("%s",char(wav_file(canciones,1)));
			printf("%s", ",joropo");
			printf("\n");
		end
            elseif (i==4)
                 vec_caract_va=[vec_caract];
		 vec_caract_va=vec_caract_va';
		 long=size(vec_caract_va)(1);
 		for canc= 1:long
 			printf("%0.10f,",vec_caract_va(canc,:));   
			printf("%s",char(wav_file(canciones,1)));
			printf("%s", ",vallenato");
			printf("\n");
		end
             end

       end            
end


csvwrite('textura_timbrica_bam.csv',[vec_caract_bam]);
csvwrite('textura_timbrica_cur.csv',[vec_caract_cur]);
csvwrite('textura_timbrica_jor.csv',[vec_caract_jor]);
csvwrite('textura_timbrica_va.csv',[vec_caract_va]);
csvwrite('textura_timbrica.csv',[vec_caract_bam;vec_caract_cur;vec_caract_jor;vec_caract_va]);
cell2csv('nombres.csv',[file1;file2;file3;file4]);
