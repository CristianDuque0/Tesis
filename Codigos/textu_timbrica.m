Tw = 25;                % Duracion de la trama de analisis (ms)
Ts = 10;                % Cambio de trama de analisis (ms)
t1= 40;
t2= 41;

for i=1:4
	if (i==1)
	    wav_file =glob('/home/jestebanza/Documentos/TESIS/canciones/bambucos/*.wav');
	    w=length(wav_file);
            vec_caract_bam=zeros(w,19);
	elseif (i==2)
	    wav_file =glob('/home/jestebanza/Documentos/TESIS/canciones/cumbias/*.wav');
	    w=length(wav_file);
            vec_caract_jor=zeros(w,19);
	elseif(i==3)
	    wav_file =glob('/home/jestebanza/Documentos/TESIS/canciones/vallenatos/*.wav');
	    w=length(wav_file);
            vec_caract_va=zeros(w,19);
	elseif (i==4)
	    wav_file =glob('/home/jestebanza/Documentos/TESIS/canciones/porros/*.wav');
	    w=length(wav_file);
            vec_caract_por=zeros(w,19);
	
        end

       for canciones=1:w
            wav=wav_file(canciones,1);
            wav=char(wav);
	    vec_caract = vector_tex_tim(wav,t=[t1 t2],Tw,Ts);
            if    (i==1)
                 vec_caract_bam(canciones,:)=[vec_caract];
            elseif (i==2)
                 vec_caract_jor(canciones,:)=[vec_caract];
            elseif (i==3)
                 vec_caract_va(canciones,:)=[vec_caract];
            elseif (i==4)
                 vec_caract_por(canciones,:)=[vec_caract];
            end
       end    

       
vec_caract_bam;
vec_caract_jor;
vec_caract_va;
vec_caract_por;


csvwrite('textura_timbrica_bam.csv',[vec_caract_bam]);
csvwrite('textura_timbrica_jor.csv',[vec_caract_jor]);
csvwrite('textura_timbrica_va.csv',[vec_caract_va]);
csvwrite('textura_timbrica_por.csv',[vec_caract_por]);
csvwrite('textura_timbrica.csv',[vec_caract_bam; vec_caract_jor; vec_caract_va; vec_caract_por]);       
end


