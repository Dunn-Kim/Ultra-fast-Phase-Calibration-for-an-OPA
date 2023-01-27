%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Periodic Optical Phased Array - MATLAB
%%%Fraunhofer Diffraction Model from "Optics", 4th edition - Eugene Hecht
%%%Created at Oct.2022 by DO HYUNG KIM (Aaron), dohyung.aaron.kim@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


N = 8;               %number of channels
angle = 10;            %steering angle
theta = -90:0.1:90;   %steering range
samp = size(theta);   %sample rate of farfield
lambda = 1.55e-6;     %wavelength in air
lambda_span = 0;      %wavelength range
l = 20e-6;            %wavegudide length
b = 1e-6;             %waveguide width
t = 0.5e-6;           %waveguide thickness
ep = 1;               %ep=epsilon=sourch strength(source intensity)
n = 1;                %refractive index of farfield material(air)
k = 2*pi/(lambda/n);  %propagation constant in farfield material
temp = (sind(theta));

X = 0;                          %x-start-position of channels
a = 3e-6;                       %y-position-difference between adjacent channels
Y = a.*(0:N-1);                 %y-positions of channels, unifomrly spaced
                                %coordinates (x,y,z):
%x is the light propagation direction. 
%Waveguides are, placed parallel to x-axis, on the xy-plane. 
%z is the height.
%length, width, thickness of waveguides are along x,y,z respectively.
%Since this simulation is for farfield, we dont need the bigger length.
m = ceil(a*sind(angle)/lambda); %the number of lambda in the path difference when steered by 'angle'
delph = k*a*sind(angle)-2*pi*m; %Applied phase difference, same to all channels
Phase = delph.*(0:N-1);         %Applied phase of channels, arithmetic progression

beta=(k*b*sind(theta))/2;      %phase difference between adjacent channels
EF=(sin(beta)./beta).^2;  %Element factor
EF(round(samp(2)/2))=1; %%%preventing NaN made by sin(0)/0 = 1
alpha=(k*a*sind(theta))/2;     %phase difference between adjacent channels at farfield
alpha(round(samp(2)/2))=0;     %%%preventing NaN made by *sind(0)
AF = 0;
Phase = [pi, pi, pi, pi, pi, pi, pi, pi];
for i=1:1:N
%%%Phase(i)=Phase(i)+rand*2*pi; %여기서 rand가 initial phase 자리
    AF=AF+exp(1i*(2*alpha*(i-1)-Phase(i)));
end
AF=abs(AF);
% AF=(sin(N*(alpha-delph/2))./sin(alpha-delph/2)).^2;
% AF(round(samp(2)/2))=(N)^2; %%preventing NaN made by sin(0)/0 = 1

                              %Array factor
%Applied phase difference(delph) is only applied on alpha even though we 
%must insert them into mode(source) as beta in MODE. 
%It is because in the fomula alpha&beta are not connected but in MODE.
%In other words, the fomula does not take into account the physical 
%connection between waveguides and the air. As such, the phase differences
%(beta) made in channels are not delivered to the air(alpha).
%Therefore, you should apply delph only on "alpha" in the formula (MATLAB),
%and on "beta" in MODE.
I = EF.*AF;                   %Intensity(MATLAB farfield)

figure;                   %plot in different window
tiledlayout(2,3);         %divide figure window into 2x3
nexttile([2 2]);          %I occupy 2x2
plot(theta,I);            %plot Intensity(MATLAB farfield)
xlim([-90 90]);           %set the range of x axis
xticks(-90:10:90);        %set the ticks of x axis
title('MATLAB farfield'); %set the name of plot
xlabel('theta');          %set the name of x axis
ylabel('Intensity');      %set the name of y axis
nexttile([1 1]);          %EF occupy 2x2
plot(theta,EF);           %plot Element factor
xlim([-90 90]);           %set the range of x axis
xticks(-90:30:90);        %set the ticks of x axis(less, since EF is small)
title('Element factor');  %set the name of plot
xlabel('theta');          %set the name of x axis
ylabel('Intensity');      %set the name of y axis
nexttile([1 1]);          %AF occupy 2x2
plot(theta,AF);           %plot Array factor
xlim([-90 90]);           %set the range of x axis
xticks(-90:30:90);        %set the ticks of x axis(less, since AF is small)
title('Array factor');    %set the name of plot
xlabel('theta');          %set the name of x axis
ylabel('Intensity');      %set the name of y axis

[pks,loc,wid]=findpeaks(I,theta,"NPeaks",4,"SortStr","descend","WidthReference","halfheight");
                    %find NPeaks in the order of SortStr with FWHM
                    %You need to install 'signal processing toolbox'
Peak=[pks;loc;wid]; %sort data of findpeaks as one matrix



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%Periodic Optical Phased Array - Lumerical app (MODE)
%%%You must read these articles for MATLAB-Lumerical interoperability
%%%https://optics.ansys.com/hc/en-us/articles/360026142074
%%%https://optics.ansys.com/hc/en-us/articles/360034407914-appopen
%%%https://optics.ansys.com/hc/en-us/articles/360034928073-appevalscript
%%%Created at Nov.2022 by DO HYUNG KIM (Aaron), dohyung.aaron.kim@gmail.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% path(path,'C:\Program Files\Lumerical\v202\api\matlab');
%                            %set path to MATLAB api folder 
%                            %replace v202 with your Lumerical version
% h=appopen('mode','-hide'); %open a Lumerical app
% 
% appputvar(h,'N',N);                     %send number of channels (to MODE)
% appputvar(h,'samp',samp(2));            %send sample rate of farfield
% appputvar(h,'lambda',lambda);           %send wavelength
% appputvar(h,'lambda_span',lambda_span); %send wavelength range
% appputvar(h,'l',l);                     %send waveguide length
% appputvar(h,'b',b);                     %send waveguide width
% appputvar(h,'t',t);                     %send waveguide thickness
% appputvar(h,'n',n);                     %refractive index of farfield material(air)
% appputvar(h,'X',X);                     %x-start-positions(x min) of channels
% appputvar(h,'Y',Y);                     %send Applied phase difference
% appputvar(h,'Phase',Phase*360/(2*pi));  %send Applied phase of channels
% 
% appevalscript(h,'save("C:\Users\ski03\Downloads\ex"+num2str(N)+".lms");');
%                        %save script file to run later
%                        %replace ski03 with the user name of your computer
% text = fileread("C:\Users\ski03\Downloads\Periodic_OPA_MODEscript_for_MATLAB.txt");
%                        %read the Lumerical script file as string 
%                        %replace ski03 with the user name of your computer
% appevalscript(h,text); %send the script file (to MODE)
%                        %run MODE
%                        %the script contains these:
% %adding materials(Si3N4,SiO2), waveguides(Periodic), var fdtd, (no mesh),
% %, modes, monitor(power). running and getting farfield.
% %For more details, check the txt file.
% 
% theta_MODE = appgetvar(h,'thetalinear'); %get theta (from MODE)
% I_MODE = appgetvar(h,'far');             %get Intensity(MODE farfield)
% Peak_MODE = appgetvar(h,'Peaks');        %get intensity&angle of four peaks
% appclose(h);                             %close the opened Lumerical app
% 
% figure;                  %plot in different window
% plot(theta_MODE,I_MODE); %plot Intensity (MODE farfield)
% xlim([-90 90]);          %set the range of x axis
% xticks(-90:30:90);       %set the ticks of x axis
% title('MODE farfield');  %set the name of plot 
% xlabel('theta');         %set the name of x axis 
% ylabel('Intensity');     %set the name of y axis
