%% Editor Log
%%%Beamforming_MATLAB_code%%%
% created by Do Hyung Kim(Aaron) in Photonics Research Lab, Kwangwoon University
% If you have questions, mail me anytime [kw.aaron.kim@gmail.com]
% I also prefer Linkedin 'https://www.linkedin.com/in/do-hyung-kim-aaron/' 
%
% The code is originated from Photonics-IC Laboraory, Pusan University
% created       : Jan.18.2023  Rewrote the original REV_Beamforming_code
%                              Organized the entire code and reduced functions
%                              (Main, getSpot, PMset, REV_method, Screenshot, Send)
%                              
% 1st update    : Feb.06.2023  Figured out how to use screencapture.m
%                              Solved 'Unable to connect to the serialport' issue
%                              Solved 'Unable to read all the requested data in timeout' issue
%                              Added TempSet.m and changed the names of funcs & variables
%                              (Main, getSpot, PhaseSet, REV_method, ScreenShot, Send, TempSet)
%                              Added a subsection: Plot before and after beamforming
%                              Added many comments(printf, error) so that the operator follows the flow
%                              Organized the entire code
%
% 2nd update    : Feb.10.2023  Solved 'Send.m second input argument' issue
%                              Successfully did the beamforming experiment
%
% 3rd update    : Feb.16.2023  Solved a number of unexpected issues with Send.m by adding pause() between wr&wr wr&re
%                              Specified CAM for two different camera software
%                              Added a subsection: Select_BeamformingPoint (mouse point, center line etc)
%                              Added a subsection: RecoverInitialState (reset volatege output) 
%                              Added a section: Save Results
%                              Changed the name of Main.m --> REVBeamforming.m
%                              Added another Main.m --> ReadLUT_BeamSteering.m
%                              




%% Initialization
fprintf("\n%s\n",datetime('now','Format','MMM.dd.yyyy  HH:mm:ss'));
fprintf("REV Beamforming MATLAB\n");
fprintf("Initialization..\n");

%%%Path%%%
% Screencapture is downloaded from the community below
% https://www.mathworks.com/matlabcentral/fileexchange/24323-screencapture-screenshot-of-component-figure-or-screen
currentpath = "C:\Users\PRL\Desktop\YJ\REV Beamforming MATLAB code by DH"; % Main.m path
addpath(currentpath + "\Functions")
addpath(currentpath + "\ScreenCapture")
addpath(currentpath + "\Results")

%%%GlobalVariables%%%
global OPA DAC CAM TEMP SP
OPA.ch     = 64;               % Number of channel
OPA.beam   = 1;                % Beam shape(0=point beam, 1=line beam)
OPA.point  = zeros(1,2);       % Position of beamforming point(row,col)
OPA.blur   = zeros(1,2);       % Measurement area around beamforming point
OPA.phase  = zeros(1,OPA.ch);  % Phase value of each channel
OPA.sample = 5;                % Sample points for sine fitting(should be an odd number)
DAC        = 11775;            % DAC value for 2pi modulation
CAM        = zeros(1,4);       % Region of interest for ScreenCapture(farfield)
TEMP       = 50;               % Default TEC value(Celsius)
SP          = serialport("COM5",115200); % Phase Modulation Board (RS-232)
SP.BaudRate = 115200;          % Board settings for serial communicaiton
SP.DataBits = 8;               
SP.StopBits = 1;               
SP.Parity   = 'none';          
SP.Timeout  = 10;       

%%%Protocol%%%
% Tx Protocol: Head(0xF1) | cmd(0xA1~6)   | Data0(Ch1~128 or 00) | Data1 | Data2 | Checksum | Tail(0xE1)
% Rx Protocol: Head(0xF2) | cmd(0xB1,2,4) | Data0(Receive cmd) or Data1&Data2(Current Temp) | Checksum | Tail(0xE2)
% A1=Ch_Control, A2=TEC_Temp_Set, A3=TEC_On/off, A4=Current_Temp, A5=Debug_TEC
% B1=Ack(acknowledgement), B2=Nak, B4=Current_Temp
% Checksum=sum(entire_Command); Checksum=Checksum(end-1:end);

%%%BootBoard%%%
fprintf("\nBoot Board\n");
%%%TemperatureSetting
fprintf("Temperature Setting ,%d C\n", TEMP);
temp = 0;                % The current temperature
iter = 0;                % Iteration count
while abs(temp-TEMP) > 1 % Wait until the current temperature reaches TEMP
    iter = iter+1;
    temp = TempSet;
    if isnan(temp) || iter == 500
        error("Err: TempSet.m error\n") % print Temp read error
    end
end
if isnan(temp)
    error("Err: Temp read error\n") % print Temp read error
else 
    fprintf("..Done  (Current: %.2f C)\n", temp);
end



%% Beamforming
fprintf("\nPrepare REV Beamforming\n");

%%%Set_CaptureRegion_OnScreen%%%
% Open the camera acquisition software 
% Press [win + â†?] so that it is placed at the first half of the screen
% If AVAL DATA Tsight(CCD camera), close every toolmenu and remove x,y lines on the image
% If RayCi64 Standard(CMOS camera), maximize the inner window and change AOI to rectangular
Camera.prompt = {'Select which camera acquisition software is used.'};
Camera.list = {'AVAL DATA Tsight(CCD Camera)','RayCi64 Standard(CMOS Camera)'};
[indx1,tf1] = listdlg('PromptString',Camera.prompt,'SelectionMode','single',...
    'ListString',Camera.list,'ListSize',[300,100]);
SIZ = size(ScreenShot([]));   % size of the entire screen
if indx1 == 1 && tf1 == 1     % if CCD camera
    CAM = [SIZ(2)*1/40,SIZ(1)*17/128,SIZ(2)*5/11,SIZ(1)*14/23];
    OPA.blur = [10,indx1];  
elseif indx1 == 2 && tf1 == 1 %if CMOS camera
    CAM = [105,260,412,413];
    OPA.blur = [2,indx1];
else
    error("Err: Camera acqusition software has not been selected");
end

%%%RecoverInitialState%%%
Recover.prompt = {'Recover Initial State?'};
Recover.list = {'Yes, reset voltage outputs','No, don''t reset'};
[indx2,tf2] = listdlg('PromptString',Recover.prompt,'SelectionMode','single',...
    'ListString',Recover.list,'ListSize',[300,100]);
if indx2 == 1 && tf2 == 1
    fprintf("RecoverInitialState: Start\n");
    PhaseSet;       % Reset voltage outputs of the board as zero
    fprintf("..Done\n");
end

%%%Select_BeamformingPoint%%%
fprintf("\nSelect the beamforming point\n");
Point.prompt = {'Select how to set the beamforming point.',...
    'Center line is a horizontal line in the middle',...
    'Mouse point and Max point might influence the performance'};
Point.list = {'1/6 of horizontal half line','2/6 of horizontal half line',...
    '3/6 of horizontal half line = Center', '4/6 of horizontal half line',...
    '5/6 of horizontal half line','Manually mark using mouse', 'Max point (random)'};
[indx3,tf3] = listdlg('PromptString',Point.prompt,'SelectionMode','single',...
    'ListString',Point.list,'ListSize',[300,300]);
Img = ScreenShot;             % Capture the CAM region on the screen
Img_size = size(Img);         % size of the CAM region
SIZ = size(ScreenShot([]));   % size of the entire screen
%%%CenterPoint%%%
if indx3 == 1 && tf3 == 1
    OPA.point = [round(Img_size(1)/2),round(Img_size(2)*1/6)];
elseif indx3 == 2 && tf3 == 1
    OPA.point = [round(Img_size(1)/2),round(Img_size(2)*2/6)];
elseif indx3 == 3 && tf3 == 1
    OPA.point = [round(Img_size(1)/2),round(Img_size(2)*3/6)];
elseif indx3 == 4 && tf3 == 1
    OPA.point = [round(Img_size(1)/2),round(Img_size(2)*4/6)];
elseif indx3 == 5 && tf3 == 1
    OPA.point = [round(Img_size(1)/2),round(Img_size(2)*5/6)];
%%%MousePoint(Manual_Input)%%%
elseif indx3 == 6 && tf3 == 1
    f4=figure(4); f4.Position=[SIZ(2)*1/4,SIZ(1)*1/5,SIZ(2)/2,SIZ(1)*3/5];
    image(Img); colorbar; colormap jet
    yti=get(gca,'YTick'); set(gca,'YTick',yti)
    yticklabels(flip(yti))
    title(sprintf("USE your mouse"));     
    [x,y] = ginput(1);
    OPA.point = [round(Img_size(1)-y),round(x)];   
    close(f4)
%%%MaxPoint%%%
elseif indx3 == 7 && tf3 == 1
    [mx, idx] = max(Img(:));             % Max point(mx:intensity, idx:linear index)
    [row, col] = ind2sub(Img_size, idx); % row & col of max point
    OPA.point = [row,col];               % position (row,col) of beamforming point
else
    error("Err: Beamforming point has not been selected");
end
fprintf("..Done\n");

%%%REV_Method%%%
OPAd = OPA.phase;           % Store default values of OPA
fprintf("\nInitiate REV Beamforming");
pause(1);
fprintf(": Started \n");
tic                         % Start timer(record current time)
result = REV_method;        % REV_method + calibration
elap = toc;                 % End timer(calculate Elapse time)
fprintf("..Done  (Elapse time: %.0f sec)\n", elap);

%%%Figure%%
fprintf("\nPlot Farfield before & after..\n");
%%%Farfield-BeforeREV%%%
SIZ = size(ScreenShot([]));
f1=figure(1); f1.Position=[SIZ(2)*2/3,40,SIZ(2)/3,SIZ(1)*2/5];
image(Img); colorbar; colormap("gray")
yti=get(gca,'YTick'); set(gca,'YTick',yti)
yticklabels(flip(yti))
title(sprintf("Farfield before REV"));     
xlabel('X, num of pixs'); ylabel('Y, num of pixs');         
%%%Farfield-AfterREV%%%
Img2=ScreenShot;
f2=figure(2); f2.Position=[SIZ(2)*2/3,SIZ(1)/2+20,SIZ(2)/3,SIZ(1)*2/5];
image(Img2); colorbar; colormap("gray")
yti2=get(gca,'YTick'); set(gca,'YTick',yti2)
yticklabels(flip(yti2))
title(sprintf("Farfield After REV"));     
xlabel('X, num of pixs'); ylabel('Y, num of pixs'); 



%% Save Results
fprintf("\nSave Results..\n");

%%%PlotSineCurveFitting%%%
fprintf("Save the plot of cosine fitting\n");
scan   = zeros(1,OPA.sample);       % samples
Int    = zeros(1,length(scan));     % intensity of samples
th     = -pi:pi/1000:pi;            % x-axis for plotting sine fit
f3     = figure(3);                 % Thirth figure for plotting sine fit      
f3.Position=[SIZ(2)/2,SIZ(1)*2/5,SIZ(2)/3,SIZ(1)*2/5]; % Position of the figure
for i=2:OPA.ch
    scan = result{3,i};        % samples
    Int  = result{4,i};        % intensity of samples
    Fit  = result{5,i};        % fitting func, f(x)=y0+y1*cos(x-x0)
    y0   = result{6,i};        
    y1   = result{7,i};        
    Pha  = result{8,i};        % Phase Calibration Value
    p1=plot(scan*pi/180, Int); p1.Color=[0 0 0]; p1.Marker='o'; hold on % plot samples
    p2=plot(th, Fit(th)); p2.Color=[0 0 0]; p2.LineStyle='-';           % plot fitting function
    p3=plot(Pha, Fit(Pha)); p3.Color=[1 0 0]; p3.Marker='*';            % plot Phase Calibration Value
    p4=bar(Pha, Fit(Pha)); p4.EdgeColor=[1 0 0]; p4.LineStyle=':'; 
    p4.LineWidth=1; p4.BarWidth=0.01; hold off                          % plot Phase Calibration Value
    title(sprintf("Channel %d",i));
    xlabel('phase (deg)'); ylabel('Intensity (a.u.)');         
    xlim([-2*pi,2*pi]); ylim([y0-2*y1,y0+2*y1]);  
    %%%SavePlot_image%%%
    saveas(f3,currentpath+"\Results\Ch"+num2str(i),'png'); % Save plot as .png
end
close(f3)

%%%SaveResultTable%%%
fprintf("Save the result table");
for i = 2:OPA.ch
    result{2,i} = "Ch"+num2str(i);
end
result{1,1} = string(datetime('now','Format','MMM.dd HH:mm'));
result{1,2} = Camera.list(indx1);
result{1,3} = Recover.list(indx2);
result{1,4} = Point.list(indx3);
result{3,1} = "5 samples";
result{4,1} = "Intensity";
result{5,1} = "Cosine Fitting";
result{6,1} = "y bias";
result{7,1} = "amplitude";
result{8,1} = "x bias (Phase Calibration value)";
save(currentpath+"\Results\result.mat",'result');

%%%SaveCurrentPhase%%%
OPAd = OPA.phase;
save(currentpath+"\Results\OPA.phase.mat",'OPAd');



%% Close Serial Communication
%%%DisconnectfromPC%%%
delete(SP)
%%%RemovefromMATLAB_Workspace%%%
clear SP
fprintf("\nBeamforming Process Terminated..\n");