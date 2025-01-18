%% Initialization
fprintf("\n%s\n",datetime('now','Format','MMM.dd.yyyy  HH:mm:ss'));
fprintf("BeamSteering\n");
fprintf("Initialization..\n");

%%%Path%%%
currentpath = "C:\Users\PRL_LABVIEW\Downloads\DH(Aaron)\Matlab"; % Main.m path
addpath(currentpath + "\Functions")
addpath(currentpath + "\Results")

%%%GlobalVariables%%%
global OPA DAC TEMP SP
OPA.ch     = 64;               % Number of channel
OPA.phase  = zeros(1,OPA.ch);  % Phase value of each channel
DAC        = 11775;            % DAC value for 2pi modulation
TEMP       = 50;               % Default TEC value(Celsius)
SP          = serialport("COM1",115200); % Phase Modulation Board (RS-232)
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
    if isnan(temp) || iter == 10
        error("Err: TempSet.m error\n") % print Temp read error
    end
end
if isnan(temp)
    error("Err: Temp read error\n") % print Temp read error
else 
    fprintf("..Done  (Current: %.2f C)\n", temp);
end



%% BeamSteering

%%%ReadLUT%%%
fprintf("\nRead LUT\n");
LUT = readtable(currentpath+"\Results\Result Table.xlsx");
if isempty(LUT)
    errror("Err: read LUT error")
end

%%%ManualInput
fprintf("Enter the steering angle(deg) ..\n");
deg = inputdlg("Enter the steering angle(deg)");
fprintf("%d (deg)\n", deg);
OPAt = OPA.phase;
if deg >= 0
    for i=2:64
        OPAt(i) = OPAt(i) + deg*i;
    end
else
    for i=2:64
        OPAt(i) = OPA(i) - deg*i;
    end
end

%% Close Serial Communication
%%%DisconnectfromPC%%%
delete(SP)
%%%RemovefromMATLAB_Workspace%%%
clear SP
fprintf("\nBeamSteering Process Terminated..\n");