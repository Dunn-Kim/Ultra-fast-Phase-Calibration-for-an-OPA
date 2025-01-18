%% TempSet
% To set the temperature of the phase shifters
% There is a TEC on the board

%%%Protocol%%%
% Tx Protocol: Head(0xF1) | cmd(0xA1~6)   | Data0(Ch1~128 or 00) | Data1 | Data2 | Checksum | Tail(0xE1)
% Rx Protocol: Head(0xF2) | cmd(0xB1,2,4) | Data0(Receive cmd) or Data1&Data2(Current Temp) | Checksum | Tail(0xE2)
% A1=Ch_Control, A2=TEC_Temp_Set, A3=TEC_On/off, A4=Current_Temp, A5=Debug_TEC
% B1=Ack(acknowledgement), B2=Nak, B4=Current_Temp
% Checksum=sum(entire_Command); Checksum=Checksum(end-1:end);


function temp=TempSet
    global TEMP % Temperature- setting value

    %%%SetTemperature%%%
    data = dec2hex(TEMP*100,4);                      
    data = [string(data(1:2)),string(data(3:4))];    % Encode TEMP
    cs = hex2dec(["F1","A3","00",data(1),data(2)]);  
    cs = dec2hex(sum(cs)); cs=cs(end-1:end);         % Checksum
    Send("F1 A3 00 " + data(1)+" " + data(2)+" " + cs+" " + "E1");

    %%%ReadTemperature%%%
    temp = Send("F1 A5 00 00 01 97 E1");             % Request the current temp
    temp1 = dec2hex(temp(3:4));                      % the returned temp is converted to char array 
    temp = "";                                       % empty temp for summation below
    for i=[1,3,2,4]                                  % minding the index of char array
        temp = temp+convertCharsToStrings(temp1(i)); % align and merge
    end
    temp = hex2dec(temp)/100;                        % Decode
        
end
