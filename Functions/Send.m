%% Send
% To communicate with the board (serial communication)

%%%Protocol%%%
% Tx Protocol: Head(0xF1) | cmd(0xA1~6)   | Data0(Ch1~128 or 00) | Data1 | Data2 | Checksum | Tail(0xE1)
% Rx Protocol: Head(0xF2) | cmd(0xB1,2,4) | Data0(Receive cmd) or Data1&Data2(Current Temp) | Checksum | Tail(0xE2)
% A1=Ch_Control, A2=TEC_Temp_Set, A3=TEC_On/off, A4=Current_Temp, A5=Debug_TEC
% B1=Ack(acknowledgement), B2=Nak, B4=Current_Temp
% Checksum=sum(entire_Command); Checksum=Checksum(end-1:end);


function out = Send(in)
    global SP

    %%%Formatting_in%%%
    if isstring(in)                      % if 'in' is string
        in_hex = in;                     % in_hex is for fprintf
        in_dec = hex2dec(split(in_hex)); % in_dec is for write
    elseif isnumeric(in)                 % if 'in' is dec
        in_dec = in;                      
        in_tem = dec2hex(in_dec);        % convert to hex(char)
        in_hex = "";                     % empty for summation below
        for i=1:length(in_tem)             
            in_hex = in_hex + convertCharsToStrings(in_tem(i,1:2)); % align & merge
        end                         
    else
        error("Err: Send.m input error. Input should be string or numeric\n")
    end                             

    %%%Write&Read%%%                       
    iter = 0;                              
    code = 0;                             
    while code ~= 1                             % While loop for write & read (if code=1, ends)
        write(SP,in_dec,'uint8');               % Send Tx(Tx_cmd) to board
        if in_dec(2) == 163 || in_dec(2) == 164 % if Tx: A3 or A4 (Temp set, TEC ON)
            pause(0.01)%Don't delete%
            out = nan;
            break                               % Nothing to return

        elseif in_dec(2) == 161                 % if Tx: A1 (ch_control)
            while SP.NumBytesAvailable == 0     % Size of Rx(Rx_cmd) -> 5
                pause(0.01)%Don't delete%
            end
            out = read(SP,SP.NumBytesAvailable,'uint8'); % Receive Rx
            if find(out==177)                   % if Rx is proper (B1(ACK))
                break
            else
                code=-1;
            end
            
        elseif in_dec(2) == 165                 % if Tx: A5 (Temp read)
            while SP.NumBytesAvailable == 0     % Size of Rx -> 6
                pause(0.01)%Don't delete%
            end
            out = read(SP,SP.NumBytesAvailable,'uint8'); % Receive Rx
            if find(out==180)                   % if Rx is proper (B4(Temp read))
                break
            else
                code=-1;
            end
            
        elseif in_dec(2) == 166                 % if Tx: A6(TEC_Debug)
            fprintf("Warning: I don't know about this Tx_cmd, sorry");
            fprintf("         If you wanna use it, plz disable an elseif statement");
            error("         (in_dec(2) == 166) in while loop in Send.m -- DH")
        
        else
            out_tem = dec2hex(out);             % convert to hex to display(fprintf)
            out_hex = "";                       % empty for summation below
            for i=1:length(out_tem)             % each hexcode to string
                out_hex = out_hex + convertCharsToStrings(out_tem(i,1:2)); % align & merge
            end    
            fprintf("  Err: Send.m input error -- Invalid cmd");
            fprintf("  Tx: ""%s"" \n", in_hex);               
            error("");
        end
 
        %%%Check_Rx_cmd%%%
        head = find(out==242);                  % Find 242(F2)
        tail = find(out==226);                  % Find 226(E2)
        if isempty(head) || isempty(tail)       % if Rx is empty
            code = -1;                          
        elseif head ~= 1                        % if Rx is not in order
            out  = [out(head:end),out(1:tail)]; % Reorder Rx to be [242,,,226]
            code = 1;
        end              
        %%%ReadError%%%
        iter = iter + 1;                           % Iteration count
        if code ~= 1 && iter == 10                 % if read fails 10 times
            out_tem = dec2hex(out);                % convert to hex to display(fprintf)
            out_hex = "";                          % empty for summation below
            for i=1:length(out_tem)                % each hexcode to string
                out_hex = out_hex + convertCharsToStrings(out_tem(i,1:2)); % align & merge
            end    
            fprintf("  Connection is corrupted."); % Return the communication status(NACK)
            fprintf("  Tx: ""%s"" \n", in_hex);
            fprintf("  Rx: ""%s"" \n", out_hex);                
            error("Err: Send.m write&read error. (It might be a timing error, add pause\n")
        end                               
    end                                  

end
