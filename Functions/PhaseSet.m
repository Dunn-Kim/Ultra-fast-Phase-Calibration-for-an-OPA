%% PMset
% To set the voltage input for phase shifters
% If ch and deg are not given, apply 'OPA' on every channel
% If both are given, apply 'ch' on a sigle channel


function out = PhaseSet(ch, deg)
    global OPA DAC    

    if nargin == 0                    % Apply for every channel
        Rem = mod(OPA.phase/360,1);   % Fractional part of OPA/360
        PM = round(sqrt(Rem)*DAC);    % DAC value for Phase(deg) modulation
        for ch=1:OPA.ch                                                  % every channel
            data = dec2hex(PM(ch),4);                                    
            data = [string(data(1:2)),string(data(3:4))];                % Encode
            cs = hex2dec(["F1","A1",dec2hex(ch),data(1),data(2)]);        
            cs = dec2hex(sum(cs)); cs=hex2dec(cs(end-1:end));            % Checksum
            Send([241,161,ch,hex2dec(data(1)),hex2dec(data(2)),cs,225]); % Ch_Control
        end                                                              

    elseif nargin == 2                % Apply for a single channel
        Rem = mod(deg/360,1);         % Fractional part of deg/360
        PM = round(sqrt(Rem)*DAC);    % DAC value for Phase(deg) modulation
            data = dec2hex(PM,4);                                        
            data = [string(data(1:2)),string(data(3:4))];                % Encode
            cs = hex2dec(["F1","A1",dec2hex(ch),data(1),data(2)]);       
            cs = dec2hex(sum(cs)); cs=hex2dec(cs(end-1:end));            % Checksum
            Send([241,161,ch,hex2dec(data(1)),hex2dec(data(2)),cs,225]); % Ch_Control
    
    else                                   % Input error
        error("Err: PMset.m Input error\n"); 
    end                                    

    out = nan;    % Return nothing
end

