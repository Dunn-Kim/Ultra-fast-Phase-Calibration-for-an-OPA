%% REV_method
% To calibrate initial phases using REV method
% 

function result = REV_method
    global OPA 
    ch     = OPA.ch;             % num of channels
    sample   = OPA.sample;       % num of samples
    scan   = zeros(1,sample);     % samples
    for i=1:sample
    scan(i) = round(3600*i/(sample+1))/10 - 180;
    end
    Int    = zeros(sample,ch-1); % intensity table
    OPAd   = OPA.phase;          % Default values of OPA
    result = cell(8,ch);         % Result table
    Pha    = [];                 % Phase Calibration value
   
    %%%PhaseErrorCalculation%%%
    for i = 2:ch                            % one channel by one (2~ch)
        %%%Apply_5_Samples%%%
        for j=1:sample                        % one sample by one
            PhaseSet(i,scan(j));            % apply a sample
            pause(0.06);                    % DAC: 0.003 s, Camera: 0.03 s(30 Hz)
            Int(j,i-1) = getSpot(ScreenShot); % intensity of the sample
        end                                 
        PhaseSet(i,OPAd(i));                % Recover the default state
                
        %%%SineCurveFitting%%%
        A = [ones(sample,1), cosd(scan)', sind(scan)']\Int(:,i-1); % A*X=B, X=A\B
        y0 = A(1);                          % y-intercept
        y1 = sqrt(A(2)^2 + A(3)^2);         % coefficient
        x0 = atan2(A(3),A(2));              % x-intercept, atan2(y,x)=arctan(y/x)
        fit = @(x) y0+y1*cos(x-x0);         % fitting function
        
        %%%StoreParameters%%%
        result{3,i} = scan;       % store 5 samples
        result{4,i} = Int(:,i-1); % store intensity of 5 samples
        result{5,i} = fit;        % store fitting func
        result{6,i} = y0;         % store coef, interc
        result{7,i} = y1;         % store coef, interc
        result{8,i} = x0;         % store coef, interc
        Pha = [Pha,x0];           % store phase calibration value
    end                                     

    %%%Calibration%%%
    fprintf("  Calibration starts \n");
    Pha = [0,Pha];
    OPA.phase = round(OPAd + Pha*180/pi); PhaseSet; % Calibrate
    
end
