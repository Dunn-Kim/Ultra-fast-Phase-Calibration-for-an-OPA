%% getSpot
% To get an intensity of a spot of interest, which is a beamforming point.
% The size and position fo the spot will influence the performance of beamforming.
% A reference image and the position of the spot should be given
% If the latter is not given, the beamforming point (max point) is used

function Spot = getSpot(Img, point)
    global OPA 
    beam = OPA.beam; % Type of beam (line=1, point=0)
    blur = OPA.blur; % Define how blurry the spot is
    Spot = 0;        % Intensity of the spot of interest
    pixs = 0;        % The number of pixels of the spot
    
    %%%SingleSpot%%%
    if isempty(blur) && nargin == 1        % if no blur around 'VAR.point'
        point = OPA.point;             % Accept 'VAR.point'
        Spot = Img(point(1),point(2)); % Spot Intensity = Intensity of 'VAR.point'
        return                         
    elseif isempty(blur) && nargin == 2    % if no blur around 'point'
        Spot = Img(point(1),point(2)); % Spot Intensity = Intensity of 'point'
        return                         
    
    %%%BlurredSpot%%%
    elseif ~isempty(blur)                 
        if nargin == 1                 % if blur around 'VAR.point'
            point = OPA.point;         % Aceept 'VAR.point'
        end
        row=point(1); col=point(2);    % Position of the point
        COL = col-blur:col+blur;       % column positions of the blurred area
        if beam == 0                        % if point beam
            ROW = @(k) row-(k-1):row+(k-1); % 
        elseif beam == 1 && blur(2) == 0    % if line beam and ccd camera
            ROW = @(k) row-15*k:row+15*k;   % extended along row
        elseif beam == 1 && blur(2) == 1    % if line beam and cmos camera
            ROW = @(k) row-3*k:row+3*k;     % extended along row  
        else
            
        end  
        for i=1:blur                   % Column x-blur(1) ~ x-1
            Spot = Spot + sum(Img(ROW(i),COL(i))); % Sum the intensity
            pixs = pixs + length(ROW(i));     % Count the Number of pixels
        end                                    
        for i=blur+1:size(COL)         % Column x ~ x+blur(1)
            j=size(COL)+1-i;                       % to make the spot symmetric
            Spot = Spot + sum(Img(ROW(j),COL(i))); % Sum the intensity
            pixs = pixs + length(ROW(j));          % Count the Number of pixels
        end                                    
        Spot = Spot / pixs;                        % Spot Intensity = Average intensity of the blurred area

    else                                     
        error("Err: getSpot Input error\n"); % Input error
    end                                      

end
