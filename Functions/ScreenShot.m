%% Screenshot
% To capture the area of interest on the screen
% cam = [x y width height]
% Screencapture.m is downloaded from the community below
% https://www.mathworks.com/matlabcentral/fileexchange/24323-screencapture-screenshot-of-component-figure-or-screen

function img = ScreenShot(pos)
    global CAM

    if nargin == 0 % if no capture region is specified
        pos = CAM;  % Apply the default(cam)
    end           

    img = ScreenCapture(0,'Position',pos); % Capture a small desktop region(pos)
    img = im2gray(img);                   % Convert it into grayscale

end
