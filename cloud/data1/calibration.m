% Projector-Camera Stereo calibration parameters:

% Intrinsic parameters of camera:
fc_left = [ 767.181624 762.367673 ]; % Focal Length
cc_left = [ 275.063817 266.980733 ]; % Principal point
alpha_c_left = [ 0.000000 ]; % Skew
kc_left = [ 0.305582 -1.252862 0.008716 -0.018121 0.000000 ]; % Distortion

% Intrinsic parameters of projector:
fc_right = [ 316.177660 306.492070 ]; % Focal Length
cc_right = [ 59.519859 33.026989 ]; % Principal point
alpha_c_right = [ 0.000000 ]; % Skew
kc_right = [ 0.342975 9.668901 -0.037361 -0.009896 0.000000 ]; % Distortion

% Extrinsic parameters (position of projector wrt camera):
om = [ -0.065112 -0.417440 0.096391 ]; % Rotation vector
T = [ 234.377511 59.037088 542.522856 ]; % Translation vector
