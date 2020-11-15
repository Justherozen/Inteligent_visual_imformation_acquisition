% Projector-Camera Stereo calibration parameters:

% Intrinsic parameters of camera:
fc_left = [ 1390.185615 1342.961913 ]; % Focal Length
cc_left = [ 319.049517 258.939992 ]; % Principal point
alpha_c_left = [ 0.000000 ]; % Skew
kc_left = [ 0.714732 -8.436035 0.013103 0.013827 0.000000 ]; % Distortion

% Intrinsic parameters of projector:
fc_right = [ 717.509864 689.482803 ]; % Focal Length
cc_right = [ 116.343537 52.639566 ]; % Principal point
alpha_c_right = [ 0.000000 ]; % Skew
kc_right = [ 0.426366 -12.540145 -0.036853 -0.006922 0.000000 ]; % Distortion

% Extrinsic parameters (position of projector wrt camera):
om = [ -0.246139 0.180124 0.002520 ]; % Rotation vector
T = [ -156.419513 -162.259277 109.642700 ]; % Translation vector
