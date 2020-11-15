% Projector-Camera Stereo calibration parameters:

% Intrinsic parameters of camera:
fc_left = [ 962.258411 879.437298 ]; % Focal Length
cc_left = [ 315.690483 195.567866 ]; % Principal point
alpha_c_left = [ 0.000000 ]; % Skew
kc_left = [ -0.147386 12.525282 -0.012123 -0.004050 0.000000 ]; % Distortion

% Intrinsic parameters of projector:
fc_right = [ 655.180570 631.692040 ]; % Focal Length
cc_right = [ 146.715921 65.996790 ]; % Principal point
alpha_c_right = [ 0.000000 ]; % Skew
kc_right = [ -0.235215 17.234434 -0.024240 0.010232 0.000000 ]; % Distortion

% Extrinsic parameters (position of projector wrt camera):
om = [ 0.045840 -0.624258 0.046310 ]; % Rotation vector
T = [ 333.954198 39.314129 511.811969 ]; % Translation vector
