%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Exploiting Spatial Reuse in Wireless Networks through Decentralised MABs
% F. Wilhelmi, B. Bellalta, A. Jonsson, C. Cano, G. Neu, S. Barrachina
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Add paths to methods folders
addpath(genpath('Experiments/'));
addpath(genpath('Power Management Methods/'));
addpath(genpath('Throughput Calculation Methods/'));
addpath(genpath('Network Generation Methods/'));
addpath(genpath('Reinforcement Learning Methods/'));
addpath(genpath('Reinforcement Learning Methods/Action Selection Methods/'));
addpath(genpath('Auxiliary Methods/'));

nWlans = 4;                         % Number of WLANs in the map
totalIterations = 10000;            % Maximum convergence time (one period implies the participation of all WLANs)
minimumIterationToConsider = 5000;  % Iteration from which to consider the obtained results
roundsPerIteration = 1;             % Maximum number of learning iterations done by each WLAN inside a general iteration
totalRepetitions = 100;             % Number of TOTAL repetitions to take the average

plotResults = 0;                    % To plot or not the results at the end of the simulation

NOISE_DBM = -100;                   % Floor NOISE_DBM in dBm

nChannels = 2;                      % Number of available channels (from 1 to n_channels)
channelActions = 1 : nChannels;     % Possible channels
ccaActions = [-82];                 % CCA levels (dBm)
txPowerActions = [5 10 15 20];      % Transmit power levels (dBm)

% Each state represents an [i,j,k] combination for indexes on "channels", "cca" and "tx_power"
possibleActions = 1:(size(channelActions, 2) * ...
    size(ccaActions, 2) * size(txPowerActions, 2));
K = size(possibleActions,2);   % Total number of actions
allCombs = allcomb(1:K, 1:K);

% Structured array with all the combinations (for computing the optimal)
possibleComb = allcomb(possibleActions,...
    possibleActions,possibleActions,possibleActions);

printInfo = 1;                      % To print info after Bandits implementation (1) or not (0)
drawMap = true;                     % Variable for drawing the map
randomInitialConfiguration = true;  % Variable for assigning random channel/tx_power/cca at the beginning

% Dimensions of the 3D map
MaxX=10;
MaxY=5; 
MaxZ=10;
% Maximum range for a STA
MaxRangeX = 1;
MaxRangeY = 1;
MaxRangeZ = 1;
MaxRange = sqrt(3);

% Policies indexes
RANDOM_POLICY = 0;
EG_POLICY = 1;
EXP3_POLICY = 2;
UCB_POLICY = 3;
TS_POLICY = 4;

% Update modes of the exploration coefficient (epsilon-greedy)
UPDATE_MODE_FAST = 0;   % epsilon = initial_epsilon / t 
UPDATE_MODE_SLOW = 1;   % epsilon = epsilon / sqrt(t)

% Selected update mode (epsilon-greedy)
updateModeEpsilon = UPDATE_MODE_SLOW;

% Save constants into current folder
save('constants.mat');  
