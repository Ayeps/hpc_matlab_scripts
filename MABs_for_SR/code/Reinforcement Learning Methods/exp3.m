%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Exploiting Spatial Reuse in Wireless Networks through Decentralised MABs
% F. Wilhelmi, B. Bellalta, A. Jonsson, C. Cano, G. Neu, S. Barrachina
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [tptExperiencedByWlan, timesArmHasBeenPlayed] = exp3(wlans, gamma, eta, varargin)

% EXP3MethodBasic Given an OBSS, applies EXP3 (basic formulation) to maximize the experienced throughput
%
%   OUTPUT: 
%       * tptExperiencedByWlan - throughput experienced by each WLAN
%         for each of the iterations done
%       * timesArmHasBeenPlayed - times each action has been played
%   INPUT: 
%       * wlan - wlan object containing information about all the WLANs
%       * gamma - weigths regulator EXP3
%       * eta - learning rate EXP3

    constants
    
    try
        % Update possible actions
        nChannels = varargin{1};
        channelActions = 1 : nChannels;
        ccaActions = varargin{2};
        txPowerActions = varargin{3};
        % Each state represents an [i,j,k] combination for indexes on "channels", "cca" and "tx_power"
        possibleActions = 1:(size(channelActions, 2) * ...
            size(ccaActions, 2) * size(txPowerActions, 2));
        K = size(possibleActions,2);   % Total number of actions
        allCombs = allcomb(1:K, 1:K);    
    catch
        disp('Wrong number of input arguments')
    end
    
    %% INITIALIZE ALGORITHM
    % Use a copy of wlan to make operations
    wlansAux = wlans;
    nWlans = size(wlansAux, 2);

    % Find the index of the initial action taken by each WLAN
    initialActionIxPerWlan = zeros(1, nWlans);
    for i=1:nWlans
        [~,indexCca] = find(ccaActions==wlansAux(i).CCA);
        [~,indexTpc] = find(txPowerActions==wlansAux(i).TxPower);
        initialActionIxPerWlan(i) = indexes2val(wlansAux(i).Channel, ...
            indexCca, indexTpc, size(channelActions,2), size(ccaActions,2));
    end
    % Initialize the indexes of the taken action
    actionIndexPerWlan = initialActionIxPerWlan;                           
    
    % Compute the maximum achievable throughput per WLAN
    powerMatrix = PowerMatrix(wlansAux);     
    upperBoundRewardPerWlan = compute_max_bound_throughput(wlansAux, powerMatrix, NOISE_DBM, max(txPowerActions));
    
    selectedArm = actionIndexPerWlan;              % Initialize arm selection for each WLAN by using the initial action
    weightsPerArm = ones(nWlans, K);             % Initialize weight to 1 for each action
    currentAction = zeros(1, nWlans);
    previousAction = selectedArm;
    timesArmHasBeenPlayed = zeros(nWlans, K);     
    transitionsCounter = zeros(nWlans, K^2);
    % Initialize arms probabilities
    initialProbabilities = (1 - gamma) / K + gamma / K;
    armsProbabilities = initialProbabilities * ones(nWlans, K);
       
    %% ITERATE UNTIL CONVERGENCE OR MAXIMUM CONVERGENCE TIME           
  
    iteration = 1;
    
%     rw = zeros(nWlans, totalIterations + 1);
%     estimated_reward = zeros(nWlans, totalIterations + 1);

    while(iteration < totalIterations + 1) 
        % Assign turns to WLANs randomly 
        order = randperm(nWlans);  
        for i = 1 : nWlans % Iterate sequentially for each agent in the random order   
            learning_iteration = 1;
            while(learning_iteration <= roundsPerIteration)
                if iteration > 1 
                    armsProbabilities(order(i), :) = (1 - gamma) * (weightsPerArm(order(i), :) ./ ...
                        sum(weightsPerArm(order(i), :))) + gamma / K;
                end
                % Draw an action according to probabilites distribution
                selectedArm(order(i)) = randsample(1:K, 1, true, armsProbabilities(order(i),:));      
                currentAction(order(i)) = selectedArm(order(i));
                ix = find(allCombs(:,1) == previousAction(order(i)) & allCombs(:,2) == currentAction(order(i)));
                previousAction(order(i)) = currentAction(order(i));                
                transitionsCounter(order(i), ix) = transitionsCounter(order(i), ix) + 1;                
                timesArmHasBeenPlayed(order(i), selectedArm(order(i))) = timesArmHasBeenPlayed(order(i), selectedArm(order(i))) + 1;           
                [a, ~, c] = val2indexes(selectedArm(order(i)), size(channelActions,2), size(ccaActions,2), size(txPowerActions,2));
                wlansAux(order(i)).Channel = a;   
                %wlan_aux(order(i)).CCA = ccaActions(b);
                wlansAux(order(i)).TxPower = txPowerActions(c);  
                     
                % Compute the reward with the throughput obtained in the round after applying the action
                powerMatrix = PowerMatrix(wlansAux);
                tpt_after_action = compute_throughput_from_sinr(wlansAux, powerMatrix, NOISE_DBM);  % bps               
                for wlan_i = 1:nWlans                    
                    rw = (tpt_after_action(wlan_i) / upperBoundRewardPerWlan(wlan_i));
                    % Compute the estimated reward                    
                    if rw == 0
                        estimated_reward = 1e-6;
                    else
                        estimated_reward = (rw / armsProbabilities(wlan_i, selectedArm(wlan_i)));
                    end
                    % Update the weight for the selected action according to the estimated reward
                    weightsPerArm(wlan_i, selectedArm(wlan_i)) = ...
                        weightsPerArm(wlan_i, selectedArm(wlan_i)) * ...
                        exp((eta * estimated_reward / K));                    
                    if weightsPerArm(wlan_i, selectedArm(wlan_i)) > 1e20, weightsPerArm(wlan_i, selectedArm(wlan_i)) = 1e20; end                    
                end                                              
                learning_iteration = learning_iteration + 1;
            end
        end
        powerMatrix = PowerMatrix(wlansAux);    
        tptExperiencedByWlan(iteration,:) = compute_throughput_from_sinr(wlansAux, powerMatrix, NOISE_DBM);  % bps    
        % Increase the number of 'learning iterations' of a WLAN
        iteration = iteration + 1;     
    end
    
    %% PRINT INFORMATION REGARDING ACTION SELECTION
    if printInfo
        % Print the preferred action per wlan
        for i=1:nWlans      
            timesArmHasBeenPlayed(i, :)/totalIterations
            a = transitionsCounter(i,:);
            % Max value
            [val1, ix1] = max(a);
            [ch1_1, ~, x] = val2indexes(possibleActions(allCombs(ix1,1)), size(channelActions,2), size(ccaActions,2), size(txPowerActions,2)); 
            tpc1_1 = txPowerActions(x);
            [ch1_2, ~, x] = val2indexes(possibleActions(allCombs(ix1,2)), size(channelActions,2), size(ccaActions,2), size(txPowerActions,2)); 
            tpc1_2 = txPowerActions(x);
            % Second max value
            [val2, ix2] = max(a(a<max(a)));
            [ch2_1, ~, x] = val2indexes(possibleActions(allCombs(ix2,1)), size(channelActions,2), size(ccaActions,2), size(txPowerActions,2)); 
            tpc2_1 = txPowerActions(x);
            [ch2_2, ~, x] = val2indexes(possibleActions(allCombs(ix2,2)), size(channelActions,2), size(ccaActions,2), size(txPowerActions,2)); 
            tpc2_2 = txPowerActions(x);
            % Third max value
            [val3, ix3] = max(a(a<max(a(a<max(a)))));
            [ch3_1, ~, x] = val2indexes(possibleActions(allCombs(ix3,1)), size(channelActions,2), size(ccaActions,2), size(txPowerActions,2)); 
            tpc3_1 = txPowerActions(x);
            [ch3_2, ~, x] = val2indexes(possibleActions(allCombs(ix3,2)), size(channelActions,2), size(ccaActions,2), size(txPowerActions,2)); 
            tpc3_2 = txPowerActions(x);   

            disp(['Probability of going from ' num2str(allCombs(ix1,1)) ' (ch=' num2str(ch1_1) '/tpc=' num2str(tpc1_1) ')' ...
                ' to ' num2str(allCombs(ix1,2)) ' (ch=' num2str(ch1_2) '/tpc=' num2str(tpc1_2) ')' ...
                ' = ' num2str(val1/totalIterations)])

            disp(['Probability of going from ' num2str(allCombs(ix2,1)) ' (ch=' num2str(ch2_1) '/tpc=' num2str(tpc2_1) ')' ...
                ' to ' num2str(allCombs(ix2,2)) ' (ch=' num2str(ch2_2) '/tpc=' num2str(tpc2_2) ')' ...
                ' = ' num2str(val2/totalIterations)])

            disp(['Probability of going from ' num2str(allCombs(ix3,1)) ' (ch=' num2str(ch3_1) '/tpc=' num2str(tpc3_1) ')' ...
                ' to ' num2str(allCombs(ix3,2)) ' (ch=' num2str(ch3_2) '/tpc=' num2str(tpc3_2) ')' ...
                ' = ' num2str(val3/totalIterations)])

        end             
    end
    
end