% Multi-Agent Reinforcement Learning Methods to Minimize OBSS Interferences
% Authors - Francesc Wilhelmi, Boris Bellalta, Anders Jonsson, Cristina Cano
function [tpt_experienced_by_WLAN] = RandomMethod(wlan, MAX_CONVERGENCE_TIME, MAX_LEARNING_ITERATIONS, ...
                                                   actions_ch, actions_cca, actions_tpc, noise)
                                               
% RandomMethod Given an OBSS, selects actions randomly to maximize the experienced throughput
%
%   OUTPUT: 
%       * tpt_experienced_by_WLAN - throughput experienced by each WLAN
%         for each of the iterations done
%   INPUT: 
%       * wlan - wlan object containing information about all the WLANs
%       * MAX_CONVERGENCE_TIME - maximum number of iterations
%       * MAX_LEARNING_ITERATIONS - maximum number of iterations that an
%       agent performs at the same time
%       * actions_ch - set of channels
%       * actions_cca - set of carrier sense thresholds
%       * actions_tpc - set of transmit power values
%       * noise - floor noise in dBm

    % Use a copy of wlan to make operations
    wlan_aux = wlan;

    n_WLANs = size(wlan_aux,2);
    % Each state represents an [i,j,k] combination for indexes on "channels", "CCA" and "TxPower"
    possible_actions = 1:(size(actions_ch,2)*size(actions_cca,2)*size(actions_tpc,2));
    % Total number of actions
    K = size(possible_actions,2);
    
    % Find the index of the initial action taken by each WLAN
    initial_action_ix_per_wlan = zeros(1, n_WLANs);
    for i=1:n_WLANs
        [~,index_cca] = find(actions_cca==wlan_aux(i).CCA);
        [~,index_tpc] = find(actions_tpc==wlan_aux(i).PTdBm);
        initial_action_ix_per_wlan(i) = indexes2val(wlan_aux(i).channel, index_cca, index_tpc, size(actions_ch,2), size(actions_cca,2));
    end
    % Initialize the indexes of the taken action
    action_ix_per_wlan = initial_action_ix_per_wlan;                           
    
    % Compute the maximum achievable throughput per WLAN
    power_matrix = PowerMatrix(wlan_aux);     
    upper_bound_tpt_per_wlan = computeMaxBoundThroughput(wlan_aux, power_matrix, noise, max(actions_tpc));
    
    % Initialize iteration_tracking to keep track of the iterations performed by each WLAN
    iteration_tracking = zeros(1, n_WLANs);

    %% ITERATE UNTIL CONVERGENCE OR MAXIMUM CONVERGENCE TIME            
    tpt_after_action = zeros(1, n_WLANs);   % Initialize the throughput experienced by all the WLANs after a single action
    
    t = 1;
    cumulative_tpt_experienced_per_WLAN = 0;
    cumulative_fairness = 0;

    while(t < MAX_CONVERGENCE_TIME + 1) 

        % Assign turns to WLANs randomly 
        order = randperm(n_WLANs);  

        for i=1:n_WLANs % Iterate sequentially for each agent in the random order                      

            learning_iteration = 1;

            while(learning_iteration <= MAX_LEARNING_ITERATIONS)
                
                % Select an action randomly
                random_action_ix = round(((K-1)*rand() + 1));
                [action1, ~, action3] = val2indexes(random_action_ix, size(actions_ch,2), size(actions_cca,2), size(actions_tpc,2));

                % Change parameters according to the action obtained
                wlan_aux(order(i)).channel = action1;   
                wlan_aux(order(i)).PTdBm = actions_tpc(action3);
                
                % Prepare the next state according to the actions performed on the current state
                [~,index_cca] = find(actions_cca==wlan_aux(order(i)).CCA);
                [~,index_tpc] = find(actions_tpc==wlan_aux(order(i)).PTdBm);
                action_ix_per_wlan(order(i)) =  indexes2val(wlan_aux(order(i)).channel, index_cca, ...
                    index_tpc, size(actions_ch,2), size(actions_cca,2));
                
                % Compute the reward with the throughput obtained in the round after applying the action
                power_matrix = PowerMatrix(wlan_aux);        
                tpt_after_action = computeThroughputFromSINR(wlan_aux, power_matrix, noise);  % bps                                    
                cumulative_tpt_experienced_per_WLAN = cumulative_tpt_experienced_per_WLAN + tpt_after_action;
                cumulative_fairness = cumulative_fairness + JainsFairness(tpt_after_action);

                learning_iteration = learning_iteration + 1;
                
            end

        end
        
        power_matrix = PowerMatrix(wlan_aux);        
        tpt_experienced_by_WLAN(t,:) = computeThroughputFromSINR(wlan_aux, power_matrix, noise);  % bps 

        % Increase the number of 'learning iterations' of a WLAN
        t = t + 1; 
    
    end
       
end