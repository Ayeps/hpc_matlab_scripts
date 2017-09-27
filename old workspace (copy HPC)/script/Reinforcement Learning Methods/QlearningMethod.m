% Multi-Agent Reinforcement Learning Methods to Minimize OBSS Interferences
% Authors - Francesc Wilhelmi, Boris Bellalta, Anders Jonsson, Cristina Cano
function [tpt_experienced_by_WLAN, Qval] = QlearningMethod(wlan, MAX_CONVERGENCE_TIME, MAX_LEARNING_ITERATIONS, gamma, ...
                                                    initial_epsilon, alpha, updateMode, actions_ch, actions_cca, actions_tpc, noise)
% QlearningMethod - Given an OBSS, applies QLearning to maximize the
% experienced throughput
%   OUTPUT: 
%       * tpt_experienced_by_WLAN - throughput experienced by each WLAN
%         for each of the iterations done
%   INPUT: 
%       * wlan - wlan object containing information about all the WLANs
%       * MAX_CONVERGENCE_TIME - maximum number of iterations
%       * MAX_LEARNING_ITERATIONS - maximum number of iterations that an
%       agent performs at the same time
%       * gamma - discount factor (Q-learning)
%       * initial_epsilon - exploration coefficient (Q-learning)
%       * alpha - learning rate (Q-learning)
%       * actions_ch - set of channels
%       * actions_cca - set of carrier sense thresholds
%       * actions_tpc - set of transmit power values
%       * noise - floor noise in dBm

    % Use a copy of wlan to make operations
    wlan_aux = wlan;

    n_WLANs = size(wlan,2);
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

    Qval = {};
    for i=1:n_WLANs
        % Fill the Q-table of each node with 0's 
        Qval{i} = zeros(1, size(possible_actions, 2));
    end
    
    % Initialize iteration_tracking to keep track of the iterations performed by each WLAN
    iteration_tracking = zeros(1, n_WLANs);

    %% ITERATE UNTIL CONVERGENCE OR MAXIMUM CONVERGENCE TIME            
    tpt_after_action = zeros(1, n_WLANs);   % Initialize the throughput experienced by all the WLANs after a single action
    
    t = 1;
    epsilon = initial_epsilon; 
    cumulative_tpt_experienced_per_WLAN = 0;
    cumulative_fairness = 0;

    while(t < MAX_CONVERGENCE_TIME + 1) 

        % Assign turns to WLANs randomly 
        order = randperm(n_WLANs);  

        for i=1:n_WLANs % Iterate sequentially for each agent in the random order                      

            learning_iteration = 1;
            while(learning_iteration <= MAX_LEARNING_ITERATIONS)
                
                % Select an action according to Q-learning policy
                selected_action = selectActionQLearning(Qval{order(i)}, ...
                    actions_ch, actions_cca, actions_tpc, epsilon);
                
                % Change parameters according to the action obtained
                wlan_aux(order(i)).channel = selected_action(1);   
                %wlan_aux(order(i)).CCA = actions_cca(selected_action(2));
                wlan_aux(order(i)).PTdBm = actions_tpc(selected_action(3));
                
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
                
                %Update Q                                    
                for wlan_i=1:n_WLANs
                    rw = (tpt_after_action(wlan_i) / upper_bound_tpt_per_wlan(wlan_i));                                      
                    Qval{wlan_i}(action_ix_per_wlan(wlan_i)) = ...
                        (1 - alpha) * Qval{wlan_i}(action_ix_per_wlan(wlan_i)) + ...
                        (alpha * rw + gamma * (max(Qval{wlan_i})));
                end
                
                % Update the exploration coefficient according to the inputted mode
                if updateMode == 0
                    epsilon = initial_epsilon / t;
                elseif updateMode == 1 
                    epsilon = initial_epsilon / sqrt(t);
                end
                            
                learning_iteration = learning_iteration + 1;
                
            end

        end
        
        power_matrix = PowerMatrix(wlan_aux);        
        tpt_experienced_by_WLAN(t,:) = computeThroughputFromSINR(wlan_aux, power_matrix, noise);  % bps 

        % Increase the number of 'learning iterations' of a WLAN
        t = t + 1; 
    
    end
      
    % Display the best configurations seen so far for each node    
    disp(['Aggregate throughput experienced on average: ' num2str(mean(sum(tpt_experienced_by_WLAN,2)))])
    disp(['Fairness on average: ' num2str(mean(JainsFairness(tpt_experienced_by_WLAN)))])
    disp(['Proportional fairness experienced on average: ' num2str(mean(sum(log(tpt_experienced_by_WLAN),2)))])
    for i = 1:n_WLANs
        [~, ix] = max(Qval{i});
        disp('---------------')
        disp(['   * WLAN' num2str(i) ' - Best configuration: ' num2str(possible_actions(ix)) ':'])
        [a, ~, c] = val2indexes(possible_actions(ix), size(actions_ch,2), size(actions_cca,2), size(actions_tpc,2));  
        disp(['       - Channel:' num2str(a)])
        disp(['       - TPC:' num2str(actions_tpc(c))])
    end
    
end