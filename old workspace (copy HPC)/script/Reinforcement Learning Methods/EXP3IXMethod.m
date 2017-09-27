% Multi-Agent Reinforcement Learning Methods to Minimize OBSS Interferences
% Authors - Francesc Wilhelmi, Boris Bellalta, Anders Jonsson, Cristina Cano
function [tpt_experienced_by_WLAN] = EXP3IXMethod(wlan, MAX_CONVERGENCE_TIME, MAX_LEARNING_ITERATIONS, ...
                                                    gamma_0, eta_0, actions_ch, actions_cca, actions_tpc, noise)
% EXP3IXMethod Given an OBSS, applies EXP3-IX to maximize the experienced throughput
%
%   OUTPUT: 
%       * tpt_experienced_by_WLAN - throughput experienced by each WLAN
%         for each of the iterations done
%   INPUT: 
%       * wlan - wlan object containing information about all the WLANs
%       * MAX_CONVERGENCE_TIME - maximum number of iterations
%       * MAX_LEARNING_ITERATIONS - maximum number of iterations that an
%       agent performs at the same time
%       * initial_epsilon - exploration coefficient (Q-learning)
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
    
    selected_arm = action_ix_per_wlan;                           % Initialize arm selection for each WLAN by using the initial action
    weights_per_arm = ones(n_WLANs, K);                         % Initialize weight to 1 for each action
    % Initialize arms probabilities
    initial_prob = (1 - gamma_0) * 1/K + gamma_0 / K;
    p_per_arm = initial_prob * ones(n_WLANs, K);
    eta = eta_0 * ones(1, n_WLANs);
    gamma = gamma_0 * ones(1, n_WLANs);
    D = log(K);
    S_t = D * ones(1, n_WLANs);
    m = 1;      
    cumulative_estimated_loss = zeros(1, n_WLANs);

                        
    %% ITERATE UNTIL CONVERGENCE OR MAXIMUM CONVERGENCE TIME            
    t = 1;
    while(t < MAX_CONVERGENCE_TIME + 1) 

        % Assign turns to WLANs randomly 
        order = randperm(n_WLANs);  

        for i=1:n_WLANs % Iterate sequentially for each agent in the random order                      

            learning_iteration = 1;

            while(learning_iteration <= MAX_LEARNING_ITERATIONS)
               
                if t > 1                   
                    p_per_arm(order(i), :) = weights_per_arm(order(i), :) ./ ...
                        sum(weights_per_arm(order(i), :));
                end
                
                % Draw an action according to probabilites distribution
                selected_arm(order(i)) = randsample(1:K, 1, true, p_per_arm(order(i),:));                                    
                [a, ~, c] = val2indexes(selected_arm(order(i)), size(actions_ch,2), size(actions_cca,2), size(actions_tpc,2));
                wlan_aux(order(i)).channel = a;   
                %wlan_aux(order(i)).CCA = actions_cca(b);
                wlan_aux(order(i)).PTdBm = actions_tpc(c);  
                     
                % Compute the reward with the throughput obtained in the round after applying the action
                power_matrix = PowerMatrix(wlan_aux);
                tpt_after_action = computeThroughputFromSINR(wlan_aux, power_matrix, noise);  % bps                                    
                
                for wlan_i = 1:n_WLANs
                    %rw = sqrt(tpt_after_action(wlan_i) / (upper_bound_tpt_per_wlan(wlan_i)^2));
                    loss = 1 - (tpt_after_action(wlan_i) / upper_bound_tpt_per_wlan(wlan_i));
                    estimated_loss = loss / (p_per_arm(wlan_i, selected_arm(wlan_i)) + gamma(wlan_i));
                    % Compute the estimated reward
                    %estimated_reward = rw / p_per_arm(wlan_i, selected_arm(wlan_i));
                    % Update the weight for the selected action according to the estimated reward
                    weights_per_arm(wlan_i, selected_arm(wlan_i)) = ...
                        weights_per_arm(wlan_i, selected_arm(wlan_i)) * ...
                        exp(- eta(wlan_i) * estimated_loss);
                    
                    % Compute the cumulative estimated loss
                    cumulative_estimated_loss(wlan_i) = ...
                        cumulative_estimated_loss(wlan_i) + estimated_loss;  
                    
                    eta(wlan_i) = sqrt(D/S_t(wlan_i));
                    gamma(wlan_i) = m * eta(wlan_i);  
                    % Update the sum of expected losses
                    S_t(wlan_i) = 1/D + cumulative_estimated_loss(wlan_i);
                    
                end                                                

                learning_iteration = learning_iteration + 1;

            end

        end
        
        power_matrix = PowerMatrix(wlan_aux);    
        tpt_experienced_by_WLAN(t,:) = computeThroughputFromSINR(wlan_aux, power_matrix, noise);  % bps    
        % Increase the number of 'learning iterations' of a WLAN
        t = t + 1; 
    
    end
    
    % Print the preferred action per wlan
%     for i=1:n_WLANs
%         [val, ix] = max(p_per_arm(i, :));
%         [a, ~, c] = val2indexes(possible_actions(ix), size(actions_ch,2), size(actions_cca,2), size(actions_tpc,2));  
%         disp(['   * WLAN' num2str(i) ':'])
%         disp(['       - Channel:' num2str(a)])
%         disp(['       - TPC:' num2str(actions_tpc(c))])
%     end
        
end

