% Multi-Agent Reinforcement Learning Methods to Minimize OBSS Interferences
% Authors - Francesc Wilhelmi, Boris Bellalta, Anders Jonsson, Cristina Cano
function [tpt_experienced_by_WLAN] = EXP3MethodBasic(wlan, MAX_CONVERGENCE_TIME, MAX_LEARNING_ITERATIONS, ...
                                                    gamma, eta, actions_ch, actions_cca, actions_tpc, noise, plotInfo)
% EXP3MethodBasic Given an OBSS, applies EXP3 (basic formulation) to maximize the experienced throughput
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
    
    selected_arm = action_ix_per_wlan;              % Initialize arm selection for each WLAN by using the initial action
    weights_per_arm = ones(n_WLANs, K);             % Initialize weight to 1 for each action
    current_action = zeros(1, n_WLANs);
    previous_action = selected_arm;
    times_arm_is_seleceted = zeros(n_WLANs, K);     
    transitions_counter = zeros(n_WLANs, K^2);
    allcombs = allcomb(1:K, 1:K);
    % Initialize arms probabilities
    initial_prob = (1 - gamma) / K + gamma / K;
    p_per_arm = initial_prob * ones(n_WLANs, K);
       
    %% ITERATE UNTIL CONVERGENCE OR MAXIMUM CONVERGENCE TIME           
  
    t = 1;
    
    rw = zeros(n_WLANs, MAX_CONVERGENCE_TIME + 1);
    estimated_reward = zeros(n_WLANs, MAX_CONVERGENCE_TIME + 1);

    while(t < MAX_CONVERGENCE_TIME + 1) 

        % Assign turns to WLANs randomly 
        order = randperm(n_WLANs);  

        for i=1:n_WLANs % Iterate sequentially for each agent in the random order                      

            learning_iteration = 1;

            while(learning_iteration <= MAX_LEARNING_ITERATIONS)
%                
                if t > 1 
                    p_per_arm(order(i), :) = (1 - gamma) * (weights_per_arm(order(i), :) ./ ...
                        sum(weights_per_arm(order(i), :))) + gamma / K;
                end
                
                % Draw an action according to probabilites distribution
                selected_arm(order(i)) = randsample(1:K, 1, true, p_per_arm(order(i),:));      
                
                current_action(order(i)) = selected_arm(order(i));
                ix = find(allcombs(:,1) == previous_action(order(i)) & allcombs(:,2) == current_action(order(i)));
                previous_action(order(i)) = current_action(order(i));                
                transitions_counter(order(i), ix) = transitions_counter(order(i), ix) + 1;
                
                times_arm_is_seleceted(order(i), selected_arm(order(i))) = times_arm_is_seleceted(order(i), selected_arm(order(i))) + 1;
                
                [a, ~, c] = val2indexes(selected_arm(order(i)), size(actions_ch,2), size(actions_cca,2), size(actions_tpc,2));
                wlan_aux(order(i)).channel = a;   
                %wlan_aux(order(i)).CCA = actions_cca(b);
                wlan_aux(order(i)).PTdBm = actions_tpc(c);  
                     
                % Compute the reward with the throughput obtained in the round after applying the action
                power_matrix = PowerMatrix(wlan_aux);
                tpt_after_action = computeThroughputFromSINR(wlan_aux, power_matrix, noise);  % bps 
              
                for wlan_i = 1:n_WLANs
                    
                    rw = (tpt_after_action(wlan_i) / upper_bound_tpt_per_wlan(wlan_i));
                    % Compute the estimated reward                    
                    if rw == 0
                        estimated_reward = 1e-6;
                    else
                        estimated_reward = (rw / p_per_arm(wlan_i, selected_arm(wlan_i)));
                    end

                    % Update the weight for the selected action according to the estimated reward
                    weights_per_arm(wlan_i, selected_arm(wlan_i)) = ...
                        weights_per_arm(wlan_i, selected_arm(wlan_i)) * ...
                        exp((eta * estimated_reward / K));
                    
                    if weights_per_arm(wlan_i, selected_arm(wlan_i)) > 1e20, weights_per_arm(wlan_i, selected_arm(wlan_i)) = 1e20; end
                    
                end                                                           

                learning_iteration = learning_iteration + 1;

            end

        end
        
        power_matrix = PowerMatrix(wlan_aux);    
        tpt_experienced_by_WLAN(t,:) = computeThroughputFromSINR(wlan_aux, power_matrix, noise);  % bps    
        % Increase the number of 'learning iterations' of a WLAN
        t = t + 1; 
    
    end
% 
    
    if plotInfo
        
        figure('pos',[450 400 500 350])
        axes;
        axis([1 20 30 70]);  

        % Print the preferred action per wlan
        for i=1:n_WLANs      

            times_arm_is_seleceted(i, :)/MAX_CONVERGENCE_TIME

            [~, ix] = max(p_per_arm(i, :));
            [a, ~, c] = val2indexes(possible_actions(ix), size(actions_ch,2), size(actions_cca,2), size(actions_tpc,2));  
            disp(['   * WN' num2str(i) ':'])
            disp(['       - Channel:' num2str(a)])
            disp(['       - TPC:' num2str(actions_tpc(c))])
            subplot(2,2,i)
            bar(1:K, times_arm_is_seleceted(i, :)/MAX_CONVERGENCE_TIME);
            hold on
            title(['WN' num2str(i)])
            axis([0 9 0 1])
            xticks(1:8)
            xticklabels(1:8)
            set(gca, 'FontSize', 22)
            % xticklabels({'ch=1/tpc=5','ch=2/tpc=5','ch=1/tpc=10','ch=2/tpc=10','ch=1/tpc=15','ch=2/tpc=15','ch=1/tpc=20','ch=2/tpc=20'})

            a = transitions_counter(i,:);
            % Max value
            [val1, ix1] = max(a);
            [ch1_1, ~, x] = val2indexes(possible_actions(allcombs(ix1,1)), size(actions_ch,2), size(actions_cca,2), size(actions_tpc,2)); 
            tpc1_1 = actions_tpc(x);
            [ch1_2, ~, x] = val2indexes(possible_actions(allcombs(ix1,2)), size(actions_ch,2), size(actions_cca,2), size(actions_tpc,2)); 
            tpc1_2 = actions_tpc(x);
            % Second max value
            [val2, ix2] = max(a(a<max(a)));
            [ch2_1, ~, x] = val2indexes(possible_actions(allcombs(ix2,1)), size(actions_ch,2), size(actions_cca,2), size(actions_tpc,2)); 
            tpc2_1 = actions_tpc(x);
            [ch2_2, ~, x] = val2indexes(possible_actions(allcombs(ix2,2)), size(actions_ch,2), size(actions_cca,2), size(actions_tpc,2)); 
            tpc2_2 = actions_tpc(x);
            % Third max value
            [val3, ix3] = max(a(a<max(a(a<max(a)))));
            [ch3_1, ~, x] = val2indexes(possible_actions(allcombs(ix3,1)), size(actions_ch,2), size(actions_cca,2), size(actions_tpc,2)); 
            tpc3_1 = actions_tpc(x);
            [ch3_2, ~, x] = val2indexes(possible_actions(allcombs(ix3,2)), size(actions_ch,2), size(actions_cca,2), size(actions_tpc,2)); 
            tpc3_2 = actions_tpc(x);   

            disp(['Probability of going from ' num2str(allcombs(ix1,1)) ' (ch=' num2str(ch1_1) '/tpc=' num2str(tpc1_1) ')' ...
                ' to ' num2str(allcombs(ix1,2)) ' (ch=' num2str(ch1_2) '/tpc=' num2str(tpc1_2) ')' ...
                ' = ' num2str(val1/MAX_CONVERGENCE_TIME)])

            disp(['Probability of going from ' num2str(allcombs(ix2,1)) ' (ch=' num2str(ch2_1) '/tpc=' num2str(tpc2_1) ')' ...
                ' to ' num2str(allcombs(ix2,2)) ' (ch=' num2str(ch2_2) '/tpc=' num2str(tpc2_2) ')' ...
                ' = ' num2str(val2/MAX_CONVERGENCE_TIME)])

            disp(['Probability of going from ' num2str(allcombs(ix3,1)) ' (ch=' num2str(ch3_1) '/tpc=' num2str(tpc3_1) ')' ...
                ' to ' num2str(allcombs(ix3,2)) ' (ch=' num2str(ch3_2) '/tpc=' num2str(tpc3_2) ')' ...
                ' = ' num2str(val3/MAX_CONVERGENCE_TIME)])

        end
        
    end
    
    % Return the throughput experienced by each WLAN at each iteration
    % ...
    
end