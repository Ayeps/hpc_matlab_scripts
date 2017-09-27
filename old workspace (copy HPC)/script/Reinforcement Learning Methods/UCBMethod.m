% Multi-Agent Reinforcement Learning Methods to Minimize OBSS Interferences
% Authors - Francesc Wilhelmi, Boris Bellalta, Anders Jonsson, Cristina Cano
function [tpt_experienced_by_WLAN] = UCBMethod(wlan, MAX_CONVERGENCE_TIME, MAX_LEARNING_ITERATIONS, ...
                                                    actions_ch, actions_cca, actions_tpc, noise, printInfo)
% UCBMethod - Given an OBSS, applies UCB to maximize the experienced throughput
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
    
    selected_arm = action_ix_per_wlan;              % Initialize arm selection for each WLAN by using the initial action
    upper_confidence_bounds = ones(n_WLANs, K);     % Initialize the upper confidence bound value of each arm 
    times_arm_has_been_played = ones(n_WLANs, K);   % Initialize the times an arm has been played
    current_action = zeros(1, n_WLANs);
    previous_action = selected_arm;
    transitions_counter = zeros(n_WLANs, K^2);
    allcombs = allcomb(1:K, 1:K);
    cumulative_rw_per_wlan = zeros(n_WLANs, K);     % Initialize the cumulative reward obtained by each WLAN for each arm
    mean_rw_per_wlan = zeros(n_WLANs, K);           % Initialize the mean reward obtained by each WLAN for each arm
            
    %% INITIALIZE THE PAYOFF OF EACH ARM    
    t = 1;
    
    % Play each arm once
    for i = 1:n_WLANs
            
        for j = 1:K

            selected_arm(i) = j;
            
            current_action(i) = j;
            ix = find(allcombs(:,1) == previous_action(i) & allcombs(:,2) == current_action(i));
            previous_action(i) = current_action(i);                
            transitions_counter(i, ix) = transitions_counter(i, ix) + 1;
                
            [a, ~, c] = val2indexes(selected_arm(i), size(actions_ch,2), size(actions_cca,2), size(actions_tpc,2));
            wlan_aux(i).channel = a;   
            %wlan_aux(order(i)).CCA = actions_cca(b);
            wlan_aux(i).PTdBm = actions_tpc(c);  

            power_matrix = PowerMatrix(wlan_aux);
            tpt_after_action = computeThroughputFromSINR(wlan_aux, power_matrix, noise);  % bps      

            % Update the reward for the new configuration
            for wlan_i = 1:n_WLANs

                 times_arm_has_been_played(wlan_i, selected_arm(wlan_i)) = times_arm_has_been_played(wlan_i, selected_arm(wlan_i)) + 1;

                 rw = (tpt_after_action(wlan_i)/((upper_bound_tpt_per_wlan(wlan_i))));
                 cumulative_rw_per_wlan(wlan_i, selected_arm(wlan_i)) = cumulative_rw_per_wlan(wlan_i, selected_arm(wlan_i)) + rw;

                 mean_rw_per_wlan(wlan_i, selected_arm(wlan_i)) = ...
                        cumulative_rw_per_wlan(wlan_i, selected_arm(wlan_i)) /...
                        times_arm_has_been_played(wlan_i, selected_arm(wlan_i));  

                 upper_confidence_bounds(wlan_i, selected_arm(wlan_i)) = ...
                        mean_rw_per_wlan(wlan_i, selected_arm(wlan_i)) + ...
                        sqrt(2 * log(t) / times_arm_has_been_played(wlan_i, selected_arm(wlan_i)));

            end 

            if t >= MAX_CONVERGENCE_TIME, break, end
                        
            power_matrix = PowerMatrix(wlan_aux);    
            tpt_experienced_by_WLAN(t, :) = computeThroughputFromSINR(wlan_aux, power_matrix, noise);  % bps   
            t = t + 1; 

        end
    end    

    %% ITERATE UNTIL CONVERGENCE OR MAXIMUM CONVERGENCE TIME      
           
    while(t < MAX_CONVERGENCE_TIME + 1) 

        % Assign turns to WLANs randomly 
        order = randperm(n_WLANs);  

        for i=1:n_WLANs % Iterate sequentially for each agent in the random order                      

            learning_iteration = 1;

            while(learning_iteration <= MAX_LEARNING_ITERATIONS)

                selected_arm(order(i)) = selectActionUCB(upper_confidence_bounds(order(i),:));
                
                current_action(order(i)) = selected_arm(order(i));
                ix = find(allcombs(:,1) == previous_action(order(i)) & allcombs(:,2) == current_action(order(i)));
                previous_action(order(i)) = current_action(order(i));                
                transitions_counter(order(i), ix) = transitions_counter(order(i), ix) + 1;                
                
                times_arm_has_been_played(order(i), selected_arm(order(i))) = times_arm_has_been_played(order(i), selected_arm(order(i))) + 1;
                
                [a, ~, c] = val2indexes(selected_arm(order(i)), size(actions_ch,2), size(actions_cca,2), size(actions_tpc,2));
                wlan_aux(order(i)).channel = a;   
                %wlan_aux(order(i)).CCA = actions_cca(b);
                wlan_aux(order(i)).PTdBm = actions_tpc(c);  

                % Compute the reward with the throughput obtained in the round after applying the action
                power_matrix = PowerMatrix(wlan_aux);                
                tpt_after_action = computeThroughputFromSINR(wlan_aux, power_matrix, noise);  % bps      

                for wlan_i = 1:n_WLANs

                    times_arm_has_been_played(wlan_i, selected_arm(wlan_i)) = times_arm_has_been_played(wlan_i, selected_arm(wlan_i)) + 1;

                    rw = (tpt_after_action(wlan_i)/((upper_bound_tpt_per_wlan(wlan_i))));
                    cumulative_rw_per_wlan(wlan_i, selected_arm(wlan_i)) = cumulative_rw_per_wlan(wlan_i, selected_arm(wlan_i)) + rw;

                    mean_rw_per_wlan(wlan_i, selected_arm(wlan_i)) = ...
                        cumulative_rw_per_wlan(wlan_i, selected_arm(wlan_i)) /...
                        times_arm_has_been_played(wlan_i, selected_arm(wlan_i));                                       

                    upper_confidence_bounds(wlan_i, selected_arm(wlan_i)) = ...
                        mean_rw_per_wlan(wlan_i, selected_arm(wlan_i)) + ...
                        sqrt(2 * log(t) / times_arm_has_been_played(wlan_i, selected_arm(wlan_i)));
                    
                end   

                learning_iteration = learning_iteration + 1;
                
            end

        end
        
        power_matrix = PowerMatrix(wlan_aux);    
        tpt_experienced_by_WLAN(t, :) = computeThroughputFromSINR(wlan_aux, power_matrix, noise);  % bps   
        % Increase the number of 'learning iterations' of a WLAN
        t = t + 1; 
    
    end
       
%     % Print the preferred action per wlan
    for i=1:n_WLANs
        [val, ix] = max(upper_confidence_bounds(i, :));
        [a, ~, c] = val2indexes(possible_actions(ix), size(actions_ch,2), size(actions_cca,2), size(actions_tpc,2));  
        disp(['   * WLAN' num2str(i) ':'])
        disp(['       - Channel:' num2str(a)])
        disp(['       - TPC:' num2str(actions_tpc(c))])
    end
    
    if printInfo
        
        figure('pos',[450 400 500 350])
        axes;
        axis([1 20 30 70]);  

        % Print the preferred action per wlan
        for i=1:n_WLANs      

            times_arm_has_been_played(i, :)/MAX_CONVERGENCE_TIME

            [~, ix] = max(upper_confidence_bounds(i, :));
            [a, ~, c] = val2indexes(possible_actions(ix), size(actions_ch,2), size(actions_cca,2), size(actions_tpc,2));  
            disp(['   * WN' num2str(i) ':'])
            disp(['       - Channel:' num2str(a)])
            disp(['       - TPC:' num2str(actions_tpc(c))])
            subplot(2,2,i)
            bar(1:K, times_arm_has_been_played(i, :)/MAX_CONVERGENCE_TIME);
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
    
end