%%
% Multi-Agent Reinforcement Learning Methods to Minimize OBSS Interferences
% Decentralized Q-learning
% Authors - Francesc Wilhelmi, Boris Bellalta, Anders Jonsson, Cristina Cano

%%
clc
clear all

% Add paths to methods folders
addpath(genpath('Power Management Methods/'));
addpath(genpath('Throughput Calculation Methods/'));
addpath(genpath('Network Generation Methods/'));
addpath(genpath('Reinforcement Learning Methods/'));
addpath(genpath('Reinforcement Learning Methods/Action Selection Methods/'));
addpath(genpath('Auxiliary Methods/'));

disp('-----------------------')
disp('DECENTRALIZED LEARNING')
disp('-----------------------')
% GLOBAL VARIABLES
n_WLANs = 4;                    % Number of WLANs in the map
n_agents = 4;                   % Number of WLANs implementing Q-learning
MAX_CONVERGENCE_TIME = 1000;    % Maximum convergence time (one period implies the participation of all WLANs)
MAX_LEARNING_ITERATIONS = 1;    % Maximum number of learning iterations done by each WLAN inside a general iteration
TOTAL_ROUNDS = 1;               % Number of TOTAL repetitions to take the average
plot_results = 0;               % To plot or not the results at the end of the simulation

nChannels = 6;              % Number of available channels (from 1 to NumChannels)
noise = -100;               % Floor noise (dBm)
B_max = 20e6;               % Available bandwidth (Hz) - Consider only one 20 MHz channel
initial_epsilon = 1;        % Initial Exploration coefficient
final_epsilon = .01;        % Minimum value that epsilon can achieve
gamma = [0 .2 .4 .6 .8 1];  % Discount factor Q-learning
gamma_bandits = .1;         % Discount rate Bandits
alpha = .9;                 % Learning rate

% LEARNING APPROACHES
%   1: Random
%   2: Q-learning
%   3: Bandits - epsilon-greedy
%   4: Bandits - EXP3
%   5: Bandits - UCB

% List of names
approaches_names = {'Static', 'Random', 'QL', 'EG', 'UCB', 'EXP'};
%approaches_names = {'Random', 'QL', 'EG', 'UCB', 'EXP3'};
%approaches_names = {'Random'};
% DEFINES
RANDOM = 8;
QLEARNING = 1;
BANDITS_EGREEDY = 13;
BANDITS_EXP3 = 3;
BANDITS_UCB = 2;
BANDITS_EXP3_IX = 6;

% Definition of actions:
global actions_ch actions_cca actions_tpc
actions_ch = 1:nChannels;       % 6 possible channels
actions_cca = [-82];            % One CCA level (dBm) -> meaningless (all interferences are considered)
actions_tpc = [5 10 15 20];     % 4 different levels of TPC (dBm)

% Each state represents an [i,j,k] combination for indexes on "channels", "CCA" and "TxPower"
global possible_actions
possible_actions = 1:(size(actions_ch,2)*size(actions_cca,2)*size(actions_tpc,2));
% Total number of actions
K = size(possible_actions,2);

% WLAN object to be modified for each number of coexistent nodes
global wlan

%% ITERATE FOR NUMBER OF REPETITIONS (TO TAKE THE AVERAGE)
for iter = 1:TOTAL_ROUNDS

    disp('------------------------------------')
    disp(['ROUND ' num2str(iter) '/' num2str(TOTAL_ROUNDS)])
    disp('------------------------------------')

    % Setup the scenario: generate WLANs and initialize states and actions
    wlan = GenerateNetwork3D(n_WLANs, nChannels, 'grid', 0);
    power_matrix = PowerMatrix(wlan); 
    upper_bound_tpt_per_wlan = computeMaxBoundThroughput(wlan, power_matrix, noise);
    
    maximum_achievable_throughput = computeTptAllCombinations(wlan, actions_ch, actions_cca, actions_tpc, noise);

%     % Static
%     tpt_evolution_per_wlan_static{iter} = StaticMethod(wlan, noise);
% 
%     % Random
%     tpt_evolution_per_wlan_random{iter}  = RandomMethod(wlan, MAX_CONVERGENCE_TIME, MAX_LEARNING_ITERATIONS, ...
%                                                actions_ch, actions_cca, actions_tpc, noise);
    % Q-learning
    tpt_evolution_per_wlan_ql{iter}  = QlearningMethod(wlan, MAX_CONVERGENCE_TIME, MAX_LEARNING_ITERATIONS, ...
                                            0, initial_epsilon, final_epsilon, alpha, actions_ch, actions_cca, actions_tpc, noise);
%     % e-greedy
%     tpt_evolution_per_wlan_eg{iter}  = eGreedyMethod(wlan, MAX_CONVERGENCE_TIME, MAX_LEARNING_ITERATIONS, ...
%                                                 initial_epsilon, actions_ch, actions_cca, actions_tpc, noise);
%     % UCB
%     tpt_evolution_per_wlan_ucb{iter}  = UCBMethod(wlan, MAX_CONVERGENCE_TIME, MAX_LEARNING_ITERATIONS, ...
%                                                 actions_ch, actions_cca, actions_tpc, noise);
%     % EXP3
%     tpt_evolution_per_wlan_exp3{iter}  = EXP3Method(wlan, MAX_CONVERGENCE_TIME, MAX_LEARNING_ITERATIONS, ...
%                                                 gamma_bandits, actions_ch, actions_cca, actions_tpc, noise);
% 
%         for g = 1:size(gamma,2)
% 
%             tpt_evolution_per_wlan = QlearningMethod(wlan, MAX_CONVERGENCE_TIME, MAX_LEARNING_ITERATIONS, ...
%                                                     gamma(g), epsilon, alpha, actions_ch, actions_cca, actions_tpc, noise);
% 
        
        for i=1:MAX_CONVERGENCE_TIME-1
            avg_tpt_evolution_random{iter}(i) = mean(tpt_evolution_per_wlan_random{iter} (i,:));
            avg_tpt_evolution_ql{iter}(i) = mean(tpt_evolution_per_wlan_ql{iter} (i,:));
            avg_tpt_evolution_eg{iter}(i) = mean(tpt_evolution_per_wlan_eg{iter} (i,:));
            avg_tpt_evolution_ucb{iter}(i) = mean(tpt_evolution_per_wlan_ucb{iter} (i,:));
            avg_tpt_evolution_exp3{iter}(i) = mean(tpt_evolution_per_wlan_exp3{iter} (i,:));
        end
        
        avg_tpt_experienced_static{iter} = mean(tpt_evolution_per_wlan_static{iter});
        avg_tpt_experienced_random{iter} = mean(avg_tpt_evolution_random{iter});
        avg_tpt_experienced_ql{iter} = mean(avg_tpt_evolution_ql{iter});
        avg_tpt_experienced_eg{iter} = mean(avg_tpt_evolution_eg{iter});
        avg_tpt_experienced_ucb{iter} = mean(avg_tpt_evolution_ucb{iter});
        avg_tpt_experienced_exp3{iter} = mean(avg_tpt_evolution_exp3{iter});
end

mean_tpt_results_static = mean(cat(TOTAL_ROUNDS+1, avg_tpt_experienced_static{:}), TOTAL_ROUNDS+1);
mean_tpt_results_random = mean(cat(TOTAL_ROUNDS+1, avg_tpt_experienced_random{:}), TOTAL_ROUNDS+1);
mean_tpt_results_ql = mean(cat(TOTAL_ROUNDS+1, avg_tpt_experienced_ql{:}), TOTAL_ROUNDS+1);
mean_tpt_results_eg = mean(cat(TOTAL_ROUNDS+1, avg_tpt_experienced_eg{:}), TOTAL_ROUNDS+1);
mean_tpt_results_ucb = mean(cat(TOTAL_ROUNDS+1, avg_tpt_experienced_ucb{:}), TOTAL_ROUNDS+1);
mean_tpt_results_exp3 = mean(cat(TOTAL_ROUNDS+1, avg_tpt_experienced_exp3{:}), TOTAL_ROUNDS+1);

bar(1:6, [mean_tpt_results_static mean_tpt_results_random mean_tpt_results_ql mean_tpt_results_eg mean_tpt_results_ucb mean_tpt_results_exp3]);
xticks(1:6)
xticklabels(approaches_names)

%             plot(1:MAX_CONVERGENCE_TIME-1, avg_tpt_evolution);
% 
%             avg_tpt_experienced(g) = mean(avg_tpt_evolution(g,:));
%             std_tpt_experienced(g) =std(avg_tpt_evolution(g,:));
% 
%             avg_tpt_experienced_per_wlan(g,:) = mean(tpt_evolution_per_wlan);
%             std_tpt_experienced_per_wlan(g,:) =std(tpt_evolution_per_wlan);
% 
%         end

%         figure
%         bar(avg_tpt_experienced)
%         hold on
%         errorbar(avg_tpt_experienced, std_tpt_experienced,'r.')
%         xticks(1:size(gamma,2))
%         xticklabels({'Gamma = 0', 'Gamma = 0.2', 'Gamma = 0.4', 'Gamma = 0.6', 'Gamma = 0.8', 'Gamma = 1'})
%         
%         ylabel('Average Experienced Throughput (Mbps)')


% 
% 
%                         case BANDITS_EXP3_IX 
%                             while(learning_iteration <= MAX_LEARNING_ITERATIONS && current_state(order(i)) == 1) 
%                                if repetition_number > 1
% %                                         p_per_arm(order(i), :) = (1 - gamma_EXP3(order(i))) * (weights_per_arm(order(i), :) ./ ...
% %                                             sum(weights_per_arm(order(i), :)) + gamma_EXP3(order(i)) / size(possible_actions, 2)); 
%                                       p_per_arm(order(i), :) = weights_per_arm(order(i), :) ./ sum(weights_per_arm(order(i), :));      
%                                 end                             
%                                 % Draw an action according to probabilites distribution
%                                 selected_arm(order(i)) = randsample(1:K, 1, true, p_per_arm(order(i),:));                                    
%                                 [a, b, c] = val2indexes(selected_arm(order(i)), nChannels, size(actions_cca,2), size(actions_tpc,2));
%                                 wlan(order(i)).channel = a;   
%                                 wlan(order(i)).CCA = actions_cca(b);
%                                 wlan(order(i)).PTdBm = actions_tpc(c);                                                                        
%                                 % Compute throughput of each WLAN after all have acted
%                                 power_matrix = PowerMatrix(wlan); 
%                                 tpt_after_action = computeThroughputFromSINR(wlan, power_matrix, noise); % bps                                    
%                                 cumulative_tpt_experienced_per_WLAN = cumulative_tpt_experienced_per_WLAN + tpt_after_action;
%                                 cumulative_fairness = cumulative_fairness + JainsFairness(tpt_after_action);
%                                 num_times_tpt_has_been_computed = num_times_tpt_has_been_computed + 1;                                                                                                           
%                                 for wlan_i = 1:n_agents(n)
%                                     % Compute the reward according to the experienced throughput                                    
%                                     rw = computeRewardEXP3(tpt_after_action(wlan_i), desired_tpt(r,wlan_i));
%                                     % Compute the estimated loss
%                                     if rw > B_max, B_max = rw; end
%                                     noticed_loss = B_max - rw;                                        
%                                     estimated_loss(wlan_i, selected_arm(wlan_i)) = noticed_loss ...
%                                         / (p_per_arm(wlan_i, selected_arm(wlan_i)) + gamma_EXP3(wlan_i));     
%                                     % Update the weight for the selected action according to the estimated reward
%                                     weights_per_arm(wlan_i, selected_arm(wlan_i)) = ...
%                                         weights_per_arm(wlan_i, selected_arm(wlan_i)) * ...
%                                         exp(-eta_EXP3(wlan_i) * estimated_loss(wlan_i, selected_arm(wlan_i)));                                        
%                                     % Compute the cumulative estimated loss
%                                     cumulative_estimated_loss(wlan_i) = ...
%                                         cumulative_estimated_loss(wlan_i) + estimated_loss(wlan_i, selected_arm(wlan_i));    
%                                     % Update the tuning parameters gamma and eta
%                                     eta_EXP3(wlan_i) = sqrt(D/S_t(wlan_i));
%                                     gamma_EXP3(wlan_i) = m * eta_EXP3(wlan_i);  
%                                     % Update the sum of expected losses
%                                     S_t(wlan_i) = 1/D + cumulative_estimated_loss(wlan_i);
%                                 end   
% 
%                                 learning_iteration = learning_iteration + 1;  
% 
%                             end
