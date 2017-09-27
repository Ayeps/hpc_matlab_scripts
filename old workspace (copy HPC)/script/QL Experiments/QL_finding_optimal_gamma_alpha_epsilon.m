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
TOTAL_ROUNDS = 100;             % Number of TOTAL repetitions to take the average
plot_results = 0;               % To plot or not the results at the end of the simulation

% WLAN object to be modified for each number of coexistent nodes
global wlan
nChannels = 2;              % Number of available channels (from 1 to NumChannels)
noise = -100;               % Floor noise (dBm)

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

% Q-learning parameters
initial_epsilon = 1;            % Initial Exploration coefficient
final_epsilon = .01;            % Minimum value that epsilon can achieve
gamma_ql = 0:.1:1;              % Discount factor Q-learning
alpha = 0:.1:1;                 % Learning rate

%% ITERATE FOR NUMBER OF REPETITIONS (TO TAKE THE AVERAGE)
for iter = 1:TOTAL_ROUNDS

    disp('------------------------------------')
    disp(['ROUND ' num2str(iter) '/' num2str(TOTAL_ROUNDS)])
    disp('------------------------------------')

    % Setup the scenario: generate WLANs and initialize states and actions
    wlan = GenerateNetwork3D(n_WLANs, nChannels, 'grid', 0);
    %power_matrix = PowerMatrix(wlan); 
    %upper_bound_tpt_per_wlan = computeMaxBoundThroughput(wlan, power_matrix, noise);
    
    %maximum_achievable_throughput = computeTptAllCombinations(wlan, actions_ch, actions_cca, actions_tpc, noise);

    for g = 1:size(gamma_ql,2)
        
            for a = 1:size(alpha,2)
                
                 %for e = 1:size(initial_epsilon,2)
        
                    tpt_evolution_per_wlan_ql{iter}  = QlearningMethod(wlan, MAX_CONVERGENCE_TIME, MAX_LEARNING_ITERATIONS, ...
                                                    gamma_ql(g), initial_epsilon, final_epsilon, alpha(a), actions_ch, actions_cca, actions_tpc, noise);
                    for j=1:MAX_CONVERGENCE_TIME-1
                        avg_tpt_evolution_ql{iter}(j) = mean(tpt_evolution_per_wlan_ql{iter} (j,:));
                    end

                    avg_tpt_experienced_ql{iter}(g,a) = mean(avg_tpt_evolution_ql{iter});              
                    std_tpt_experienced_ql{iter}(g,a) = std(avg_tpt_evolution_ql{iter}); 
                    
                 %end
                
            end
        
    end
    
%     % Static
%     tpt_evolution_per_wlan_static{iter} = StaticMethod(wlan, noise);
% 
%     % Random
%     tpt_evolution_per_wlan_random{iter}  = RandomMethod(wlan, MAX_CONVERGENCE_TIME, MAX_LEARNING_ITERATIONS, ...
%                                                actions_ch, actions_cca, actions_tpc, noise);

end

mean_tpt_results_ql = mean(cat(TOTAL_ROUNDS+1, avg_tpt_experienced_ql{:}), TOTAL_ROUNDS+1);
mean_std_results_ql = mean(cat(TOTAL_ROUNDS+1, std_tpt_experienced_ql{:}), TOTAL_ROUNDS+1);

figure
surf(gamma_ql,alpha,mean_tpt_results_ql)
xlabel('\gamma','FontSize',14)
ylabel('\alpha','FontSize',14)
zlabel('Network Throughput (Mbps)','FontSize',14)

% Maximum value and index:
[maxVal,ix] = max(mean_tpt_results_ql(:));
[m,n] = ind2sub(size(mean_tpt_results_ql),ix);
disp(['Optimal gamma = ' num2str(gamma_ql(m))]);
disp(['Optimal alpha = ' num2str(alpha(n))]);

figure
bar(mean_tpt_results_ql)
hold on
errorbar(mean_tpt_results_ql, mean_std_results_ql,'r.')
xticks(1:size(gamma_ql,2))
l = {};
for i=1:size(gamma_ql,2)
    l = [l ['\gamma = ' num2str(gamma_ql(i))]];
end
xticklabels(l)
ylabel('Average Experienced Throughput (Mbps)')