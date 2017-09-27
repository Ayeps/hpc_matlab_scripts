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
TOTAL_ROUNDS = 5;               % Number of TOTAL repetitions to take the average
plot_results = 0;               % To plot or not the results at the end of the simulation

nChannels = 6;              % Number of available channels (from 1 to NumChannels)
noise = -100;               % Floor noise (dBm)
B_max = 20e6;               % Available bandwidth (Hz) - Consider only one 20 MHz channel
initial_epsilon = 1;        % Initial Exploration coefficient
final_epsilon = .01;        % Minimum value that epsilon can achieve
gamma_ql = 0:.1:1;  % Discount factor Q-learning
gamma_bandits = .1;         % Discount rate Bandits
alpha = .9;                 % Learning rate

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
    
    %maximum_achievable_throughput = computeTptAllCombinations(wlan, actions_ch, actions_cca, actions_tpc, noise);

    for i = 1:size(gamma_ql,2)
        
        disp('++++++++++')
        disp(['Gamma = ' num2str(gamma_ql(i))])
        disp('++++++++++')
        
        tpt_evolution_per_wlan_ql{iter}  = QlearningMethod(wlan, MAX_CONVERGENCE_TIME, MAX_LEARNING_ITERATIONS, ...
                                        gamma_ql(i), initial_epsilon, final_epsilon, alpha, actions_ch, actions_cca, actions_tpc, noise);
        for j=1:MAX_CONVERGENCE_TIME-1
            avg_tpt_evolution_ql{iter}(j) = mean(tpt_evolution_per_wlan_ql{iter} (j,:));
        end
        
        avg_tpt_experienced_ql{iter}(i) = mean(avg_tpt_evolution_ql{iter});              
        std_tpt_experienced_ql{iter}(i) = std(avg_tpt_evolution_ql{iter}); 
        
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