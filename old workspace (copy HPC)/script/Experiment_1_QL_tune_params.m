% Multi-Agent Reinforcement Learning Methods to Minimize OBSS Interferences
% Authors - Francesc Wilhelmi, Boris Bellalta, Anders Jonsson, Cristina Cano

% EXPERIMENT EXPLANATION:
% By using a simple grid of 4 WLANs sharing 2 channels, we want to test several values of
% gamma, alpha and initial epsilon to evaluate the performance of
% Q-learning for each of them. We compare the obtained results with the
% optimal configurations in terms of proportional fairness and aggregate
% throughput.

clc
clear all

% Add paths to methods folders
addpath(genpath('Power Management Methods/'));
addpath(genpath('Throughput Calculation Methods/'));
addpath(genpath('Network Generation Methods/'));
addpath(genpath('Reinforcement Learning Methods/'));
addpath(genpath('Reinforcement Learning Methods/Action Selection Methods/'));
addpath(genpath('Auxiliary Methods/'));

nWorkers = 8;
parpool('local', nWorkers)

disp('-----------------------')
disp('Q-learning: finding the best parameters')
disp('-----------------------')

%% DEFINE THE VARIABLES TO BE USED

% GLOBAL VARIABLES
n_WLANs = 4;                    % Number of WLANs in the map
n_agents = 4;                   % Number of WLANs implementing Q-learning
MAX_CONVERGENCE_TIME = 100;
MIN_SAMPLE_CONSIDER = MAX_CONVERGENCE_TIME/2 + 1;
MAX_LEARNING_ITERATIONS = 1;    % Maximum number of learning iterations done by each WLAN inside a general iteration
TOTAL_ROUNDS = 10;              % Number of TOTAL repetitions to take the average
plot_results = 0;               % To plot or not the results at the end of the simulation

% WLAN object to be modified for each number of coexistent nodes
global wlan
nChannels = 2;              % Number of available channels (from 1 to NumChannels)
noise = -100;               % Floor noise (dBm)

% Definition of actions:
global actions_ch actions_cca actions_tpc
actions_ch = 1:nChannels;       % nChannels possible channels
actions_cca = [-82];            % One CCA level (dBm) -> meaningless (all interferences are considered)
actions_tpc = [5 10 15 20];     % 4 different levels of TPC (dBm)

% Each state represents an [i,j,k] combination for indexes on "channels", "CCA" and "TxPower"
global possible_actions
possible_actions = 1:(size(actions_ch,2)*size(actions_cca,2)*size(actions_tpc,2));
% Total number of actions
K = size(possible_actions,2);
% All possible combinations of configurations for the entire scenario
possible_comb = allcomb(possible_actions,possible_actions,possible_actions,possible_actions);

% Q-learning parameters
initial_epsilon = 1;            % Initial Exploration coefficient
final_epsilon = 0;            % Minimum value that epsilon can achieve

updateMode = 1;

gamma_epsilon_pairs = [.95 1; 0.5 1; .05 1; .95 .5; .5 .5; .05 .5];
alpha = 0:.1:1;                 % Learning Rate

% wlan = GenerateRandomNetwork3D(n_WLANs, nChannels, 0); % RANDOM CONFIGURATION
wlan = GenerateNetwork3D(n_WLANs, nChannels, 'grid', 2, 0); % Grid configuration
%DrawNetwork3D(wlan)
    
%% ITERATE FOR NUMBER OF REPETITIONS (TO TAKE THE AVERAGE) 
tpt_evolution_per_wlan_ql = cell(1, TOTAL_ROUNDS);

avg_tpt_evolution_ql = cell(1, TOTAL_ROUNDS);
fairness_evolution = cell(1, TOTAL_ROUNDS);

avg_tpt_experienced_ql = cell(1, TOTAL_ROUNDS);
std_tpt_experienced_ql = cell(1, TOTAL_ROUNDS);

aggregate_tpt = cell(1, TOTAL_ROUNDS);

avg_fairness_experienced = cell(1, TOTAL_ROUNDS);

parfor (iter = 1:TOTAL_ROUNDS, nWorkers)

    disp('------------------------------------')
    disp(['ROUND ' num2str(iter) '/' num2str(TOTAL_ROUNDS)])
    disp('------------------------------------')

% Setup the scenario: generate WLANs and initialize states and actions

    for a = 1:size(alpha,2)

        for g_e = 1:size(gamma_epsilon_pairs,1)

            tpt_evolution_per_wlan_ql{iter}  = QlearningMethod(wlan, MAX_CONVERGENCE_TIME, MAX_LEARNING_ITERATIONS, ...
                                            gamma_epsilon_pairs(g_e,1), gamma_epsilon_pairs(g_e,2), alpha(a), updateMode, ...
                                            actions_ch, actions_cca, actions_tpc, noise);

            for j=1:MAX_CONVERGENCE_TIME
                avg_tpt_evolution_ql{iter}(j) = mean(tpt_evolution_per_wlan_ql{iter}(j,:));
                fairness_evolution{iter}(j) = JainsFairness(tpt_evolution_per_wlan_ql{iter}(j,:));
            end

            avg_tpt_experienced_ql{iter}(a, g_e) = mean(avg_tpt_evolution_ql{iter}(MIN_SAMPLE_CONSIDER:MAX_CONVERGENCE_TIME));              
            std_tpt_experienced_ql{iter}(a, g_e) = std(avg_tpt_evolution_ql{iter}(MIN_SAMPLE_CONSIDER:MAX_CONVERGENCE_TIME)); 

            aggregate_tpt{iter}(a, g_e) = mean(sum(tpt_evolution_per_wlan_ql{iter}(MIN_SAMPLE_CONSIDER:MAX_CONVERGENCE_TIME, :), 2));
            avg_fairness_experienced{iter}(a, g_e) = mean(sum(fairness_evolution{iter}(MIN_SAMPLE_CONSIDER:MAX_CONVERGENCE_TIME), 2));
            
        end

    end

end

%% PLOT THE RESULTS

% Compute the average results found for the aggregated throughput
aux_array = zeros(size(alpha, 2), size(gamma_epsilon_pairs, 1));
for a = 1 : size(alpha, 2)     
    for g_e = 1 : size(gamma_epsilon_pairs, 1)
        for i = 1 : TOTAL_ROUNDS
            aux_array(a, g_e) = aux_array(a, g_e) + aggregate_tpt{i}(a, g_e);
        end
    end
end
mean_aggregate_tpt = aux_array / TOTAL_ROUNDS;
disp('mean_aggregate_tpt')
disp(mean_aggregate_tpt)
% Compute the standard deviation for the aggregated throughput
aux_array_std = zeros(size(alpha, 2), size(gamma_epsilon_pairs, 1));
for a = 1 : size(alpha, 2)     
    for g_e = 1 : size(gamma_epsilon_pairs, 1)
        for i = 1:TOTAL_ROUNDS
            aux_array_std(a, g_e) = aux_array_std(a, g_e) + ((aggregate_tpt{i}(a, g_e) - mean_aggregate_tpt(a, g_e))^2);
        end
    end
end
std_aggregate_tpt = sqrt(aux_array_std / TOTAL_ROUNDS);
disp('std_aggregate_tpt')
disp(std_aggregate_tpt)

save('test.mat', '-v7.3')

delete(gcp('nocreate'))