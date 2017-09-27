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
disp('e-greedy: finding the best parameters')
disp('-----------------------')

%% DEFINE THE VARIABLES TO BE USED

% GLOBAL VARIABLES
n_WLANs = 4;                    % Number of WLANs in the map
n_agents = 4;                   % Number of WLANs implementing Q-learning
MAX_CONVERGENCE_TIME = 10000;   % Maximum convergence time (one period implies the participation of all WLANs)
MAX_LEARNING_ITERATIONS = 1;    % Maximum number of learning iterations done by each WLAN inside a general iteration
TOTAL_ROUNDS = 100;             % Number of TOTAL repetitions to take the average
plot_results = 0;               % To plot or not the results at the end of the simulation

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
possible_comb = allcomb(possible_actions,possible_actions,possible_actions,possible_actions);

initial_epsilon = 0:.1:1;
updateMode = 1;         % 0: epsilon = initial_epsilon / t ; 1: epsilon = epsilon / sqrt(t)

printInfo = 0;
  
% Initialize variables at which to store information per repetition
tpt_evolution_per_wlan_eg = cell(1, TOTAL_ROUNDS);
avg_tpt_evolution_eg = cell(1, TOTAL_ROUNDS);
avg_tpt_experienced_eg = cell(1, TOTAL_ROUNDS);
avg_aggregate_tpt = cell(1, TOTAL_ROUNDS);
proportional_fairness = cell(1, TOTAL_ROUNDS);

parfor (iter = 1:TOTAL_ROUNDS, nWorkers)

    disp('------------------------------------')
    disp(['ROUND ' num2str(iter) '/' num2str(TOTAL_ROUNDS)])
    disp('------------------------------------')
    
    wlan = GenerateNetwork3D(n_WLANs, nChannels, 'grid', 2, 0); % SAFE CONFIGURATION
    
    for e = 1:size(initial_epsilon, 2) 

        tpt_evolution_per_wlan_eg{iter}  = eGreedyMethod(wlan, MAX_CONVERGENCE_TIME, MAX_LEARNING_ITERATIONS, ...
                                          initial_epsilon(e), updateMode, actions_ch, actions_cca, actions_tpc, noise, printInfo);
        for j=1:MAX_CONVERGENCE_TIME
            avg_tpt_evolution_eg{iter}(j) = mean(tpt_evolution_per_wlan_eg{iter}(j,:));
        end
        
        avg_tpt_experienced_eg{iter}(e) = mean(avg_tpt_evolution_eg{iter});              
        avg_aggregate_tpt{iter}(e) = mean(sum(tpt_evolution_per_wlan_eg{iter}, 2));
        proportional_fairness{iter}(e) = mean(sum(log(tpt_evolution_per_wlan_eg{iter}), 2));

    end

end

%% PLOT THE RESULTS
for e = 1:size(initial_epsilon, 2)
    for i = 1:TOTAL_ROUNDS
        aux_array(i,e) = avg_aggregate_tpt{i}(e);
    end
end
mean_aggregate_tpt_per_epsilon = mean(aux_array, 1);
std_aggregate_tpt_per_epsilon = std(aux_array, 1);

disp('mean_aggregate_tpt_per_epsilon')
disp(mean_aggregate_tpt_per_epsilon)
disp('std_aggregate_tpt_per_epsilon')
disp(std_aggregate_tpt_per_epsilon)

save('eg_exp1_workspace.mat')

delete(gcp('nocreate'))