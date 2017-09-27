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
disp('EXP3: finding the best parameters')
disp('-----------------------')

%% DEFINE THE VARIABLES TO BE USED

% GLOBAL VARIABLES
n_WLANs = 4;                    % Number of WLANs in the map
n_agents = 4;                   % Number of WLANs implementing Q-learning
MAX_CONVERGENCE_TIME = 10000;    % Maximum convergence time (one period implies the participation of all WLANs)
MIN_SAMPLE_CONSIDER = MAX_CONVERGENCE_TIME/2 + 1;
MAX_LEARNING_ITERATIONS = 1;    % Maximum number of learning iterations done by each WLAN inside a general iteration
TOTAL_ROUNDS = 100;             % Number of TOTAL repetitions to take the average
plot_results = 0;          

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
possible_comb = allcomb(possible_actions,possible_actions,possible_actions,possible_actions);

% Setup the scenario: generate WLANs and initialize states and actions
    
eta = 0:.1:1;
gamma = 0;

tpt_evolution_per_wlan_exp3 = cell(1, TOTAL_ROUNDS);
avg_tpt_evolution_per_wlan_exp3 = cell(1, TOTAL_ROUNDS);
fairness_evolution = cell(1, TOTAL_ROUNDS);
avg_tpt_experienced_exp3 = cell(1, TOTAL_ROUNDS);
aggregate_tpt = cell(1, TOTAL_ROUNDS);

aggregate_tpt_transitory = cell(1, TOTAL_ROUNDS);
aggregate_tpt_permanent = cell(1, TOTAL_ROUNDS);

%% ITERATE FOR NUMBER OF REPETITIONS (TO TAKE THE AVERAGE)

for iter = 1:TOTAL_ROUNDS

    wlan = GenerateRandomNetwork3D(n_WLANs, nChannels, 0); % SAFE CONFIGURATION
    
    disp('------------------------------------')
    disp(['ROUND ' num2str(iter) '/' num2str(TOTAL_ROUNDS)])
    disp('------------------------------------')

    for g = 1:size(gamma, 2)
    
        for e = 1:size(eta, 2)       
            
        tpt_evolution_per_wlan_exp3{iter}  = EXP3MethodBasic(wlan, MAX_CONVERGENCE_TIME, MAX_LEARNING_ITERATIONS, ...
                                                    gamma(g), eta(e), actions_ch, actions_cca, actions_tpc, noise);

        for j=1:MAX_CONVERGENCE_TIME
            avg_tpt_evolution_per_wlan_exp3{iter}(j) = mean(tpt_evolution_per_wlan_exp3{iter}(j,:));
            fairness_evolution{iter}(j) = JainsFairness(tpt_evolution_per_wlan_exp3{iter}(j,:));
        end

        avg_tpt_experienced_exp3{iter}(g, e) = mean(avg_tpt_evolution_per_wlan_exp3{iter});            
        
        aggregate_tpt_transitory{iter}(g, e) = mean(sum(tpt_evolution_per_wlan_exp3{iter}(1:MIN_SAMPLE_CONSIDER, :), 2));
        aggregate_tpt_permanent{iter}(g, e) = mean(sum(tpt_evolution_per_wlan_exp3{iter}(MIN_SAMPLE_CONSIDER+1:MAX_CONVERGENCE_TIME, :), 2));
                    
        end
        
    end
    
end

%% PLOT THE RESULTS

% Compute the average results found and the standard deviation
aux_array_transitory = zeros(size(gamma, 2), size(eta, 2));
aux_array_permanent = zeros(size(gamma, 2), size(eta, 2));
for g = 1 : size(gamma, 2)     
    for e = 1 : size(eta, 2)
        for i = 1 : (TOTAL_ROUNDS/2)
            aux_array_transitory(g, e) = aux_array_transitory(g, e) + aggregate_tpt_transitory{i}(g, e);
            aux_array_permanent(g, e) = aux_array_permanent(g, e) + aggregate_tpt_permanent{i}(g, e);
        end
    end
end
mean_aggregate_tpt_transitory = aux_array_transitory / (TOTAL_ROUNDS/2);
mean_aggregate_tpt_permanent = aux_array_permanent / (TOTAL_ROUNDS/2);

disp('mean_aggregate_tpt_transitory')
disp(mean_aggregate_tpt_transitory)
disp('mean_aggregate_tpt_permanent')
disp(mean_aggregate_tpt_permanent)

aux_array_std_transitory = zeros(size(gamma, 2), size(eta, 2));
aux_array_std_permanent = zeros(size(gamma, 2), size(eta, 2));
for g = 1:size(gamma, 2)     
    for e = 1:size(eta, 2)
        for i = 1 : (TOTAL_ROUNDS/2)
            aux_array_std_transitory(g, e) = aux_array_std_transitory(g, e) + ((aggregate_tpt_transitory{i}(g, e) - mean_aggregate_tpt_transitory(g, e))^2);
            aux_array_std_permanent(g, e) = aux_array_std_permanent(g, e) + ((aggregate_tpt_permanent{i}(g, e) - mean_aggregate_tpt_permanent(g, e))^2);
        end
    end
end
std_aggregate_tpt_transitory = sqrt(aux_array_std_transitory / (TOTAL_ROUNDS/2));
std_aggregate_tpt_permanent = sqrt(aux_array_std_permanent / (TOTAL_ROUNDS/2));

disp('std_aggregate_tpt_transitory')
disp(std_aggregate_tpt_transitory)
disp('std_aggregate_tpt_permanent')
disp(std_aggregate_tpt_permanent)



delete(gcp('nocreate'))