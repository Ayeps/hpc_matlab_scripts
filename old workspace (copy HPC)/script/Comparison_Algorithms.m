% Multi-Agent Reinforcement Learning Methods to Minimize OBSS Interferences
% Authors - Francesc Wilhelmi, Boris Bellalta, Anders Jonsson, Cristina Cano

clc
clear all

addpath(genpath('Power Management Methods/'));
addpath(genpath('Throughput Calculation Methods/'));
addpath(genpath('Network Generation Methods/'));
addpath(genpath('Reinforcement Learning Methods/'));
addpath(genpath('Reinforcement Learning Methods/Action Selection Methods/'));
addpath(genpath('Auxiliary Methods/'));

nWorkers = 8;
parpool('local', nWorkers)

disp('-----------------------')
disp('OPTIMAL CONFIGURATION CALCULATION')
disp('-----------------------')

% GLOBAL VARIABLES
plot_results = 0;               % To plot or not the results at the end of the simulation

MAX_CONVERGENCE_TIME = 10000;    % Maximum convergence time (one period implies the participation of all WLANs)
MIN_SAMPLE_CONSIDER = MAX_CONVERGENCE_TIME/2 + 1;
MAX_LEARNING_ITERATIONS = 1;    % Maximum number of learning iterations done by each WLAN inside a general iteration
TOTAL_ROUNDS = 100;

update_mode = 1;
printInfo = 0;

nWLANs = [2 4 6 8];

tpt_evolution_per_wlan_random = cell(1, TOTAL_ROUNDS);
tpt_evolution_per_wlan_eg = cell(1, TOTAL_ROUNDS);
tpt_evolution_per_wlan_exp3 = cell(1, TOTAL_ROUNDS);
tpt_evolution_per_wlan_ucb = cell(1, TOTAL_ROUNDS);

fairness_evolution_random_transitory = cell(1, TOTAL_ROUNDS);
fairness_evolution_random_permanent = cell(1, TOTAL_ROUNDS);
aggregate_tpt_random_transitory = cell(1, TOTAL_ROUNDS);
aggregate_tpt_random_permanent = cell(1, TOTAL_ROUNDS);

fairness_evolution_eg_transitory = cell(1, TOTAL_ROUNDS);
fairness_evolution_eg_permanent = cell(1, TOTAL_ROUNDS);
aggregate_tpt_eg_transitory = cell(1, TOTAL_ROUNDS);
aggregate_tpt_eg_permanent = cell(1, TOTAL_ROUNDS);

fairness_evolution_exp3_transitory = cell(1, TOTAL_ROUNDS);
fairness_evolution_exp3_permanent = cell(1, TOTAL_ROUNDS);
aggregate_tpt_exp3_transitory = cell(1, TOTAL_ROUNDS);
aggregate_tpt_exp3_permanent = cell(1, TOTAL_ROUNDS);

fairness_evolution_ucb_transitory = cell(1, TOTAL_ROUNDS);
fairness_evolution_ucb_permanent = cell(1, TOTAL_ROUNDS);
aggregate_tpt_ucb_transitory = cell(1, TOTAL_ROUNDS);
aggregate_tpt_ucb_permanent = cell(1, TOTAL_ROUNDS);

for s = 1:size(nWLANs, 2)    
    
    disp('------------------------------------')
    disp(['  Number of WLANs = ' num2str(nWLANs(s))])
    disp('------------------------------------')
        
    parfor (iter = 1:TOTAL_ROUNDS, nWorkers)
    
        disp('++++++++++++++++')
        disp(['ROUND ' num2str(iter) '/' num2str(TOTAL_ROUNDS)])
        disp('++++++++++++++++')

        nChannels = nWLANs(s)/2;              % Number of available channels (from 1 to NumChannels)
        noise = -100;               % Floor noise (dBm)

        % Definition of actions:
        actions_ch = 1:nChannels;       % nChannels possible channels
        actions_cca = [-82];            % One CCA level (dBm) -> meaningless (all interferences are considered)
        actions_tpc = [5 10 15 20];     % 4 different levels of TPC (dBm)

        % Each state represents an [i,j,k] combination for indexes on "channels", "CCA" and "TxPower"
        possible_actions = 1:(size(actions_ch,2)*size(actions_cca,2)*size(actions_tpc,2));
        % Total number of actions
        K = size(possible_actions,2);
        % All the possible combinations of configurations for the entire scenario
        possible_comb = allcomb(possible_actions,possible_actions,possible_actions,possible_actions);
        % Setup the scenario: generate WLANs and initialize states and actions
        wlan = GenerateRandomNetwork3D(nWLANs(s), nChannels, 0); % RANDOM CONFIGURATION
        %wlan = GenerateNetwork3D(n_WLANs, nChannels, 'grid', 2, 0); % GRID (SAFE) CONFIGURATION

    %     % Compute the optimal configuration to compare the approaches
    %     maximum_achievable_throughput = computeTptAllCombinations(wlan, actions_ch, actions_cca, actions_tpc, noise);

        % RANDOM APPROACH
        tpt_evolution_per_wlan_random{iter} = ...
            RandomMethod(wlan, MAX_CONVERGENCE_TIME, MAX_LEARNING_ITERATIONS, ...
                actions_ch, actions_cca, actions_tpc, noise);

        fairness_evolution_random_transitory{iter}(s, :) = JainsFairness(tpt_evolution_per_wlan_random{iter}(1:MIN_SAMPLE_CONSIDER, :));
        fairness_evolution_random_permanent{iter}(s, :) = JainsFairness(tpt_evolution_per_wlan_random{iter}(MIN_SAMPLE_CONSIDER+1:MAX_CONVERGENCE_TIME, :));   
        aggregate_tpt_random_transitory{iter}(s,:) = sum(tpt_evolution_per_wlan_random{iter}(1:MIN_SAMPLE_CONSIDER, :), 2);
        aggregate_tpt_random_permanent{iter}(s,:) = sum(tpt_evolution_per_wlan_random{iter}(MIN_SAMPLE_CONSIDER+1:MAX_CONVERGENCE_TIME, :), 2);
                       
        % E-GREEDY APPROACH
        initial_epsilon = .8;            % Initial Exploration coefficient
        tpt_evolution_per_wlan_eg{iter}  = ...
            eGreedyMethod(wlan, MAX_CONVERGENCE_TIME, MAX_LEARNING_ITERATIONS, ...
                initial_epsilon, update_mode, actions_ch, actions_cca, actions_tpc, noise, printInfo);

        fairness_evolution_eg_transitory{iter}(s, :) = JainsFairness(tpt_evolution_per_wlan_eg{iter}(1:MIN_SAMPLE_CONSIDER, :));
        fairness_evolution_eg_permanent{iter}(s, :) = JainsFairness(tpt_evolution_per_wlan_eg{iter}(MIN_SAMPLE_CONSIDER+1:MAX_CONVERGENCE_TIME, :));   
        aggregate_tpt_eg_transitory{iter}(s,:) = sum(tpt_evolution_per_wlan_eg{iter}(1:MIN_SAMPLE_CONSIDER, :), 2);
        aggregate_tpt_eg_permanent{iter}(s,:) = sum(tpt_evolution_per_wlan_eg{iter}(MIN_SAMPLE_CONSIDER+1:MAX_CONVERGENCE_TIME, :), 2);

        % EXP3 APPROACH
        gamma = 0;
        eta = .1;
        tpt_evolution_per_wlan_exp3{iter}  = ...
            EXP3MethodBasic(wlan, MAX_CONVERGENCE_TIME, MAX_LEARNING_ITERATIONS, ...
                gamma, eta, actions_ch, actions_cca, actions_tpc, noise, printInfo);
            
        fairness_evolution_exp3_transitory{iter}(s, :) = JainsFairness(tpt_evolution_per_wlan_exp3{iter}(1:MIN_SAMPLE_CONSIDER, :));
        fairness_evolution_exp3_permanent{iter}(s, :) = JainsFairness(tpt_evolution_per_wlan_exp3{iter}(MIN_SAMPLE_CONSIDER+1:MAX_CONVERGENCE_TIME, :));   
        aggregate_tpt_exp3_transitory{iter}(s,:) = sum(tpt_evolution_per_wlan_exp3{iter}(1:MIN_SAMPLE_CONSIDER, :), 2);
        aggregate_tpt_exp3_permanent{iter}(s,:) = sum(tpt_evolution_per_wlan_exp3{iter}(MIN_SAMPLE_CONSIDER+1:MAX_CONVERGENCE_TIME, :), 2);

        % UCB APPROACH
        tpt_evolution_per_wlan_ucb{iter}  = ...
            UCBMethod(wlan, MAX_CONVERGENCE_TIME, MAX_LEARNING_ITERATIONS, ...
                actions_ch, actions_cca, actions_tpc, noise, printInfo);
            
        fairness_evolution_ucb_transitory{iter}(s, :) = JainsFairness(tpt_evolution_per_wlan_ucb{iter}(1:MIN_SAMPLE_CONSIDER, :));
        fairness_evolution_ucb_permanent{iter}(s, :) = JainsFairness(tpt_evolution_per_wlan_ucb{iter}(MIN_SAMPLE_CONSIDER+1:MAX_CONVERGENCE_TIME, :));   
        aggregate_tpt_ucb_transitory{iter}(s,:) = sum(tpt_evolution_per_wlan_ucb{iter}(1:MIN_SAMPLE_CONSIDER, :), 2);
        aggregate_tpt_ucb_permanent{iter}(s,:) = sum(tpt_evolution_per_wlan_ucb{iter}(MIN_SAMPLE_CONSIDER+1:MAX_CONVERGENCE_TIME, :), 2);
        
    end
    
end

for s = 1:size(nWLANs,2)
    
    agg_tpt_cat_random = [];
    fairness_cat_random = [];
    
    agg_tpt_cat_eg = [];
    fairness_cat_eg = [];

    agg_tpt_cat_exp3 = [];
    fairness_cat_exp3 = [];

    agg_tpt_cat_ucb = [];
    fairness_cat_ucb = [];
    
    for i = 1:TOTAL_ROUNDS
        
        agg_tpt_cat_random = [agg_tpt_cat_random; aggregate_tpt_random_permanent{i}(s,:)];
        fairness_cat_random = [fairness_cat_random; fairness_evolution_random_permanent{i}(s,:)];
        
        agg_tpt_cat_eg = [agg_tpt_cat_eg; aggregate_tpt_eg_permanent{i}(s,:)];
        fairness_cat_eg = [fairness_cat_eg; fairness_evolution_eg_permanent{i}(s,:)];
        
        agg_tpt_cat_exp3 = [agg_tpt_cat_exp3; aggregate_tpt_exp3_permanent{i}(s,:)];
        fairness_cat_exp3 = [fairness_cat_exp3; fairness_evolution_exp3_permanent{i}(s,:)];
        
        agg_tpt_cat_ucb = [agg_tpt_cat_ucb; aggregate_tpt_ucb_permanent{i}(s,:)];
        fairness_cat_ucb = [fairness_cat_ucb; fairness_evolution_ucb_permanent{i}(s,:)];
        
    end
    
    mean_agg_tpt_random(s) = mean(mean(agg_tpt_cat_random, 2));
    std_agg_tpt_random(s) = std(mean(agg_tpt_cat_random, 2));
    mean_fairness_random(s) = mean(mean(fairness_cat_random, 2));
    std_fairness_random(s) = std(mean(fairness_cat_random, 2));

    mean_agg_tpt_eg(s) = mean(mean(agg_tpt_cat_eg, 2));
    std_agg_tpt_eg(s) = std(mean(agg_tpt_cat_eg, 2));
    mean_fairness_eg(s) = mean(mean(fairness_cat_eg, 2));
    std_fairness_eg(s) = std(mean(fairness_cat_eg, 2));
    
    mean_agg_tpt_exp3(s) = mean(mean(agg_tpt_cat_exp3, 2));
    std_agg_tpt_exp3(s) = std(mean(agg_tpt_cat_exp3, 2));
    mean_fairness_exp3(s) = mean(mean(fairness_cat_exp3, 2));
    std_fairness_exp3(s) = std(mean(fairness_cat_exp3, 2));
    
    mean_agg_tpt_ucb(s) = mean(mean(agg_tpt_cat_ucb, 2));
    std_agg_tpt_ucb(s) = std(mean(agg_tpt_cat_ucb, 2));
    mean_fairness_ucb(s) = mean(mean(fairness_cat_ucb, 2));
    std_fairness_ucb(s) = std(mean(fairness_cat_ucb, 2));    
    
%     mean_agg_tpt_random(s) = mean(mean(cat(TOTAL_ROUNDS, aggregate_tpt_random{:}(s,:))));
%     mean_fairness_random = mean(JainsFairness(cat(TOTAL_ROUNDS, aggregate_tpt_random{:})));
%     std_agg_tpt_random(s) = mean(std(cat(TOTAL_ROUNDS, aggregate_tpt_random{:}(s,:))));
end

plot_agg_tpt = [];
plot_std_agg_tpt = [];
plot_fairness = [];
plot_std_fairness = [];

for s = 1:size(nWLANs,2)
    
    plot_agg_tpt = [plot_agg_tpt; [mean_agg_tpt_random(s) mean_agg_tpt_eg(s) mean_agg_tpt_exp3(s) mean_agg_tpt_ucb(s)]];
    plot_std_agg_tpt = [plot_std_agg_tpt; [std_agg_tpt_random(s) std_agg_tpt_eg(s) std_agg_tpt_exp3(s) std_agg_tpt_ucb(s)]];
    plot_fairness = [plot_fairness; [mean_fairness_random(s) mean_fairness_eg(s) mean_fairness_exp3(s) mean_fairness_ucb(s)]];
    plot_std_fairness = [plot_std_fairness; [std_fairness_random(s) std_fairness_eg(s) std_fairness_exp3(s) std_fairness_ucb(s)]];
    
end

disp('plot_agg_tpt')
disp(plot_agg_tpt)
disp('plot_std_agg_tpt')
disp(plot_std_agg_tpt)
disp('plot_fairness')
disp(plot_fairness)
disp('plot_std_fairness')
disp(plot_std_fairness)

save('final_comparison_workspace.mat')