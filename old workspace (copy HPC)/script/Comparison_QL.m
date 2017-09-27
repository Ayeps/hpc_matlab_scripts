% Multi-Agent Reinforcement Learning Methods to Minimize OBSS Interferences
% Authors - Francesc Wilhelmi, Boris Bellalta, Anders Jonsson, Cristina Cano

% EXPERIMENT EXPLANATION:
% By using a simple grid of 4 WLANs sharing 2 channels, we want to find the
% optimal configurations that mazimizes the aggregated throughput and the
% proportional fairness, respectively

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

nWorkers = 8;
parpool('local', nWorkers)

disp('-----------------------')
disp('OPTIMAL CONFIGURATION CALCULATION')
disp('-----------------------')

% GLOBAL VARIABLES
plot_results = 0;               % To plot or not the results at the end of the simulation

% WLAN object to be modified for each number of coexistent nodes
global wlan

MAX_CONVERGENCE_TIME = 10000;    % Maximum convergence time (one period implies the participation of all WLANs)
MIN_SAMPLE_CONSIDER = MAX_CONVERGENCE_TIME/2 + 1;
MAX_LEARNING_ITERATIONS = 1;    % Maximum number of learning iterations done by each WLAN inside a general iteration
TOTAL_ROUNDS = 100;

update_mode = 1;

nWLANs = [2 4 6 8];

tpt_evolution_per_wlan_static = cell(1, TOTAL_ROUNDS);
tpt_evolution_per_wlan_random = cell(1, TOTAL_ROUNDS);
tpt_evolution_per_wlan_ql = cell(1, TOTAL_ROUNDS);

fairness_evolution_static_transitory = cell(1, TOTAL_ROUNDS);
fairness_evolution_static_permanent = cell(1, TOTAL_ROUNDS);
aggregate_tpt_static_transitory = cell(1, TOTAL_ROUNDS);
aggregate_tpt_static_permanent = cell(1, TOTAL_ROUNDS);

fairness_evolution_random_transitory = cell(1, TOTAL_ROUNDS);
fairness_evolution_random_permanent = cell(1, TOTAL_ROUNDS);
aggregate_tpt_random_transitory = cell(1, TOTAL_ROUNDS);
aggregate_tpt_random_permanent = cell(1, TOTAL_ROUNDS);

fairness_evolution_ql_transitory = cell(1, TOTAL_ROUNDS);
fairness_evolution_ql_permanent = cell(1, TOTAL_ROUNDS);
aggregate_tpt_ql_transitory = cell(1, TOTAL_ROUNDS);
aggregate_tpt_ql_permanent = cell(1, TOTAL_ROUNDS);

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
        tpt_evolution_per_wlan_static{iter} = ...
            RandomMethod(wlan, MAX_CONVERGENCE_TIME, MAX_LEARNING_ITERATIONS, ...
                actions_ch, actions_cca, actions_tpc, noise);
            
        fairness_evolution_static_transitory{iter}(s, :) = JainsFairness(tpt_evolution_per_wlan_static{iter}(1:MIN_SAMPLE_CONSIDER, :));
        fairness_evolution_static_permanent{iter}(s, :) = JainsFairness(tpt_evolution_per_wlan_static{iter}(MIN_SAMPLE_CONSIDER+1:MAX_CONVERGENCE_TIME, :));   
        aggregate_tpt_static_transitory{iter}(s,:) = sum(tpt_evolution_per_wlan_static{iter}(1:MIN_SAMPLE_CONSIDER, :), 2);
        aggregate_tpt_static_permanent{iter}(s,:) = sum(tpt_evolution_per_wlan_static{iter}(MIN_SAMPLE_CONSIDER+1:MAX_CONVERGENCE_TIME, :), 2);
        
        % RANDOM APPROACH
        tpt_evolution_per_wlan_random{iter} = ...
            RandomMethod(wlan, MAX_CONVERGENCE_TIME, MAX_LEARNING_ITERATIONS, ...
                actions_ch, actions_cca, actions_tpc, noise);

        fairness_evolution_random_transitory{iter}(s, :) = JainsFairness(tpt_evolution_per_wlan_random{iter}(1:MIN_SAMPLE_CONSIDER, :));
        fairness_evolution_random_permanent{iter}(s, :) = JainsFairness(tpt_evolution_per_wlan_random{iter}(MIN_SAMPLE_CONSIDER+1:MAX_CONVERGENCE_TIME, :));   
        aggregate_tpt_random_transitory{iter}(s,:) = sum(tpt_evolution_per_wlan_random{iter}(1:MIN_SAMPLE_CONSIDER, :), 2);
        aggregate_tpt_random_permanent{iter}(s,:) = sum(tpt_evolution_per_wlan_random{iter}(MIN_SAMPLE_CONSIDER+1:MAX_CONVERGENCE_TIME, :), 2);
                       
        % Q-LEARNING APPROACH
        gamma_ql = 0.95;
        initial_epsilon = 1;            % Initial Exploration coefficient
        final_epsilon = .01;            % Minimum value that epsilon can achieve
        alpha_ql = 1;
        [tpt_evolution_per_wlan_ql{iter}, ~] = ...
            QlearningMethod(wlan, MAX_CONVERGENCE_TIME, MAX_LEARNING_ITERATIONS, ...
                gamma_ql, initial_epsilon, alpha_ql, update_mode, ...
                actions_ch, actions_cca, actions_tpc, noise);
                   
        fairness_evolution_ql_transitory{iter}(s, :) = JainsFairness(tpt_evolution_per_wlan_ql{iter}(1:MIN_SAMPLE_CONSIDER, :));
        fairness_evolution_ql_permanent{iter}(s, :) = JainsFairness(tpt_evolution_per_wlan_ql{iter}(MIN_SAMPLE_CONSIDER+1:MAX_CONVERGENCE_TIME, :));   
        aggregate_tpt_ql_transitory{iter}(s,:) = sum(tpt_evolution_per_wlan_ql{iter}(1:MIN_SAMPLE_CONSIDER, :), 2);
        aggregate_tpt_ql_permanent{iter}(s,:) = sum(tpt_evolution_per_wlan_ql{iter}(MIN_SAMPLE_CONSIDER+1:MAX_CONVERGENCE_TIME, :), 2);
      
    end
    
end

for s = 1:size(nWLANs,2)
    
    agg_tpt_cat_static = [];
    fairness_cat_static = [];
    
    agg_tpt_cat_random = [];
    fairness_cat_random = [];
    
    agg_tpt_cat_ql = [];
    fairness_cat_ql = [];

    for i = 1:TOTAL_ROUNDS
        
        agg_tpt_cat_static = [agg_tpt_cat_static; aggregate_tpt_static_permanent{i}(s,:)];
        fairness_cat_static = [fairness_cat_static; fairness_evolution_static_permanent{i}(s,:)];
        
        agg_tpt_cat_random = [agg_tpt_cat_random; aggregate_tpt_random_permanent{i}(s,:)];
        fairness_cat_random = [fairness_cat_random; fairness_evolution_random_permanent{i}(s,:)];
        
        agg_tpt_cat_ql = [agg_tpt_cat_ql; aggregate_tpt_ql_permanent{i}(s,:)];
        fairness_cat_ql = [fairness_cat_ql; fairness_evolution_ql_permanent{i}(s,:)];

    end
    
    mean_agg_tpt_static(s) = mean(mean(agg_tpt_cat_static, 2));
    std_agg_tpt_static(s) = std(mean(agg_tpt_cat_static, 2));
    mean_fairness_static(s) = mean(mean(fairness_cat_static, 2));
    std_fairness_static(s) = std(mean(fairness_cat_static, 2));
    
    mean_agg_tpt_random(s) = mean(mean(agg_tpt_cat_random, 2));
    std_agg_tpt_random(s) = std(mean(agg_tpt_cat_random, 2));
    mean_fairness_random(s) = mean(mean(fairness_cat_random, 2));
    std_fairness_random(s) = std(mean(fairness_cat_random, 2));
    
    mean_agg_tpt_ql(s) = mean(mean(agg_tpt_cat_ql, 2));
    std_agg_tpt_ql(s) = std(mean(agg_tpt_cat_ql, 2));
    mean_fairness_ql(s) = mean(mean(fairness_cat_ql, 2));
    std_fairness_ql(s) = std(mean(fairness_cat_ql, 2));
        
end

plot_agg_tpt = [];
plot_std_agg_tpt = [];
plot_fairness = [];
plot_std_fairness = [];

for s = 1:size(nWLANs,2)
    
    plot_agg_tpt = [plot_agg_tpt; [mean_agg_tpt_static(s) mean_agg_tpt_random(s) mean_agg_tpt_ql(s)]];
    plot_std_agg_tpt = [plot_std_agg_tpt; [std_agg_tpt_static(s) std_agg_tpt_random(s) std_agg_tpt_ql(s)]];
    plot_fairness = [plot_fairness; [mean_fairness_static(s) mean_fairness_random(s) mean_fairness_ql(s)]];
    plot_std_fairness = [plot_std_fairness; [std_fairness_static(s) std_fairness_random(s) std_fairness_ql(s)]];
    
end


disp('plot_agg_tpt')
disp(plot_agg_tpt)
disp('plot_std_agg_tpt')
disp(plot_std_agg_tpt)
disp('plot_fairness')
disp(plot_fairness)
disp('plot_std_fairness')
disp(plot_std_fairness)

% bar(plot_agg_tpt)
% hold on
% errorbar(1:size(plot_agg_tpt,2), plot_agg_tpt, plot_std_agg_tpt, '.')
% legend({'Random', 'Q-learning', 'MAB (\epsilon-greedy)', 'MAB (EXP3)', 'MAB (UCB)'})
% xticks(1:4)
% xticklabels({'nWLANs = 2', 'nWLANs = 4', 'nWLANs = 6', 'nWLANs = 8'})


%% PLOT AGGREGATED THROUGHPUT PER APPROACH AND SIZE OF OBSS

% figure

% figure('pos', [450 400 500 350])
% axes;
% axis([1 20 30 70]);
% h = bar(plot_agg_tpt);
% set(h,'BarWidth',1);    % The bars will now touch each other
% set(gca,'YGrid','on')
% set(gca,'GridLineStyle','-')
% hold on;
% numgroups = size(plot_agg_tpt, 1); 
% numbars = size(plot_agg_tpt, 2); 
% groupwidth = min(0.8, numbars/(numbars+1.5));
% for i = 1:numbars
%       % Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
%       x = (1:numgroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*numbars);  % Aligning error bar with individual bar
%       errorbar(x, plot_agg_tpt(:,i), plot_std_agg_tpt(:,i), 'k', 'linestyle', 'none');
% end
% legend({'Static', 'Random', 'Q-learning'}, 'fontsize', 20)
% xticks(1:4)
% xticklabels({'N = 2', 'N = 4', 'N = 6', 'N = 8'})
% xt = get(gca, 'XTick');
% set(gca, 'FontSize', 22)
% ylabel('Network Throughput (Mbps)', 'fontsize', 24)
% 
%     
% %% PLOT FAIRNESS PER APPROACH AND SIZE OF OBSS
% 
% figure('pos', [450 400 500 350])
% axes;
% axis([1 20 30 70]);
% h = bar(plot_fairness);
% set(h,'BarWidth',1);    % The bars will now touch each other
% set(gca,'YGrid','on')
% set(gca,'GridLineStyle','-')
% hold on;
% numgroups = size(plot_fairness, 1); 
% numbars = size(plot_fairness, 2); 
% groupwidth = min(0.8, numbars/(numbars+1.5));
% for i = 1:numbars
%       % Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
%       x = (1:numgroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*numbars);  % Aligning error bar with individual bar
%       errorbar(x, plot_fairness(:,i), plot_std_fairness(:,i), 'k', 'linestyle', 'none');
% end
% xticks(1:4)
% xticklabels({'N = 2', 'N = 4', 'N = 6', 'N = 8'})
% xt = get(gca, 'XTick');
% set(gca, 'FontSize', 22)
% ylabel('Fairness', 'fontsize', 24)
% 
% % mean_agg_tpt = [mean_agg_tpt_random; mean_agg_tpt_ql; mean_agg_tpt_eg; mean_agg_tpt_exp3; mean_agg_tpt_ucb];
% % mean_fairness = [mean_fairness_random; mean_fairness_ql; mean_fairness_eg; mean_fairness_exp3; mean_fairness_ucb];
% % std_agg_tpt = [std_agg_tpt_random; std_agg_tpt_ql; std_agg_tpt_eg; std_agg_tpt_exp3; std_agg_tpt_ucb];
% % 
% % algs = {'Random', 'QL', '\epsilon-g', 'EXP3', 'UCB'};
% % ix = 1;
% % for s = 1:size(nWLANs, 2)
% %     mean_agg_tpt_aux{s} = [mean_agg_tpt(:,s) zeros(5,1)];
% %     mean_fairness_aux{s} = [zeros(5,1) mean_fairness(:,s)];
% %     subplot(2,2,ix)
% %     [AX,H1,H2] = plotyy([1:5], mean_agg_tpt_aux{s}, [1:5], mean_fairness_aux{s}, 'bar', 'bar');
% %     hold on
% %     yyaxis left
% %     ylabel('Aggregated Throughput (Mbps)')
% %     yyaxis right
% %     ylabel('Jain''s Fairness Index')
% %     xticks(1:5)
% %     xticklabels(algs)
% %     title(['Number of WLANs = ' num2str(nWLANs(s))])
% %     ix = ix + 1;
% % end

save('final_comparison_ql.mat', '-v7.3')

delete(gcp('nocreate'))