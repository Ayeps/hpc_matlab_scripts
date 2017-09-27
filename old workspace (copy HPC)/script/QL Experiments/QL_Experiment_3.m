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
MAX_CONVERGENCE_TIME = 100;    % Maximum convergence time (one period implies the participation of all WLANs)
MAX_LEARNING_ITERATIONS = 1;    % Maximum number of learning iterations done by each WLAN inside a general iteration
TOTAL_ROUNDS = 1;              % Number of TOTAL repetitions to take the average
plot_results = 0;               % To plot or not the results at the end of the simulation

% WLAN object to be modified for each number of coexistent nodes
global wlan
nChannels = 6;              % Number of available channels (from 1 to NumChannels)
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

alpha_epsilon_pairs = [.95 1; 0.5 1; .05 1; .95 .5; .5 .5; .05 .5];
gamma_ql = 0:.1:1;              % Discount factor Q-learning

%% ITERATE FOR NUMBER OF REPETITIONS (TO TAKE THE AVERAGE)
for iter = 1:TOTAL_ROUNDS

    disp('------------------------------------')
    disp(['ROUND ' num2str(iter) '/' num2str(TOTAL_ROUNDS)])
    disp('------------------------------------')

    % Setup the scenario: generate WLANs and initialize states and actions
    wlan = GenerateNetwork3D(n_WLANs, nChannels, 'grid', 3, 0);
    DrawNetwork3D(wlan)
    %power_matrix = PowerMatrix(wlan); 
    %upper_bound_tpt_per_wlan = computeMaxBoundThroughput(wlan, power_matrix, noise);
    
    %maximum_achievable_throughput = computeTptAllCombinations(wlan, actions_ch, actions_cca, actions_tpc, noise);

    for g = 1:size(gamma_ql,2)
        
            for a = 1:size(alpha_epsilon_pairs,1)
        
                    tpt_evolution_per_wlan_ql{iter}  = QlearningMethod(wlan, MAX_CONVERGENCE_TIME, MAX_LEARNING_ITERATIONS, ...
                                                    gamma_ql(g), alpha_epsilon_pairs(a,2), final_epsilon, alpha_epsilon_pairs(a,1), actions_ch, actions_cca, actions_tpc, noise);
                    for j=1:MAX_CONVERGENCE_TIME-1
                        avg_tpt_evolution_ql{iter}(j) = mean(tpt_evolution_per_wlan_ql{iter} (j,:));
                    end

                    avg_tpt_experienced_ql{iter}(g,a) = mean(avg_tpt_evolution_ql{iter});              
                    std_tpt_experienced_ql{iter}(g,a) = std(avg_tpt_evolution_ql{iter}); 
                                   
            end
        
    end
    
end

% figure
% surf(gamma_ql,alpha,mean_tpt_results_ql)
% xlabel('\gamma','FontSize',14)
% ylabel('\alpha','FontSize',14)
% zlabel('Network Throughput (Mbps)','FontSize',14)

% Maximum value and index:
[maxVal,ix] = max(avg_tpt_experienced_ql{1}(:));
[m,n] = ind2sub(size(avg_tpt_experienced_ql{1}),ix);
disp(['Best gamma = ' num2str(gamma_ql(m))]);
disp(['Best alpha/epsilon = ' num2str(alpha_epsilon_pairs(n,:))]);

r = {'-*','-+','-o','-d','-s','--'};
l = {};
figure
if TOTAL_ROUNDS > 1
    mean_tpt_results_ql = mean(cat(TOTAL_ROUNDS+1, avg_tpt_experienced_ql{:}), TOTAL_ROUNDS+1);
    mean_std_results_ql = mean(cat(TOTAL_ROUNDS+1, std_tpt_experienced_ql{:}), TOTAL_ROUNDS+1);
    for i=1:size(alpha_epsilon_pairs,1)
        plot(gamma_ql,mean_tpt_results_ql(:,i)',r{i})
        hold on
        l = [l ['\alpha = ' num2str(alpha_epsilon_pairs(i,1)) ' \epsilon_{0} = ' num2str(alpha_epsilon_pairs(i,2))]];
        %errorbar(mean_tpt_results_ql(:,i)', mean_std_results_ql(:,i)','r.')
        xticks(gamma_ql)
    end
elseif TOTAL_ROUNDS == 1
    for i=1:size(alpha_epsilon_pairs,1)
        plot(gamma_ql,avg_tpt_experienced_ql{1}(:,i)',r{i})
        hold on
        l = [l ['\alpha = ' num2str(alpha_epsilon_pairs(i,1)) ' \epsilon_{0} = ' num2str(alpha_epsilon_pairs(i,2))]];
        %errorbar(mean_tpt_results_ql(:,i)', mean_std_results_ql(:,i)','r.')
        xticks(gamma_ql)
    end
end
legend(l)
ylabel('Average Experienced Throughput (Mbps)')
xlabel('\gamma', 'FontSize', 15)