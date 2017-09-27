% Multi-Agent Reinforcement Learning Methods to Minimize OBSS Interferences
% Authors - Francesc Wilhelmi, Boris Bellalta, Anders Jonsson, Cristina Cano
function arm = selectActionUCB(rewards_per_configuration)
% selectActionUCB: returns the best possible arm given the current distribution
%   OUTPUT:
%       * arm - chosen arm to be played - configuration composed by [channel,CCA,TPC]
%   INPUT:
%       - rewards_per_configuration: rewards noticed at each configuration
    
    [val,~] = max(rewards_per_configuration);
    
    % Break ties randomlyselected_armtime_arm_is_played
    
    if sum(rewards_per_configuration==val) > 1
        if val ~= Inf
            indexes = find(rewards_per_configuration == val);
            arm = randsample(indexes,1);
        else
            arm = randsample(1:size(rewards_per_configuration, 2), 1);
        end
        
    % Select arm with maximum reward
    else
        [~, arm] = max(rewards_per_configuration);
    end
    
end