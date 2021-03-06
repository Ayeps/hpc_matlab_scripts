% Multi-Agent Reinforcement Learning Methods to Minimize OBSS Interferences
% Authors - Francesc Wilhelmi, Boris Bellalta, Anders Jonsson, Cristina Cano
function [tpt_experienced_by_WLAN] = StaticMethod(wlan, noise)
% QLEARNINGMETHOD Given an OBSS, computes the experienced throughput if using a static approach
%
%   OUTPUT: 
%       * tpt_experienced_by_WLAN - throughput experienced by each WLAN
%         for each of the iterations done
%   INPUT: 
%       * wlan - wlan object containing information about all the WLANs
%       * noise - floor noise in dBM

    % Use a copy of wlan to make operations
    wlan_aux = wlan;
    n_WLANs = size(wlan,2);
    % Compute the maximum achievable throughput per WLAN
    power_matrix = PowerMatrix(wlan_aux);    
    tpt_experienced_by_WLAN = computeThroughputFromSINR(wlan_aux, power_matrix, noise);  % bps 
    
end

