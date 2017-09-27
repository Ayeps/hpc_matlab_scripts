% Multi-Agent Reinforcement Learning Methods to Minimize OBSS Interferences
% Authors - Francesc Wilhelmi, Boris Bellalta, Anders Jonsson, Cristina Cano
function C = computeTheoreticalCapacity(B, sinr)
% Computes the theoretical capacity given a bandwidth and a SINR
%
% OUTPUT:
%   * C - capacity in bps
% INPUT:
%   * B - Available Bandwidth (Hz) 
%   * sinr - Signal to Interference plus Noise Ratio (-)

    C = B * log2(1+sinr);

end