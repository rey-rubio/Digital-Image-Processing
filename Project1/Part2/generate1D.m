%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function name:   
%   generate1D(S, minVal, maxVal)
%
% Description:
%   Generates a 1D array with values from minVal to maxVal
%
% Parameters:
%   S:          array size
%   minVal:     minimum value in array
%   maxVal:     maximum value in array
%
% Output:
%   h:          random 1D array
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}
function h = generate1D(S,minVal,maxVal)
    h = zeros(S);
    for s = 1:S    
       h(s) = minVal + rand()*(maxVal-minVal);
    end
end
