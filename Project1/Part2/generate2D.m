%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function name:   
%   generate2D(M, N, minVal, maxVal)
%
% Description:
%   Generates a 2D array with values from minVal to maxVal
%
% Parameters:
%   M:          number of rows
%   N:          number of columns
%   minVal:     minimum value in array
%   maxVal:     maximum value in array
%
% Output:
%   h:          random 2D array
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}
function f = generate2D(M,N,minVal,maxVal)
    f = zeros(M,N);
    for m = 1:M         
        for n = 1:N
           f(m,n) = minVal + rand()*(maxVal-minVal);
        end
    end
end
