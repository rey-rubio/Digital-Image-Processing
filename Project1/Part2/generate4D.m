%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function name:   
%   generate4D(M, N, minVal, maxVal)
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
%   h:          random 4D array
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}
function h = generate4D(M,N,minVal,maxVal)
    h = zeros(M,N,M,N);
    % generate random values for h[u][v][m][n]
    for u = 1:M         
        for v = 1:N
            for m = 1:M
                for n = 1:N
                    h(u,v,m,n)= minVal + rand()*(maxVal-minVal);
                end
            end
        end
    end
end

