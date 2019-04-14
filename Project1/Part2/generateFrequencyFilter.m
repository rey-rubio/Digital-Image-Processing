%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function name:   
%   generateFrequencyFilter(M, N)
%
% Description:
%   Generates a 2D frequency filter of size MxN. Only filters 5x5 corners.
%
% Parameters:
%   M:          number of rows
%   N:          number of columns
%
% Output:
%   H:          2D frequency filter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}
function H = generateFrequencyFilter(M,N)
    H = ones(M,N);
    for u = 1:M         
        for v = 1:N
            if( (u <= 5 && v <= 5) || (u > (M - 5) && v <= 5) || (u <= 5 && v > (N - 5)) || (u > (M - 5) && v > (N - 5)))
                H(u,v) = 0.5;
            end
        end
    end
end