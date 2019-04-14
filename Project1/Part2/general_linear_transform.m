%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function name:   
%   general_linear_transform(f, h)
%
% Description:
%   Computes General (shift-variant) linear transform (GLT):
%
% Parameters:
%   f:       input image f[m][n]
%   h:       input filter h[u][v][m][n]
%
% Output:
%   GLT:     g[u][v] general (shift-variant) linear transform (GLT) :
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}
function g = general_linear_transform(f, h)
    [M, N] = size(f);       % M: Num. of Rows
                            % N: Num. of Columns 
    for u = 1:M         
        for v = 1:N
            sum = 0.0;
            for m = 1:M
                for n = 1:N
                    sum = sum + h(u,v,m,n) * f(m,n);
                end
            end
            g(u,v) = sum;
        end
    end         
end