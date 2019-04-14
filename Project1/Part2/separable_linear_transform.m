%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function name:   
%   separable_linear_transform(f, h1, h2)
%
% Description:
%   Computes the separable linear transform (SLT):
%
% Parameters:
%   f:       f[m][n] input image
%   h1:      h[u][m] input filter
%   h2:      h[v][n] input filter
%
% Output:
%   g:       g[u][v] Separable linear transform (SLT)
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}
function g = separable_linear_transform(f, h1, h2)
    [M, N] = size(f);       % M: Num. of Rows
                            % N: Num. of Columns 
    
    % calculate g1[m][v]
    for m = 1:M
        for v = 1:N
            sum = 0.0;
            for n = 1:N
                sum = sum + h2(v,n)*f(m,n);
            end
            g1(m,v) = sum;
        end
    end
    
    % calculate g[u][v]
    for u = 1:M
        for v = 1:N
            sum = 0.0;
            for m = 1:M
                sum = sum + h1(u,m) * g1(m,v);
            end
            g(u,v) = sum;
        end
    end
end