%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function name:   
%   separable_convolution_transform(f, h1c, h2c)
%
% Description:
%   Computes the separable convolution linear transform (SCT):
%
% Parameters:
%   f:        f[m][n] input image
%   h1c:      h[m] input filter
%   h2c:      h[n] input filter
%
% Output:
%   g:       g[u][v] separable convolution linear transform (SCT)
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}
function g = separable_convolution_transform(f, h1c, h2c)
    [M, N] = size(f);       % M: Num. of Rows
                            % N: Num. of Columns 
                            
    % calculate g1[m][v]
    for m = 1:M
        for v = 1:N
            sum = 0.0;
            for n = 1:N
                index = mod((v-n+N),N) + 1;
                sum = sum + h2c(index)*f(m,n);
            end
            g1(m,v) = sum;
        end
    end                   
    
    % calculate g[u][v]
    for u = 1:M
        for v = 1:N
            sum = 0.0;
            for m = 1:M
                index = mod((u-m+M),M) + 1;
                sum = sum + h1c(index) * g1(m,v);
            end
            g(u,v) = sum;
        end
    end
end
