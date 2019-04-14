%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function name:   
%   convolution_linear_transform(f, h)
%
% Description:
%   Computes the convolution l (linear shift-invariant) transform:
%
% Parameters:
%   f:       f[m][n] input image
%   h:       hc[m][n] input filter
%
% Output:
%   g:       g[u][v] Convolution (linear shift-invariant) transform: CLT
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}
function g = convolution_linear_transform(f, h)
    [M, N] = size(f);       % M: Num. of Rows
                            % N: Num. of Columns 
     for u = 1:M         
        for v = 1:N
            sum = 0.0;
            for m = 1:M
                for n = 1:N
                    row = mod((u-m+M),M) + 1;
                    col = mod((v-n+N),N) + 1;
                    sum = sum + h(row, col) * f(m,n);
                end
            end
            g(u,v) = sum;
        end
    end
end