
%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function name:   
%   median_filter(f, S)
%
% Description:
%   Computes edge-preserving noise smoothing using SxS Median Filter
%
% Parameters:
%   f:           image to be filtered
%   S:           filter size
%
% Output:
%   g:           g[m][n] SxS Median filtered image
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}
function g = median_filter(f, S)
    [M, N] = size(f);     % M: Num. of Rows
                          % N: Num. of Columns 
    P = floor(S / 2); 
    Q = floor(S / 2);                        
    total = (2*P+1)*(2*Q+1);
    med = zeros(1, total);
    for m = 1:M         % each row of input 
        for n = 1:N     % each column of input 
            r = 1;      % index for med[]
            for p = -P:P
                pval = (m - p);
                if(pval < 0)
                    k = abs(pval);
                elseif(pval > M-1)
                    k = M-1-(pval-(M-1));
                elseif(pval == 0)
                    k = 1;
                else
                    k = pval;
                end
                for q = -Q:Q
                    qval = (n - q);
                    if(qval < 0)
                        l = abs(qval);
                    elseif(qval > N-1)
                        l = N-1-(qval-(N-1));
                    elseif(qval == 0)
                        l = 1;
                    else
                        l = qval;
                    end    
                  
                    point = f(k,l);
                    med(r) = point;
                    r = r + 1;
                end 
            end
            g(m,n) = median(med);
        end
    end         
end