%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function name:   
%   k_nearest_neighbors(f, S, K)
%
% Description:
%   Computes edge-preserving noise smoothing using SxS K-Nearest Neighbor
%
% Parameters:
%   f:       image to be filtered
%   S:       filter size (SxS)
%   K:       value for number of neighbors
%
% Output:
%   g:    g[m][n] an SxS k nearest neighbor filtered image	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}
function g = k_nearest_neighbors(f, S, K)
    [M, N] = size(f);     % M: Num. of Rows , 
                            % N: Num. of Columns 
    P = floor(S / 2); 
    Q = floor(S / 2);                        
    total = (2*P+1)*(2*Q+1);
    centerindex = 2*P+1;
    neighbors = zeros(1, total);
    distances = zeros(1, total);
    for m = 1:M     % each row of input 
        for n = 1:N % each column of input 
            r = 1;      % index for neighbors[]
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
                    
                    center = f(m,n);
                    point = f(k,l);
                    neighbors(r) = point;
                    distances(r) = abs(point - center);
                    r = r + 1;
                end 
            end
            
            % don't include center index when finding K smallest distances
            [~, kindex] = mink(distances,K+1);
            
            mean = 0;
            for i = 1:K+1
               meanindex = kindex(i);
                % don't include center index
               if(meanindex ~= centerindex)
                   mean = mean + neighbors(meanindex);
               end
            end
            mean = mean / K;
            g(m,n) = mean;
        end
    end         
end