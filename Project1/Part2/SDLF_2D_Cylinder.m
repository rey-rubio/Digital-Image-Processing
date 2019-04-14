%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function name:   
%   SDLF_2D_Cylinder(f, r)
%
% Description:
%   Computes the blurred image of image f with a 2D cylindrical function
%   with radius r. Implemented the cylindrical PSF as a direct convolution
%   in the spatial domain.
%
% Parameters:
%   f:       f[m][n] input image
%   r:       radius of filter
%
% Output:
%   g:       g[m][n] Spatial Domain Filter w/ 2d Cylindrical PSF
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}
function g = SDLF_2D_Cylinder(f, r)
    [M, N] = size(f);     % M: Num. of Rows
                          % N: Num. of Columns 
    P = r;
    Q = r;        
    g = zeros(M,N);
    for m = 1:M         % each row of input 
        for n = 1:N     % each column of input 
            sum = 0.0;
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
                    
                    sum = sum + cylindrical_filter(p,q,r) * f(k,l);
                end 
            end
            g(m,n) = sum;
        end
    end         
end


%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function name:   
%   cylindrical_filter(p, q, r)
%
% Description:
%   Calculates cylindrical  filter for given (p,q)  and radius.
%
% Parameters:
%   p:      p coordinate to output cylindrical filter
%   q:      q coordinate to output cylindrical filter
%   r:      radius of cylindrical filter
%
% Output:
%   h:      output cylindrical filter value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}
function h = cylindrical_filter(p, q, r)
    if(p^2 + q^2 < r^2)
        h = 1 / (pi * r^2);
    else
        h = 0;
    end
end
