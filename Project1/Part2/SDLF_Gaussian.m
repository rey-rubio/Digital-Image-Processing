%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function name:   
%   SDLF_Gaussian(f,gfp, gfq, sigma)
%
% Description:
%   Computes the blurred image of image f with 1D separable gaussian 
%   filters gfp and gfq. Implemented the gaussian PDF as a separable filter
% 
% Parameters:
%   f:      f[m][n] input image
%   sigma:	sigma  value in range -2*sigma to 2*sigma
% Output:
%   g:      g[m][n] Spatial Domain Filter w/ Gaussian PSF
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}
function g = SDLF_Gaussian(f, sigma)
    [M, N] = size(f);       % M: Num. of Rows
                            % N: Num. of Columns 
                            
    % generate gaussian filters with given sigma
    gfp = gaussian_filter(sigma);
    gfq = gaussian_filter(sigma);   
    
    % total gaussian values                        
    total = sigma*2 + sigma*2 + 1;
                            
    % calculate g1[m][v]
    for m = 1:M
        for v = 1:N
            sum = 0.0;
            for n = 1:N
                
                % check bounds
                index = v - n;
                if(index >= 1 && index <= total)
                    sum = sum + gfq(index)*f(m,n);
                end
            end
            g1(m,v) = sum;
        end
    end                   
    
    % calculate g[u][v]
    for u = 1:M
        for v = 1:N
            sum = 0.0;
            for m = 1:M
                index = u - m;
                if(index >= 1 && index <= total)
                    sum = sum + gfp(index)*g1(m,v);
                end
            end
            g(u,v) = sum;
        end
    end
end



%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function name:   
%   gaussian_filter(sigma)
%
% Description:
%   Calculates SEPARABLE 1D guassian filter for given sigma with filter
%   coefficient normalization
%
% Parameters:
%   sigma:  sigma value for gaussian
%
% Output:
%   h:      calculated gaussian array
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}
function h = gaussian_filter(sigma)
    
    % calculate filter coefficient
    filterCoefficient = 0.0;
    for i = -sigma*2:sigma*2
        value = (1 / (sqrt(2 * pi * sigma ))) * (exp(-(i^2) / (2*sigma^2)));
        filterCoefficient = filterCoefficient + value;
        h(i + sigma*2 + 1) = value;
    end
    
    % normalize
    for i = -sigma*2:sigma*2
        h(i + sigma*2 + 1) = h(i + sigma*2 + 1) / filterCoefficient;
    end
end
