%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function name:   
%   Freq_IDFT(G)
%
% Description:
%   Computes the inverse Discrete Fourier Transform of a Frequency-Filtered
%   Discrete Fourier Transform
%   F:      Inverse DFT
%   H:      frequency filter
% 
% Parameters:
%   G:      G[u][v] Frequency-Filtered Discrete Fourier Transform image
% Output:
%   f:      f[m][n] input image
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}
function f = Freq_IDFT(G)
    [M, N] = size(G);       % M: Num. of Rows
                            % N: Num. of Columns 
    
    % calculate F[u][v]
    H = generateFrequencyFilter(M,N);
    for u = 1:M
        for v = 1:N
            F(u,v) = G(u,v) / H(u,v);
        end
    end      
    
    
    % calculate f1[u][n]
    for u = 1:M
        for n = 1:N
            sum = 0.0;
            for v = 1:N
                theta = 2*pi*(v*n/N);
                sum = sum + F(u,v) * exp(1j * theta);
            end
            f1(u,n) = sum;
        end
    end                        
                            
    
    % calculate f[m][n]
    for m = 1:M
        for n = 1:N
            sum = 0.0;
            for u = 1:M
                theta = 2*pi*(u*m/M);
                sum = sum + f1(u,n) * exp(1j * theta);
            end
            f(m,n) = sum;
        end
    end
    
end


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