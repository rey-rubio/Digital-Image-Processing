%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function name:   
%   Freq_DFT(f)
%
% Description:
%   Computes the Discrete Fourier Transform of an input image, with a
%   frequency filter
%   F:      DFT
%   H:      frequency filter
% 
% Parameters:
%   f:      f[m][n] input image
%
% Output:
%   G:      G[u][v] Frequency-Filtered Discrete Fourier Transform
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}
function G = Freq_DFT(f)
    [M, N] = size(f);       % M: Num. of Rows
                            % N: Num. of Columns 
    
    % calculate F1[m][v]
    for m = 1:M
        for v = 1:N
            sum = 0.0;
            for n = 1:N
                theta = -2*pi*(v*n/N);
                sum = sum + f(m,n) * exp(1j * theta);
%               sum = sum + f(m,n) * (cos(theta) + 1j*sin(theta));
            end
            F1(m,v) = sum * (1/N);
        end
    end
    
    % calculate F[u][v]
    for u = 1:M
        for v = 1:N
            sum = 0.0;
            for m = 1:M
                theta = -2*pi*(u*m/M);
                sum = sum + F1(m,v) * exp(1j * theta);
%               sum = sum + F1(m,v) * (cos(theta) + 1j*sin(theta));
            end
            F(u,v) = sum * (1/M);
        end
    end
    
    % calculate G[u][v]
    H = generateFrequencyFilter(M,N);
    for u = 1:M
        for v = 1:N
            G(u,v) = F(u,v) * H(u,v);
        end
    end
end


