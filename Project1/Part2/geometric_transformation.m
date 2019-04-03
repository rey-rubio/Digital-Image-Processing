%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reynerio Rubio
% ID# 109899097
% ESE 558 
% SPRING 2019
% 04/05/2019
% 
% GEOMETRIC TRANSFORMATION OF IMAGES
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Read an RGB image color image in images folder,  'images/food1.jpg'.

%IMG0 = imread('images/mona-lisa.png');
IMG0 = imread('images/food1.jpg');
%[M, N, C] = size(IMG0);     % M: Num. of Rows , 
                            % N : Num. of Columns , 
                            % C : Num. of color bands = 3
figure
imshow(IMG0);
title("IMG0: Original Image");

% figure
% I6 = double(I1)/255.0;
% imshow(I6);
% title('I6: Original Image w/ fp')

% figure
IMG1= rgb2gray(IMG0);
% imshow(IMG1);
% title("IMG1: Grayscale");
 
figure
IMG1fp = double(IMG1)/255.0;
imshow(IMG1fp);
title("IMG1fp: Grayscale w/ fp");
 
figure
S = 5;
IMG2a = MF(IMG1fp, S);
imshow(IMG2a);
title(sprintf("IMG2a: Median Filter, %dx%d", S, S));
% 
figure
S = 3;
K = 4;
IMG2b = KNN(IMG1fp, S, K);
imshow(IMG2b);
title(sprintf("IMG2b: %dx%d K-Nearest Neighbors,  K = %d", S, S, K));

%figure
min = 0.0; 
max = 100.0;
M = 8;
N = 10;
f = generate2D(M, N, min, max);
%imshow(f);
%title("f(m,n): INPUT image General Linear Transform");


%figure
huvmn = generate4D(M, N, min, max);
IMG3a = GLT(f,huvmn);
%imshow(IMG3a);
%title("IMG3a: General Linear Transform (GLT)");

%figure
h1um = generate2D(M, M, min, max);
h2vn = generate2D(N, N, min, max);
IMG3b = SLT(f,h1um, h2vn);
%imshow(IMG3b);
%title("IMG3b: Separable Linear Transform (SLT)");


%figure
hcmn = generate2D(M, N, min, max);
IMG3c = CLT(f,hcmn);
%imshow(IMG3c);
%title("IMG3c: Circular Linear Transform (CLT)");



% figure
h1cm = generate1D(M, min, max);
h2cn = generate1D(N, min, max);
IMG3d = SCT(f,h1cm,h2cm);
% imshow(IMG3d);
% title("IMG3d: Separable Convolution linear transform (SCT)");

figure
r = 5;
IMG4 = SDLF(IMG1fp,r);
imshow(IMG4)
title(sprintf("IMG4: Spatial domain linear filtering ,  r = %d", r));



%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function name:   
%   MF(f, S)
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
function g = MF(f, S)
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


%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function name:   
%   knn(f, S, K)
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
function g = KNN(f, S, K)
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


%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function name:   
%   GLT(f, h)
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
function g = GLT(f, h)
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


%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function name:   
%   SLT(f, h1, h2)
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
function g = SLT(f, h1, h2)
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


%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function name:   
%   CLT(f, h)
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
function g = CLT(f, h)
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


%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function name:   
%   SCT(f, h1c, h2c)
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
function g = SCT(f, h1c, h2c)
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


%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function name:   
%   SDLF(f, r)
%
% Description:
%   Computes the computing the blurred image of a planar object by adding 
%   different levels of Gaussian independent noise. 
%
% Parameters:
%   f:       f[m][n] input image
%   r:       radius of filter
%
% Output:
%   g:       g[m][n] Spatial domain linear filtered image
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}
function g = SDLF(f, r)
    [M, N] = size(f);     % M: Num. of Rows
                          % N: Num. of Columns 
    P = r;
    Q = r;                
    sigma = 1.0;
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
                    
                    sum = sum + gaussian_filter(p,q,sigma) * f(k,l);
                end 
            end
            g(m,n) = sum;
        end
    end         
end


%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function name:   
%   gaussian_filter(x, y, sigma)
%
% Description:
%   Calculates guassian interpolation filter for given (x,y)  and sigma.
%
% Parameters:
%   x:      x coordinate to calculate gaussian
%   y:      y coordinate to calculate gaussian
%   sigma:  sigma value for gaussian
%
% Output:
%   h = calculated gaussian value
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}
function h = gaussian_filter(x, y, sigma)
    h = (1 / (2 * pi * sigma*sigma )) * (exp(-(x*x + y*y) / (2*sigma*sigma)));
end


%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function name:   
%   generate1D(S, minVal, maxVal)
%
% Description:
%   Generates a 1D array with values from minVal to maxVal
%
% Parameters:
%   S:          array size
%   minVal:     minimum value in array
%   maxVal:     maximum value in array
%
% Output:
%   h:          random 1D array
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}
function h = generate1D(S,minVal,maxVal)
    h = zeros(S);
    for s = 1:S    
       h(s) = minVal + rand()*(maxVal-minVal);
    end
end


%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function name:   
%   generate2D(M, N, minVal, maxVal)
%
% Description:
%   Generates a 2D array with values from minVal to maxVal
%
% Parameters:
%   M:          number of rows
%   N:          number of columns
%   minVal:     minimum value in array
%   maxVal:     maximum value in array
%
% Output:
%   h:          random 2D array
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}
function f = generate2D(M,N,minVal,maxVal)
    f = zeros(M,N);
    for m = 1:M         
        for n = 1:N
           f(m,n) = minVal + rand()*(maxVal-minVal);
        end
    end
end


%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function name:   
%   generate4D(M, N, minVal, maxVal)
%
% Description:
%   Generates a 2D array with values from minVal to maxVal
%
% Parameters:
%   M:          number of rows
%   N:          number of columns
%   minVal:     minimum value in array
%   maxVal:     maximum value in array
%
% Output:
%   h:          random 4D array
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}
function h = generate4D(M,N,minVal,maxVal)
    h = zeros(M,N,M,N);
    % generate random values for h[u][v][m][n]
    for u = 1:M         
        for v = 1:N
            for m = 1:M
                for n = 1:N
                    h(u,v,m,n)= minVal + rand()*(maxVal-minVal);
                end
            end
        end
    end
end



