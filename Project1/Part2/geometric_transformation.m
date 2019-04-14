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

% figure
h1um = generate2D(M, M, min, max);
h2vn = generate2D(N, N, min, max);
IMG3b = SLT(f,h1um, h2vn);
% imshow(IMG3b);
% title("IMG3b: Separable Linear Transform (SLT)");


%figure
hcmn = generate2D(M, N, min, max);
IMG3c = CLT(f,hcmn);
%imshow(IMG3c);
%title("IMG3c: Circular Linear Transform (CLT)");



%figure
h1cm = generate1D(M, min, max);
h2cn = generate1D(N, min, max);
IMG3d = SCT(f,h1cm,h2cn);
%test1 = generate1D(533, min, max);
%test2 = generate1D(799, min, max);
%IMG3d = SCT(IMG1fp,test1,test2);
%imshow(IMG3d);
%title("IMG3d: Separable Convolution linear transform (SCT)");

figure
r = 5;
IMG4a = SDLF_2D_Cylinder(IMG1fp,r);
imshow(IMG4a)
title(sprintf("IMG4a: Spatial Domain Linear Filtering, 2D Cylindrical Filter, w/ radius = %d", r));


figure
sigma = 2.0;
IMG4b = SDLF_Gaussian(IMG1fp,sigma);
imshow(IMG4b)
title(sprintf("IMG4b: Spatial Domain Linear Filtering , Gausian Filter, w/ sigma = %d", sigma));



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

% 
% function F = filter(img, sigma, k, xn, yn, xc, yc, minx, maxx, miny, maxy, c)
%     
%     % compute filter coefficients
%     sum = 0.0;
%     for m1 = -k : k
%         for n1 = -k : k
%              
%             % make sure the indices are within bounds. If they are 
%             % not within image bounds, then set the filter coeff 
%             % below to zero (and do not add that weight in 
%             % computing the normalization_factor.
%              xs = xn-m1; % filter sample x point
%              ys = yn-n1; % filter sample y point
%              if(xs >= minx  && xs <= maxx && ys >= miny && ys <= maxy)
%                  
%                 sampleValue = img(xs,ys,c); % get sample value 
%                 
%                 % make sure the indices are within bounds
%                 %if(m1-xc > minx  && m1-xc < maxx && n1-yc > miny && n1-yc < maxy)
%                 %filterCoeff = (sigma/normalization_factor) * img((m1-xc),(n1-yc)); 
%                 filterCoeff = (1.0) * gaussian_filter((m1-xc),(n1-yc), sigma); 
%                 % normalization factor is the sum of 
%                 % all those filter coeffs for which
%                 % sample I6(xn-m1,yn-n1,:) is available (within the
%                 % image) of 
%                 % the filter for this point (xc,yc).
%                  sum = sum +(filterCoeff * sampleValue);
%                 %end
%              end
%          end
%     end
%     % Normalize the Coefficients
%     normalization_factor = sum;
%     sum = 0.0;
%     for m1 = -k : k
%         for n1 = -k : k
%              xs = xn-m1; % filter sample x point
%              ys = yn-n1; % filter sample y point
%              if(xs >= minx  && xs <= maxx && ys >= miny && ys <= maxy)
%                 sampleValue = img(xs,ys,c);
%                 filterCoeff = (1.0/normalization_factor) * gaussian_filter((m1-xc),(n1-yc), sigma); 
%                 sum = sum +(filterCoeff * sampleValue);
%              end
%          end
%     end   
%     F = sum;
% end
%}

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



