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

IMG0 = imread('images/mona-lisa.png');
%IMG0 = imread('images/food1.jpg');
[M, N, C] = size(IMG0);     % M: Num. of Rows , 
                            % N : Num. of Columns , 
                            % C : Num. of color bands = 3
figure
imshow(IMG0);
title("IMG0: Original Image");

% figure
% I6 = double(I1)/255.0;
% imshow(I6);
% title('I6: Original Image w/ fp')

figure
IMG1= rgb2gray(IMG0);
imshow(IMG1);
title("IMG1: Grayscale");
 
figure
IMG1fp = double(IMG1)/255.0;
imshow(IMG1fp);
title("IMG1fp: Grayscale w/ fp");
 
% change this matrix A for different rotation, scaling, 
% and affine transformation
 
% Affine transform matrix A
% This specifies the transformation that the input image
% must undergo to form the output image
% General form is
%
% A  = [ a11  a12
%       a21   a22 ]
%
% Test input is for rotation , scaling x-axis, and translation T
%
theta=0;
A = [  cosd(theta) -sind(theta)
     sind(theta)  cosd(theta) ];
% A = [  1 0
%       0  1 ];    
%T = [ 10 5 ]'; % change this for translations
T = [ 0 0 ]'; % change this for translations
% In Affine transform, straight lines map to 
% straight lines. 
% Therefore, first map corner points (1,1),
% (M,1), (1,N), and (M,N)
 
p = A * [ 1 1 ]' + T; % first corner point
x1=p(1);
y1=p(2);
p= A * [ 1 N ]' + T; % second corner point
x2=p(1);
y2=p(2);
p= A * [ M 1 ]' + T; % third corner point
x3=p(1);
y3=p(2);
p= A * [ M N ]' + T; % fourth corner point
x4=p(1);
y4=p(2);
 
% Determine background image size (excluding translation)
xmin = floor( min( [ x1 x2 x3 x4 ] ));
xmax = ceil( max( [ x1 x2 x3 x4 ] ));
ymin = floor(min( [ y1 y2 y3 y4 ] ));
ymax = ceil(max( [ y1 y2 y3 y4 ] ));
Mp=ceil(xmax-xmin)+1; % number of rows
Np=ceil(ymax-ymin)+1; % number of columns
 
I8=zeros(Mp,Np); % output gray scale image
 
% I4=zeros(Mp,Np,3); % output color image
% I5=zeros(Mp,Np,3); % output color image 
% We need to map position of output image pixels
% to a position in the input image. Therefore, find the
% inverse map.
 
 

% figure
% imshow(I8);
% title('I8');

figure
S = 5;
IMG2a = medianfilter(IMG1fp, S);
imshow(IMG2a);
title(sprintf("IMG2a: Median Filter, %dx%d", S, S));

figure
S = 3;
K = 4;
IMG2b = knearestneighbor(IMG1fp, S, K);
imshow(IMG2b);
title(sprintf("IMG2b: %dx%d K-Nearest Neighbors,  K = %d", S, S, K));




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function name:   
%   medianfilter(img, S)
%
% Description:
%   Computes edge-preserving noise smoothing using SxS Median Filter
%
% Parameters:
%   img:        image to be filtered
%   S:          filter size
%
% Output:
%   MedFilter:  SxS Median filtered image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function MF = medianfilter(img, S)
    [M, N] = size(img);     % M: Num. of Rows
                            % N: Num. of Columns 
    P = floor(S / 2); 
    Q = floor(S / 2);                        
    total = (2*P+1)*(2*Q+1);
    med = zeros(1, total);
    for m = 1:M     % each row of input 
        for n = 1:N % each column of input 
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
                  
                    point = img(k,l);
                    med(r) = point;
                    r = r + 1;
                end 
            end
            %sort(med);
            MF(m,n) = median(med);
            %MF(m,n) = med(round(total/2));
        end
    end         
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function name:   
%   knearestneighbor(img, S, K)
%
% Description:
%   Computes edge-preserving noise smoothing using SxS K-Nearest Neighbor
%
% Parameters:
%   img:    image to be filtered
%   S:      filter size (SxS)
%   K:      value for number of neighbors
%
% Output:
%   KNN:    an SxS k nearest neighbor filtered image	
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function KNN = knearestneighbor(img, S, K)
    [M, N] = size(img);     % M: Num. of Rows , 
                            % N: Num. of Columns 
    P = floor(S / 2); 
    Q = floor(S / 2);                        
    total = (2*P+1)*(2*Q+1);
    centerindex = 2*P+1;
    neighbors = zeros(1, total);
    distance = zeros(1, total);
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
                    
                    center = img(m,n);
                    point = img(k,l);
                    neighbors(r) = point;
                    distance(r) = abs(point - center);
                    r = r + 1;
                end 
            end
            
            % don't include center index when finding K smallest distances
            %distance(centerindex) =[];
            [~, kindex] = mink(distance,K+1);
            
            mean = 0;
            for i = 1:K+1
               meanindex = kindex(i);
               % adjust indexes after deleted center index
               if(meanindex ~= centerindex)
                   %meanindex = meanindex + 1;
                   mean = mean + neighbors(meanindex);
               end
               
            end
            mean = mean / K;
            KNN(m,n) = mean;
%             for k = 1:K
%                 
%                 MF(m,n) = neighbors(round(total/2));
        end
    end         
end







