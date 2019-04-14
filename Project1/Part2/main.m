%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reynerio Rubio
% ID# 109899097
% ESE 558: Digital Image Processing
% SPRING 2019
% 04/13/2019
%
% IMAGES
%   IMG0:         original image
%   IMG1:         converted IMG0 to gray level image
%   IMG1fp:       converted IMG1 to floating point 
%   IMG2a:        computed Median filter for IMG1fp
%   IMG2b:        computed K-Nearest Neighbor filter (KNN) for IMG1fp
%   IMG3a:        computed General Linear Transform (GLT) for IMG1fp
%   IMG3b:        computed Separable Linear Transform (SLT) for IMG1fp
%   IMG3c:        computed Circular Convolution Linear Transform (CLT) for IMG1fp
%   IMG3d:        computed Separable Convolution Transform (SCT) for IMG1fp
%   IMG4a:        computed 2D Cylinder PSF for IMG1fp
%   IMG4b:        computed Gaussian PSF for IMG1fp
%   IMG5a:        computed Frequency-filtered DFT for IMG1fp
%   IMG5b:        computed Frequency-filtered IDFT for IMG5a
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%}

% Pick image to use as input
%IMG0 = imread('input-images/food.jpg');
%IMG0 = imread('input-images/mona-lisa.png');
%IMG0 = imread('input-images/aaron-judge-grey-128x128.jpg');
IMG0 = imread('input-images/food-grey-128x128.jpg');
figure; imshow(IMG0); title("IMG0: Original Image");

% convert to grayscale if not already grayscale
if ndims(IMG0) == 3
    IMG1= rgb2gray(IMG0);
else
    IMG1 = IMG0;
end   
IMG1fp = double(IMG1)/255.0;
figure; imshow(IMG1fp); title("IMG1fp: Grayscale w/ fp");
 
S = 5;  % SxS filter
IMG2a = median_filter(IMG1fp, S);
figure; imshow(IMG2a); title(sprintf("IMG2a: Median Filter, %dx%d", S, S));

S = 3;  % SxS filter
K = 4;  % number of neighbots
IMG2b = k_nearest_neighbors(IMG1fp, S, K);
figure; imshow(IMG2b); title(sprintf("IMG2b: %dx%d K-Nearest Neighbors,  K = %d", S, S, K));

% input parameters needed for parts 3a - 3d
min = 0.0;      % minimum random value
max = 100.0;    % maximum random value
M = 8;          % fixed number of rows
N = 10;         % fixed number of columns
f = generate2D(M, N, min, max); % randomly generated 2D input image
% figure; imshow(f); title("f(m,n): INPUT image");

huvmn = generate4D(M, N, min, max); % randomly generated 4D filter
IMG3a = general_linear_transform(f,huvmn); % compute GLT
% figure; imshow(IMG3a); title("IMG3a: General Linear Transform (GLT)");

h1um = generate2D(M, M, min, max); % randomly generated 2D filter
h2vn = generate2D(N, N, min, max); % randomly generated 2D filter
IMG3b = separable_linear_transform(f,h1um, h2vn); % compute SLT
% figure; imshow(IMG3b); title("IMG3b: Separable Linear Transform (SLT)");

hcmn = generate2D(M, N, min, max); % randomly generated 2D filter
IMG3c = convolution_linear_transform(f,hcmn); % compute CLT
% figure; imshow(IMG3c); title("IMG3c: Circular Linear Transform (CLT)");

h1cm = generate1D(M, min, max); % randomly generated 1D filter
h2cn = generate1D(N, min, max); % randomly generated 1D filter
IMG3d = separable_convolution_transform(f,h1cm,h2cn); % compute SCT
% figure; imshow(IMG3d); title("IMG3d: Separable Convolution linear transform (SCT)");

r = 5; % radius of 2D Cylinder PSF
IMG4a = SDLF_2D_Cylinder(IMG1fp,r); % compute 2D Cylinder PSF
figure; imshow(IMG4a); title(sprintf("IMG4a: Spatial Domain Linear Filtering, 2D Cylindrical Filter, w/ radius = %d", r));

sigma = 2.0; % sigma of Gaussian PSF
IMG4b = SDLF_Gaussian(IMG1fp,sigma); % compute Gaussian PSF
figure; imshow(IMG4b); title(sprintf("IMG4b: Spatial Domain Linear Filtering , Gausian Filter, w/ sigma = %d", sigma));

IMG5a = Freq_DFT(IMG1fp); % compute Freq Filtered DFT
figure; imshow(real(IMG5a)); title("IMG5a: Frequency Domain Filtering, Discrete Fourier Transform");

IMG5b = Freq_IDFT(IMG5a); % compute Freq Filtered Inverse DFT
figure; imshow(IMG5b); title("IMG5b: Frequency Domain Filtering, Inverse Discrete Fourier Transform");









