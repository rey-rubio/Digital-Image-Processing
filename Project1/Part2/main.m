%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reynerio Rubio
% ID# 109899097
% ESE 558: Digital Image Processing
% SPRING 2019
% 04/13/2019
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%IMG0 = imread('images/mona-lisa.png');
%IMG0 = imread('images/food1.jpg');
%IMG0 = imread('images/aaron-judge-128x128.jpg');
%IMG0 = imread('images/photographer.png');
%IMG0 = imread('images/google-earth.jpg');
IMG0 = imread('images/aaron-judge-grey-128x128.jpg');
figure; imshow(IMG0); title("IMG1: Original Image");

% IMG0fp = double(IMG0)/255.0;
% figure; imshow(IMG0fp); title('IMG0fp: Original Image w/ fp')

IMG1= rgb2gray(IMG0);
% figure; imshow(IMG1); title("IMG1: Grayscale");
 
IMG1fp = double(IMG1)/255.0;
figure; imshow(IMG1fp); title("IMG1fp: Grayscale w/ fp");
 
S = 5;
IMG2a = median_filter(IMG1fp, S);
figure; imshow(IMG2a); title(sprintf("IMG2a: Median Filter, %dx%d", S, S));

S = 3;
K = 4;
IMG2b = k_nearest_neighbors(IMG1fp, S, K);
figure; imshow(IMG2b); title(sprintf("IMG2b: %dx%d K-Nearest Neighbors,  K = %d", S, S, K));

min = 0.0; 
max = 100.0;
M = 8;
N = 10;
f = generate2D(M, N, min, max);
%figure; imshow(f); title("f(m,n): INPUT image General Linear Transform");

huvmn = generate4D(M, N, min, max);
IMG3a = general_linear_transform(f,huvmn);
% figure; imshow(IMG3a); title("IMG3a: General Linear Transform (GLT)");

h1um = generate2D(M, M, min, max);
h2vn = generate2D(N, N, min, max);
IMG3b = separable_linear_transform(f,h1um, h2vn);
% figure; imshow(IMG3b); title("IMG3b: Separable Linear Transform (SLT)");

hcmn = generate2D(M, N, min, max);
IMG3c = convolution_linear_transform(f,hcmn);
% figure; imshow(IMG3c); title("IMG3c: Circular Linear Transform (CLT)");

h1cm = generate1D(M, min, max);
h2cn = generate1D(N, min, max);
IMG3d = separable_convolution_transform(f,h1cm,h2cn);
% figure; imshow(IMG3d); title("IMG3d: Separable Convolution linear transform (SCT)");

r = 5;
IMG4a = SDLF_2D_Cylinder(IMG1fp,r);
figure; imshow(IMG4a); title(sprintf("IMG4a: Spatial Domain Linear Filtering, 2D Cylindrical Filter, w/ radius = %d", r));

sigma = 2.0;
IMG4b = SDLF_Gaussian(IMG1fp,sigma);
figure; imshow(IMG4b); title(sprintf("IMG4b: Spatial Domain Linear Filtering , Gausian Filter, w/ sigma = %d", sigma));

IMG5a = Freq_DFT(IMG1fp);
figure; imshow(IMG5a); title("IMG5a: Frequency Domain Filtering, Discrete Fourier Transform");

IMG5b = Freq_IDFT(IMG5a);
figure; imshow(IMG5b); title("IMG5b: Frequency Domain Filtering, Inverse Discrete Fourier Transform");









