%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Reynerio Rubio
% ID# 109899097
% ESE 558 
% SPRING 2019
% 03/04/2019
% 
% GEOMETRIC TRANSFORMATION OF IMAGES
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Read an RGB image color image in images folder,  'images/food1.jpg'.

I1 = imread('images/food1.jpg');
[M, N, C] = size(I1);   % M: Num. of Rows , 
                        % N : Num. of Columns , 
                        % C : Num. of color bands = 3

%figure
%imshow(I1);
%title('I1: Original Image');

figure
I6 = double(I1)/255.0;
imshow(I6);
title('I6: Original Image w/ fp')

%figure
I2 = rgb2gray(I1);
%imshow(I2);
%title('I2: Grayscale');
 
figure
I7 = double(I2)/255.0;
imshow(I7);
title('I7: Grayscale w/ fp');
 
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
theta=5;
A = [  cosd(theta) -sind(theta)
     sind(theta)  cosd(theta) ];
% A = [  1 0
%       0  1 ];    
%T = [ 10 5 ]'; % change this for translations
T = [ 10 5 ]'; % change this for translations
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
 
I4=zeros(Mp,Np,3); % output color image
I5=zeros(Mp,Np,3); % output color image 
% We need to map position of output image pixels
% to a position in the input image. Therefore, find the
% inverse map.
 
Ap = inv(A); 
 
for i = xmin : xmax
    for j = ymin : ymax
        p = Ap * ( [ i j ]' - T );
        
        % coordinates of point where we need to find the
        % image value through interpolation. 
        x0 = p(1);
        y0 = p(2);
        
        % coordinates of nearest sample point
        xn = round(x0);
        yn = round(y0);
        
        xc = x0 - xn; % (xc,yc) gives the displacement
        yc = y0 - yn; %  of filter center h
       
        % make sure the nearest point (xn,yn) is within the
        % input image
         if( (1<=xn) && (xn<=M) && (1<=yn) && (yn<=N) )
             
             x=round(i-xmin+1);  % shift (xmin, ymin)
                                 % pixel position (1,1)
                                 % in the output image
 
             y=round(j-ymin+1);
 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %  BILINEAR INTERPOLATION
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % coordinates of sample points for bilinear interpolation are
            minx = floor(x0); 
            maxx = ceil(x0);
            miny = floor(y0);
            maxy = ceil(y0);
            
            % Check Bounds for sample points
            if ((1 <= minx) && (maxx <= M) && (1 <= miny) && (maxy <= N))
                % Interpolate for each RGB channels separately 
                % (c: 1 = red, 2 = green, 3 = blue)
                for c = 1:C
                    I4(x,y,c) = bilinearInterpolation(I6, x0, y0, minx, maxx, miny, maxy, c);
                end
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            %  USING THE CONVOLUTION INTERPOLATION FILTER (GAUSSIAN)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Filter size (2k+1)X(2k+1)=5X5 
            %  Note that the (2k+1)X(2k+1)sample window is not precisely 
            %  at the center (x0,y0), but at (xn,yn). This results in some
            %  approximation which is small if the farthest weights of the 
            %  filter are relatively small near its border. 
            %
            sigma = 1.0;
            k = 2;
            % Filter for each RGB channels separately 
            % (c: 1 = red, 2 = green, 3 = blue)
             % Check Bounds for sample points
            if ((1 <= minx) && (maxx <= M) && (1 <= miny) && (maxy <= N))
                % Interpolate for each RGB channels separately 
                % (c: 1 = red, 2 = green, 3 = blue)
                for c = 1:C
                    I5(x,y,c) = filter(I6, sigma, k, xn, yn, xc, yc, minx, maxx, miny, maxy, c);
                end
            end
        end
    end
end
 

%imshow(I3 , [ 0 255 ]);
%title('I3');
 
figure
imshow(I4);
title('I4: Bilinear Interpolation');

figure
imshow(I5);
title('I5: Normalized Gaussian Filter');

%figure
%imshow(I8);
%title('I8');


figure
I9 = zoomBilinear(I6, 0.25);
imshow(I9);
title('I9: Zoomed (0.25)');

figure
I10 = rotate(I6, 150, 5, 70);
imshow(I10);
title('I10: Rotated 150, 5) @ 70°');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function name:   
%   bilinearInterpolation(img, x0, y0, minx, maxx, miny, maxy, c)
%
% Description:
%   Scales image by factor "scale" using bilinear interpolation.
%
% Parameters:
%   img:    image to be sampled
%   x0:     x coordinate to find value for interpolation            
%   y0:     y coordinate to find value for interpolation    
%   minx:	minimum x value in this iteration of interpolation
%   maxx:	maximum x value in this iteration of interpolation
%   miny:	minimum y value in this iteration of interpolation
%   maxy:	maximum y value in this iteration of interpolation
%   c:      color band (c: 1 = red, 2 = green, 3 = blue)
%
% Output:
%   B:  sampled point with bilinear interpolation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function B = bilinearInterpolation(img, x0, y0, minx, maxx, miny, maxy, c)
    s1 = img(minx, miny, c);
    s2 = img(minx, maxy, c); 
    s3 = img(maxx, miny, c);
    s4 = img(maxx, maxy, c);
    f_xy1 = (maxx - x0)/(maxx - minx) * s1 + (x0 - minx)/(maxx - minx) * s3;
    f_xy2 = (maxx - x0)/(maxx - minx) * s2 + (x0 - minx)/(maxx - minx) * s4;
    B = (maxy - y0)/(maxy - miny) * f_xy1 + (y0 - miny)/(maxy - miny) * f_xy2;
end
 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  USING THE CONVOLUTION INTERPOLATION FILTER (GAUSSIAN)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function name:   
%  filter(img, sigma, k, xn, yn, xc, yc, minx, maxx, miny, maxy, c)
%
% Description:
%   USING THE CONVOLUTION INTERPOLATION FILTER (GAUSSIAN)
%   interpolation.
%
% Parameters:
%   img:    image to be sampled
%   x0:     x coordinate to find value for interpolation            
%   y0:     y coordinate to find value for interpolation    
%   minx:	minimum x value in this iteration of interpolation
%   maxx:	maximum x value in this iteration of interpolation
%   miny:	minimum y value in this iteration of interpolation
%   maxy:	maximum y value in this iteration of interpolation
%   c:      color band (c: 1 = red, 2 = green, 3 = blue)
%
% Output:
%   B:  sampled point with bilinear interpolation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function F = filter(img, sigma, k, xn, yn, xc, yc, minx, maxx, miny, maxy, c)
    
    % compute filter coefficients
    sum = 0.0;
    for m1 = -k : k
        for n1 = -k : k
             
            % make sure the indices are within bounds. If they are 
            % not within image bounds, then set the filter coeff 
            % below to zero (and do not add that weight in 
            % computing the normalization_factor.
             xs = xn-m1; % filter sample x point
             ys = yn-n1; % filter sample y point
             if(xs >= minx  && xs <= maxx && ys >= miny && ys <= maxy)
                 
                sampleValue = img(xs,ys,c); % get sample value 
                
                % make sure the indices are within bounds
                %if(m1-xc > minx  && m1-xc < maxx && n1-yc > miny && n1-yc < maxy)
                %filterCoeff = (sigma/normalization_factor) * img((m1-xc),(n1-yc)); 
                filterCoeff = (1.0) * gaussian_filter((m1-xc),(n1-yc), sigma); 
                % normalization factor is the sum of 
                % all those filter coeffs for which
                % sample I6(xn-m1,yn-n1,:) is available (within the
                % image) of 
                % the filter for this point (xc,yc).
                 sum = sum +(filterCoeff * sampleValue);
                %end
             end
         end
    end
    % Normalize the Coefficients
    normalization_factor = sum;
    sum = 0.0;
    for m1 = -k : k
        for n1 = -k : k
             xs = xn-m1; % filter sample x point
             ys = yn-n1; % filter sample y point
             if(xs >= minx  && xs <= maxx && ys >= miny && ys <= maxy)
                sampleValue = img(xs,ys,c);
                filterCoeff = (1.0/normalization_factor) * gaussian_filter((m1-xc),(n1-yc), sigma); 
                sum = sum +(filterCoeff * sampleValue);
             end
         end
    end   
    F = sum;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function h = gaussian_filter(x, y, sigma)
    h = (1 / (2 * pi * sigma*sigma )) * (exp(-(x*x + y*y) / (2*sigma*sigma)));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function name:   
%   zoomBilinear(img, scale)
%
% Description:
%   Scales image by factor "scale" using bilinear
%   interpolation.
%
% Parameters:
%   img:    image to be scaled
%   scale:	value to scale image by 
%
% Output:
%   Z = zoomed/scaled image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Z = zoomBilinear(img, scale)
    [r, c , ~] = size(img);  
    zr = r * scale ;
    zc = c * scale;

    for i = 1 : zr

        x = i / scale;

        x1 = floor(x);
        x2 = ceil(x);
        if x1 == 0
            x1 = 1;
        end
        xrem = rem(x,1);
        
        for j = 1 : zc

            y= j / scale;

            y1 = floor(y);
            y2 = ceil(y);
            if y1 == 0
                y1 = 1;
            end
            yrem = rem(y,1);

            q11 = img(x1,y1,:);
            q12 = img(x1,y2,:);
            q21 = img(x2,y1,:);
            q22 = img(x2,y2,:);

            f_xy1 = q21 * yrem + q11 * (1 - yrem);
            f_xy2 = q22 * yrem + q12 * (1 - yrem);

            z_img(i,j,:) = (f_xy1 * xrem) + (f_xy2 *(1 - xrem));
        end
    end
    Z = z_img;

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function name: 
%   rotate(img, xr, yr, theta)
%
% Description:
%   Rotates given image around a center of rotation (xr,yr)
%   and given andle of rotation theta.
%
% Parameters:
%   img:    image array to be rotated
%   xr:     x point in center of rotation
%   yr:     y point in center of rotation
%   theta:	angle of rotation  
%
% Output:
%   R = rotated image
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function R = rotate(img, xr, yr, theta)
    [r, c , d] = size(img);  
    rotatedImage = zeros(r,c,d);
     for i = 1:r
         xvalue = i - xr;
         for j = 1:c
             yvalue = j - yr;

             y = round(( xvalue*cos(theta) + yvalue*sin(theta))) + xr;
             x = round((-xvalue*sin(theta) + yvalue*cos(theta))) + yr;

             if (x >= 1 && y >= 1 && x <= c && y <= r)
                 rotatedImage(i, j,:) = img(x,y,:);
             end
         end
     end
     R = rotatedImage;
end

