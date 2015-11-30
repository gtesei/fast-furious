function mask = createretinamaskcolored(img)
% mask = createretinamaskcolored(img)
%
% Receives a nonmydriatic colored image and creates a mask with the
% region of interest that is inside the camera's aperture.

%
% Copyright (C) 2006  João Vitor Baldini Soares
%
% This program is free software; you can redistribute it and/or
% modify it under the terms of the GNU General Public License
% as published by the Free Software Foundation; either version 2
% of the License, or (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 51 Franklin Street, Fifth Floor,
% Boston, MA 02110-1301, USA.
%

% Uses the red channel for finding the mask.
red = img(:,:,1);
red = double(red) ./ 255;

%Calculates edges with laplacian of gaussian
edges = edge(red, 'log', 0.0005, 3);

% Fills in little missing parts in edges.
edges = myimclose(edges, strel('diamond', 2));
  
% Adds a countour around the edges image.
[nlins,ncols] = size(edges);
edges(1,:) =     ones(1, ncols);
edges(nlins,:) = ones(1, ncols);
edges(:,1) =     ones(nlins, 1);
edges(:,ncols) = ones(nlins, 1);

%figure; imshow(edges);
% Creates the seed for the outer region by thresholding.
maxred = max(red(:));
seed = uint8(red < 0.15 * maxred);
seed = bwareaopen(seed, 10);

% Takes seed away from the border.
seed(1,:) =     zeros(1, ncols);
seed(nlins,:) = zeros(1, ncols);
seed(:,1) =     zeros(nlins, 1);
seed(:,ncols) = zeros(nlins, 1);
seed = imerode(seed, strel('diamond', 10));
%figure; imshow(seed);

% Fills the seed until it reaches the edges of tha aperture.
notmask = imfill(edges, find(seed > 0));
mask = ~notmask;
mask = imdilate(mask, strel('diamond', 1));

%figure; imshow(mask);

% Filling in missing parts.
mask = bwareaclose(mask, 5000);

% Removing false positive contour.
mask = imopen(mask, strel('diamond', 6));

%Removing other false positives.
mask = bwareaopen(mask, 50000);

%figure; imshow(mask);

function bw2 = bwareaclose(bw1, n)

bw2 = ~bwareaopen(~bw1, n);

% For some reason, imclose from MATLAB 6.0 doesn't work properly, so
% we use this one instead.
function closed = myimclose(img, element)

img = imdilate(img, element);
closed = imerode(img, element);
