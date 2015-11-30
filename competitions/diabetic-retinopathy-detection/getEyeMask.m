function [ eyeMask ] = getEyeMask( img , varargin)

%% parse params 
if isempty(varargin)
   perimeter_tickness = 10;
   do_plot = 0;
elseif (length(varargin) == 1 && strcmp(varargin{1}, 'do_plot')) || ...
        (length(varargin) == 3 && strcmp(varargin{3}, 'do_plot'))
   perimeter_tickness = 10;
   do_plot = 1; 
elseif (length(varargin) == 2 || length(varargin) == 3) && strcmp(varargin{1}, 'perimeter')
   perimeter_tickness = varargin{2};
   do_plot = 0; 
elseif length(varargin) == 3 && strcmp(varargin{2}, 'perimeter')
   perimeter_tickness = varargin{3};
   do_plot = 0; 
else 
   error('Incorrect inputs.')
end

%% create mask 
if size(img,3) > 1 
grayImg = rgb2gray(img);
else 
    grayImg = img;
end
eyeMask = im2bw(grayImg , graythresh(grayImg));
eyeMask = imfill(eyeMask,'holes');

%% refine mask 
CC = bwconncomp(eyeMask);
numPixels = cellfun(@numel,CC.PixelIdxList);
[biggest,idx] = max(numPixels);

% reset other small CCs (noise)
for i = 1:length(CC.PixelIdxList)
    if (i ~= idx)
        eyeMask(CC.PixelIdxList{i}) = 0;
    end
end


% find perimter
perimeter = bwperim(eyeMask);
perimeter = imdilate(perimeter,strel('disk',perimeter_tickness));

% sutract the perimeter to the mask
old_mask = eyeMask;
eyeMask(perimeter > 0) = 0;

%% plot 
if do_plot
    figure
    subplot(2,2,1);
    imshow(img)
    title('Initial image')
    
    subplot(2,2,2);
    imshow(eyeMask)
    title('final mask')
    
    subplot(2,2,3);
    imshow(perimeter)
    title('perimeter')
    
    subplot(2,2,4);
    imshow(old_mask)
    title('Mask with perimter')
end

