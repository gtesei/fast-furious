function [ area , area_ratio , blook_vessel_mask ] = get_vessel_area( img , varargin )
%% extract grayscale representation 
grayImg = rgb2gray(img);


%% it migth be useful create a mask of the eye
eyeMask = im2bw(grayImg , graythresh(grayImg));
eyeMask = imfill(eyeMask,'holes');

%% apply mask to elimiate the background
grayImg(~eyeMask) = 0;

cc = bwconncomp(grayImg);
stats = regionprops(cc,'basic');
A = [stats.Area];
area_big = sum(A);


%% segment the vessel 
%vesselMask = edge(grayImg,'canny',0.10,1);
vesselMask = edge(grayImg,'canny',0.07,3);

%% get rid of edges 
vesselMask(~ imerode(eyeMask,strel('disk',6))) = 0;
vesselMask_not_dil = vesselMask;

%% Dilate 
vesselMask = imdilate(vesselMask,strel('disk',6));
vesselMask_dilated = vesselMask;

%% ... then skeletonize to get rid of small spurs 
vesselMask = bwmorph(vesselMask,'skel',Inf);

% get rid of sime suprs 
vesselMask = bwmorph(vesselMask,'spur',5);

cc = bwconncomp(vesselMask);
stats = regionprops(cc,'basic');
A = [stats.Area];
area = sum(A);
area_ratio = area / area_big;

%% blook vessel mask 
blook_vessel_mask = vesselMask;
% img_no_vessel = img;
% img_no_vessel2 = img;
% 
% img_no_vessel(blook_vessel_mask,1) = 0;
% img_no_vessel(blook_vessel_mask,2) = 0;
% img_no_vessel(blook_vessel_mask,3) = 0;
% 
% img_no_vessel2(vesselMask_dilated,1) = 0;
% img_no_vessel2(vesselMask_dilated,2) = 0;
% img_no_vessel2(vesselMask_dilated,3) = 0;

if length(varargin) == 1 && strcmp(varargin{1}, 'do_plot')
    figure
    subplot(3,2,1);
    imshow(img)
    title('Initial truecolor image')
    
    subplot(3,2,2);
    imshow(grayImg)
    title('Grayscale intensity image from image')
    
    subplot(3,2,3);
    imshow(eyeMask)
    title('The eye mask')
  
    subplot(3,2,4);
    imshow(vesselMask_not_dil)
    title('Not dilated')
    
    subplot(3,2,5);
    imshow(vesselMask_dilated)
    title('Dilated')
    
    subplot(3,2,6);
    imshow(vesselMask)
    title('Blood vessel segmentation')
end