function [ vesselMask ] = my_vessel_segment( img )
%% extract grayscale representation 
grayImg = rgb2gray(img);
%h(1) = imshow(grayImg);
%title('Grayscale Image');


%% it migth be useful create a mask of the eye
eyeMask = im2bw(grayImg , graythresh(grayImg));
eyeMask = imfill(eyeMask,'holes');
%imshow(eyeMask);
%title ('Eye Mask');

%% apply mask to elimiate the background
grayImg(~eyeMask) = 0;
%imshow(grayImg);
%title ('Eye Mask - backgroung eliminated');


%% segment the vessel 
%vesselMask = edge(grayImg,'canny',0.10,1);
vesselMask = edge(grayImg,'canny',0.07,3);
%figure(2),imshow(vesselMask);
%vs = imgca;

%% get rid of edges 
vesselMask(~ imerode(eyeMask,strel('disk',6))) = 0;
%imshow(vesselMask);

%% Dilate 
vesselMask = imdilate(vesselMask,strel('disk',6));
%imshow(vesselMask);

%doc bwmorph

%% ... then skeletonize to get rid of small spurs 
vesselMask = bwmorph(vesselMask,'skel',Inf);
% get rid of sime suprs 
vesselMask = bwmorph(vesselMask,'spur',5);
%imshow(vesselMask);


end

