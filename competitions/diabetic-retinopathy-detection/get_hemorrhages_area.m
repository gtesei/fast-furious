function [ area , area_ratio ] = get_hemorrhages_area( imgIN , varargin)

%% adjust 
img_adj = imadjust(imgIN(:,:,3),stretchlim(imgIN(:,:,3)) , []);

%% transformations 
hmg = intrans( intrans(img_adj,'stretch'),'log',3) ;
%imshow( hmg );
hmg_bkp = hmg;

%% mask 
grayImg = rgb2gray(imgIN);
eyeMask = im2bw(grayImg , graythresh(grayImg));
eyeMask = imfill(eyeMask,'holes');
hmg(~eyeMask) = 255;
%imshow(hmg); 

%% compute 
hmgBW = im2bw(hmg , graythresh(hmg)) ;
area = nnz(~hmgBW); 
area_ratio = area / nnz(eyeMask);

if length(varargin) == 1 && strcmp(varargin{1}, 'do_plot')
    figure
    subplot(3,3,1);
    imshow(imgIN)
    title('Initial Image')
    
    subplot(3,3,2);
    imshow(img_adj)
    title('Image afeter adjiusting')
    
    subplot(3,3,3);
    imshow(hmg_bkp)
    title('Image afeter transformations')
    
    subplot(3,3,4);
    imshow(eyeMask)
    title('The eye mask')
    
    subplot(3,3,5);
    imshow(hmg)
    title('Final image')
    
    subplot(3,3,6);
    imshow(hmgBW)
    title('Final image as BW')
    
    subplot(3,3,7);
    imhist(hmg)
    title(sprintf('imhist of final image - graythresh:%f %f',graythresh(grayImg)))
end

