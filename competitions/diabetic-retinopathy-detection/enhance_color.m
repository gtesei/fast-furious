function [ ] = enhance_color( imgIN , varargin)
srgb2lab = makecform('srgb2lab');
lab2srgb = makecform('lab2srgb');

shadow_lab = applycform(imgIN, srgb2lab); % convert to L*a*b*

% the values of luminosity can span a range from 0 to 100; scale them
% to [0 1] range (appropriate for MATLAB(R) intensity images of class double)
% before applying the three contrast enhancement techniques
max_luminosity = 100;
L = shadow_lab(:,:,1)/max_luminosity;

% replace the luminosity layer with the processed data and then convert
% the image back to the RGB colorspace
shadow_imadjust = shadow_lab;
shadow_imadjust(:,:,1) = imadjust(L)*max_luminosity;
shadow_imadjust = applycform(shadow_imadjust, lab2srgb);

shadow_histeq = shadow_lab;
shadow_histeq(:,:,1) = histeq(L)*max_luminosity;
shadow_histeq = applycform(shadow_histeq, lab2srgb);

shadow_adapthisteq = shadow_lab;
shadow_adapthisteq(:,:,1) = adapthisteq(L)*max_luminosity;
shadow_adapthisteq = applycform(shadow_adapthisteq, lab2srgb);

if length(varargin) == 1 && strcmp(varargin{1}, 'do_plot')
    figure
    subplot(2,2,1);
    imshow(imgIN)
    title('Initial Image')
    
    subplot(2,2,2);
    imshow(shadow_imadjust)
    title('Initial Image - imadjust')
    
    subplot(2,2,3);
    imshow(shadow_histeq)
    title('Initial Image - histeq')
    
    subplot(2,2,4);
    imshow(shadow_adapthisteq)
    title('Initial Image - adapthisteq')
end

