function [ ] = Application( )
% Delineation of blood vessels in retinal images based on combination of BCOSFIRE filters responses.
%
% VERSION 09/09/2014
% CREATED BY: George Azzopardi (1), Nicola Strisciuglio (1,2), Mario Vento (2) and Nicolai Petkov (1)
%             1) University of Groningen, Johann Bernoulli Institute for Mathematics and Computer Science, Intelligent Systems
%             1) University of Salerno, Dept. of Information Eng., Electrical Eng. and Applied Math., MIVIA Lab
%
%   If you use this script please cite the following paper:
%   "George Azzopardi, Nicola Strisciuglio, Mario Vento, Nicolai Petkov, 
%   Trainable COSFIRE filters for vessel delineation with application to retinal images, 
%   Medical Image Analysis, Available online 3 September 2014, ISSN 1361-8415, 
%   http://dx.doi.org/10.1016/j.media.2014.08.002"
%
%
% EXAMPLE APPLICATION.

% Example with an image from DRIVE data set
image = double(imread('./data/DRIVE/test/images/01_test.tif')) ./ 255;

%% Symmetric filter params
symmfilter = struct();
symmfilter.sigma     = 2.4;
symmfilter.len       = 8;
symmfilter.sigma0    = 3;
symmfilter.alpha     = 0.7;

%% Asymmetric filter params
asymmfilter = struct();
asymmfilter.sigma     = 1.8;
asymmfilter.len       = 22;
asymmfilter.sigma0    = 2;
asymmfilter.alpha     = 0.1;

%% Filters responses
% Tresholds values
% DRIVE -> preprocessthresh = 0.5, thresh = 37
% STARE -> preprocessthresh = 0.5, thresh = 40
% CHASE_DB1 -> preprocessthresh = 0.1, thresh = 38
[resp segresp r1 r2] = BCOSFIRE(image, symmfilter, asymmfilter, 0.5, 37);