function [ ] = BeforeUsing( )
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
% Compile utility MEX functions before using the filters.

disp('Compiling...');
cd COSFIRE
mex dilateDisc.c -output dilate
cd ..
disp('Compile done.');