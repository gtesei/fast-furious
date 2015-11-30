function params = SystemConfig
% VERSION 09/09/2014
% CREATED BY: George Azzopardi (1), Nicola Strisciuglio (1,2), Mario Vento (2) and Nicolai Petkov (1)
%             1) University of Groningen, Johann Bernoulli Institute for Mathematics and Computer Science, Intelligent Systems
%             1) University of Salerno, Dept. of Information Eng., Electrical Eng. and Applied Math., MIVIA Lab
%
% SystemConfig returns a structure of the parameters required by the
% BCOSFIRE filter

% Use hashtable
params.ht                             = 1;

% The radii list of concentric circles
params.COSFIRE.rholist                = 0:2:8;

% Minimum distance between dominant contours lying on the same concentric circle
params.COSFIRE.eta                    = 150*pi/180;

% Threshold parameter used to suppress the input filters responses that are less than a
% fraction t1 of the maximum
params.COSFIRE.t1                     = 0;

% Threshold parameter used to select the channels of input filters that
% produce a response larger than a fraction t2 of the maximum
params.COSFIRE.t2                     = 0.4;

% Parameters of the Gaussian function used to blur the input filter
% responses. sigma = sigma0 + alpha*rho_i
params.COSFIRE.sigma0                 = 3/6;
params.COSFIRE.alpha                  = 0.8/6;

% Parameters used for the computation of the weighted geometric mean. 

% mintupleweight is the weight assigned to the peripherial contour parts
params.COSFIRE.mintupleweight         = 0.5;
params.COSFIRE.outputfunction         = 'geometricmean'; % geometricmean OR weightedgeometricmean 
params.COSFIRE.blurringfunction       = 'max'; %max or sum

% Weights are computed from a 1D Gaussian function. weightingsigma is the
% standard deviation of this Guassian function
params.COSFIRE.weightingsigma         = sqrt(-max(params.COSFIRE.rholist)^2/(2*log(params.COSFIRE.mintupleweight)));

% Threshold parameter used to suppress the responses of the COSFIRE filters
% that are less than a fraction t3 of the maximum response.
params.COSFIRE.t3                     = 0;

% % Parameters of some geometric invariances
numoriens = 12;
params.invariance.rotation.psilist    = 0:pi/numoriens:(pi)-(pi/numoriens);
params.invariance.scale.upsilonlist   = 2.^((0)./2);

% Reflection invariance about the y-axis. 0 = do not use, 1 = use.
params.invariance.reflection          = 0;

% Minimum distance allowed between detected keypoints. If the distance
% between any two pairs of detected keypoints is less than
% params.distance.mindistance then we keep only the stronger one.
params.detection.mindistance          = 8;

% Parameters of the input filter. Here we use symmetric Gabor filters.
% Gabor filters are, however, not intrinsic to the method and any other
% filters can be used.
params.inputfilter.name                     = 'DoG';

params.inputfilter.Gabor.thetalist          = 0:pi/8:pi-pi/8;
params.inputfilter.Gabor.lambdalist         = 4.*(sqrt(2).^(0:2));
params.inputfilter.Gabor.phaseoffset        = pi;
params.inputfilter.Gabor.halfwaverect       = 0;
params.inputfilter.Gabor.bandwidth          = 2;
params.inputfilter.Gabor.aspectratio        = 0.5;
params.inputfilter.Gabor.inhibition.method  = 1;
params.inputfilter.Gabor.inhibition.alpha   = 0;
params.inputfilter.Gabor.thinning           = 0;
    
params.inputfilter.DoG.polaritylist         = [1];
params.inputfilter.DoG.sigmalist            = 2.4;
params.inputfilter.DoG.sigmaratio           = 0.5;
params.inputfilter.DoG.halfwaverect         = 0;

if strcmp(params.inputfilter.name,'Gabor')
    params.inputfilter.symmetric = ismember(params.inputfilter.Gabor.phaseoffset,[0 pi]);
elseif strcmp(params.inputfilter.name,'DoG')
    params.inputfilter.symmetric = 1;
end