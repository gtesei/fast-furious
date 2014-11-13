function nn = false_neighbors_kd(x, m, D, n, r, w, q, dc)
% FALSE_NEIGHBORS_KD estimate appropriate embedding dimension by estimating
%       false nearest neighbors in a reconstructed phase space based on a
%       delay coordinate embedding first advocated in:
%          "Determining embedding dimension for phase-space 
%           reconstruction using a geometrical construction", 
%           M. B. Kennel, R. Brown, and H.D.I. Abarbanel,
%           Physical Review A, Vol 45, No 6, 15 March 1992, 
%           pp 3403-3411.
%       the method was further improved to account for (d+1) nn components
%       that were larger than mean random distance, nn distances that were
%       larger than mean random distances.
%       for more details, please see this post:
%       http://egr.uri.edu/nld/2013/12/16/notes-on-false-nearest-neighbors/
%
% CALL: 
%   nn = false_neighbors_kd(x, m, D, n, r, w, q, dc)
%
% INPUT:
%   x  - scalar time series
%   m  - delay time (if too small use dc = 1)
%   D  - maximum embedding dimension to consider (don't go crazy)
%   n  - number of phase space points to consider
%   r  - (optional, 1 default) number of temporarily uncorrelated nearest 
%        neighbors to consider for each phase space point
%   w  - (optional, 0 default) temporal correlation (Theiler) window
%   q  - (optional, 0 default) temporal neigborhood, floor(strand length/2)
%   dc - (optional, 0 default) set to 1 to remove linear correlations from
%        the reconstructed phase space coordinates caused by small delay
%
% OUTPUT:
%   nn.i - n x D stores true nearest neighbor index in each dimension
%   nn.d - n x D stores distances to nearest neighbors in each dimension
%   nn.z - n x D stores distances between d+1 coordinate of nn.i point
%   nn.s - same as above but for the synthetic surrogates
%
% POSTPROCESSING:
%   plot_false_neighbors.m
%
% copyright 2013-2014 David Chelidze
 
if nargin < 8 || isempty(dc) % do not decorrelate the phase space
    dc = 0;
end
 
if nargin < 7 || isempty(q) % do not look at temporal neighbors or r nns
    q = 0;
end
 
if nargin < 6 || isempty(w) % do not remove temporal correlations
    w = 0;
end
 
if nargin < 5 || isempty(r) % find only one nn
    r = 1;
end
 
if nargin < 4 % need time series
    error('ERROR: need at least four input variables')
elseif min(size(x)) == 1 % normalize input time series
    x = (x(:)' - mean(x))/std(x);
    nm = size(x, 2) - D*m;
    n = min(nm, n);
else % cannot deal with vector-valued time series
    error('ERROR: x needs to be a scalar time series')
end
 
% number of nearest neighbors to look up. should be larger than r.
knn = max(2*r, 6*m); % my heuristic number for nn look up
 
% reconstruct the phase space
y = zeros(D+1, n); 
for d = 1:D+1
    y(d, :) = x((1:n) + (d-1)*m);
end
 
si = randperm(n); % randomized indexes for the surrogate data analysis
 
sw = -q:q; % strand window indices for each r nns
 
dn = zeros(1,r); % initialize distances to nns
 
% initialize output variables
nn.d = zeros(n, D); nn.z = nn.d; nn.s = nn.d; nn.i = nn.d;
 
for d = 1:D % find the false nearest neighbors for each embedding dimension   
    tic % start the timer 
    fprintf('Estimating false nearest neighbors for d = %d\n', d)
    if dc % remove linear correlations
        yd = decorr(y(1:d+1, :), d); % remove linear correlations
    else % do not remove linear correlations
        yd = y(1:d+1, :); 
    end
    
    % partition all phase space points for fast nn searching
    kdtree = KDTreeSearcher(yd(1:d, :)' ,...
                            'Distance', 'euclidean', 'BucketSize', 2*knn);
                        
    % find the nearest neighbor for each of the phase space point
    [nni, nnd] = knnsearch(kdtree, yd(1:d, :)', 'k', knn);
    % find r nearest neighbor record for each of the phase space point:
    ji = zeros(n,r); di = ji;
    for k = 1:n % find the first temporarily uncorrelated nn
        tmp = abs(nni(k,:) - k) > w; % uncorrelated indices
        nit = nni(k, tmp);
        dit = nnd(k, tmp);
        ji(k,:) = nit(1:r);
        di(k,:) = dit(1:r);
    end
    nni = ji; nnd = di;% stores r nns record numbers and the distances
    clear ji di nit dit tmp

    
    for k = 1:n % for each phase space point get the statistics on nns
        
        bsi = k + sw; % k-th base strand indices
        is = (bsi>0 & bsi<n); % valid k-th base strand indices
                
        for l = 1:r % find mean squared distance to all r nearest neighbors
            nsi = nni(k,l) + sw; % l-th nn strand indices
            in =  is & (nsi>0 & nsi<n); % valid l-th nn strand indices
            % the mean squared distance to the l-th nn strand:
            dn(l) = mean(sum((y(1:d, bsi(in)) - y(1:d, nsi(in))).^2, 1));
        end
        
        [~, inn] = min(dn); % find closest nn--"true" nn index
        nn.i(k, d)  = inn; % record number of the "true" nn point
        % for the random data first nn has d-d.o.f. chi distribution
        nn.d(k, d) = nnd(k, inn); % the corresponding distance in d-dims
        nn.z(k, d) = abs(yd(d+1, k) - yd(d+1, nni(k, inn))); 
        % delta lengths in (d+1)-dimension, for random data this has chi
        % distribution at each dimension d
        nn.s(k, d) = abs(yd(d+1, k) - yd(d+1, si(nni(k, inn)))); 
        % same distance for the synthetic surrogate data
    end     
    toc
end
 
%%-------------------------------------------------------------------------
function y = decorr(y,d)
% DECORR    remove linear correlations from y and rotate
 
[~,~,V] = svd(y',0); % singular value decomposition for decorrelation
y = V*y; % decorrelated phase space
% now rotate the phase space so that the first d-coordinates are only a
% function of the first d coordinates:
xe = y(:,end);
u = xe - norm(xe)*[zeros(d,1); 1];
v = u/norm(u);
Q = eye(d+1,d+1) - 2*(v*v');
y = Q*y;       