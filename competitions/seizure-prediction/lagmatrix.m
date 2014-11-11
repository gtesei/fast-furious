function xLag = lagmatrix(x , lags)
%LAGMATRIX Create a lagged time series matrix.
%   Create a lagged (i.e., shifted) version of a time series matrix. Positive 
%   lags correspond to delays, while negative lags correspond to leads. This 
%   function is useful for creating a regression matrix of explanatory 
%   variables for fitting the conditional mean of a return series.
%
%   XLAG = lagmatrix(X , Lags)
%
% Inputs:
%   X - Time series of explanatory data. X may be a vector or a matrix. As
%     a vector (row or column), X represents a univariate time series whose
%     first element contains the oldest observation and whose last element
%     contains the most recent observation. As a matrix, X represents a
%     multivariate time series whose rows correspond to time indices in which
%     the first row contains the oldest observations and the last row contains 
%     the most recent observations. For a matrix X, observations across any 
%     given row are assumed to occur at the same time for all columns, and each
%     column is an individual time series.
%
%   Lags - Vector of integer lags applied to each time series in X. All lags are
%     applied to each time series in X, one lag at a time. To include a time 
%     series as is, include a zero lag. Positive lags correspond to delays, and 
%     shift a series back in time; negative lags correspond to leads, and shift 
%     a series forward in time. Lags must be a vector integers.
%
% Output:
%   XLAG - Lagged transform of the time series X. Each time series in X is
%     shifted by each lag in Lags, one lag at a time for each successive time
%     series. Since XLAG is intended to represent an explanatory regression 
%     matrix, XLAG is returned in column-order format, such that each column 
%     is an individual time series. XLAG will have the same number of rows as 
%     observations in X, but with column dimension equal to the product of the 
%     number of time series in X and the number of lags applied to each time 
%     series. Missing values, indicated by 'NaN' (Not-a-Number), are used to 
%     pad undefined observations of XLAG.
%
% Example:
%   The following example creates a simple bi-variate time series matrix X with
%   5 observations each, then creates a lagged matrix XLAG composed of X and 
%   the first 2 lags of X. XLAG will be a 5-by-6 matrix.
%
%      X = [1 -1; 2 -2 ;3 -3 ;4 -4 ;5 -5]  % Create a simple bi-variate series.
%   XLAG = lagmatrix(X , [0 1 2])          % Create lagged matrix.
%
% See also NaN, ISNAN, FILTER.

%   Copyright 1999-2003 The MathWorks, Inc.   
%   $Revision: 1.1.8.1 $   $Date: 2008/04/18 21:15:50 $

if nargin ~= 2
    error('econ:lagmatrix:UnspecifiedInput' , ' Inputs ''X'' and ''Lags'' are both required.');
end

%
% If time series X is a vector (row or column), then assume 
% it's a univariate series and ensure a column vector.
%

if numel(x) == length(x)             % check for a vector.
   x  =  x(:);
end

%
% Ensure LAGS is a vector of integers. 
%

if numel(lags) ~= length(lags)       % check for a non-vector.
   error('econ:lagmatrix:NonVectorLags' , ' ''Lags'' must be a vector.');
end

lags  =  lags(:);                         % Ensure a column vector.

if any(round(lags) - lags)
   error('econ:lagmatrix:NonIntegerLags' , ' All elements of ''Lags'' must be integers.')
end

missingValue  =  NaN;  % Assign default missing value.

%
% Cycle through the LAGS vector and shift the input time series. Positive 
% lags are delays, and can be processed directly by FILTER. Negative lags
% are leads, and are first flipped (reflected in time), run through FILTER,
% then flipped again. Zero lags are simply copied.
%

nLags =  length(lags);  % # of lags to apply to each time series.

[nSamples , nTimeSeries] = size(x);

xLag  =  zeros(nSamples , nTimeSeries * nLags);

for c = 1:nLags

    columns  =  (nTimeSeries*(c - 1) + 1):c*nTimeSeries;   % Columns to fill for this lag.

    if lags(c) > 0     % Time delays.

       xLag(:,columns) = filter([zeros(1,lags(c)) 1] , 1 , x , missingValue(ones(1,lags(c))));

    elseif lags(c) < 0 % Time leads.

       xLag(:,columns) = flipud(filter([zeros(1,abs(lags(c))) 1] , 1 , flipud(x) , missingValue(ones(1,abs(lags(c))))));

    else               % No shifts.

       xLag(:,columns) = x;

    end

end
