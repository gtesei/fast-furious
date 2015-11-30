function printf(varargin)
%printf(varargin)
%   Same as fprintf, but outputs to stdout
%
%Example:
%   >> printf('foo %d\n', 10);
%   foo 10
%   >>
%
%Written by Gerald Dalley (dalleyg@mit.edu), 2004

fprintf(1, varargin{:});