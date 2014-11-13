function [embedm fnn1 fnn2]=fnn(y,maxm)

% Usage: This function  calculates corrected false nearest neighbour.
 
% Inputs: 
%   y  is a  vertical vector of time series.
%   maxm: maximum value of embedding dimension.
 
% Output:
%   embedm: proper value for embedding dimension.
%   fnn1: First criteria of false nearest neighbors.
%   fnn2: second criteria of false nearest neighbors.
 
 
% Copyright(c) Shapour Mohammadi, University of Tehran, 2009
% shmohammadi@gmail.com
 
% Keywords: Embedding Dimension, Chaos Theory, Lyapunov Exponent, 
% False Nearest Neighbors.
 
% Ref:
% -Sprott, J. C. (2003). Chaos and Time Series Analysis. Oxford University
%  Press.



%__________________________________________________________________________
y=y(:);
RT=15;
AT=2;
sigmay=std(y);
[nyr,nyc]=size(y);
%Embedding matrix
m=maxm;

    EM=lagmatrix(y,0:m-1);

%EM after nan elimination.
EEM=EM(1+(m-1):end,:);
[rEEM cEEM]=size(EEM);

embedm=[];

for k=1:cEEM
fnn1=[];
fnn2=[];
   D=dist(EEM(:,1:k)'); 
   
   for i=1:rEEM-m-k
       
       d11 = min(D(i,1:i-1));
       d12=min(D(i,i+1:end));
       Rm=min([d11;d12]);
       l=find(D(i,1:end)== Rm);
       if Rm>0
       if l+m+k-1<nyr 
       fnn1=[fnn1;abs(y(i+m+k-1,1)-y(l+m+k-1,1))/Rm];
       fnn2=[fnn2;abs(y(i+m+k-1,1)-y(l+m+k-1,1))/sigmay];
       end 
       end
   end
   Ind1=find(fnn1>RT);
   Ind2=find(fnn2>AT);
   if length(Ind1)/length(fnn1)<.1 && length(Ind2)/length(fnn1)<.1;
   embedm=k; break
   
   end
end

%_____________________________End__________________________________________

