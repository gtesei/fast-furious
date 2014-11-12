function  [LLE LLE_mean LLE_sd]=lyaprosen(y,tau,m)

%__________________________________________________________________________
% Usage: Calculates  largest Lyapunov exponent
 
% INPUTES:
%   y: y is vector of values(time series data)
%   tau: embedding lag of state space reconstruction. When you have not
%   any information about tau please let it zero. The code will calculates
%   the tau.
%   m: m is embedding dimension. If you have not any information about 
%   embedding dimension please let it zero. the code will find proper
%   embedding dimension. 
 
% OUTPUTS:
%   LLE: Largest Lyapunov Exponent
%   lambda: Lyapunov exponents for various ks. Plot of this exponents is 
%   very helpful. If embedding dimension be selected correctly lambda curve
%   will have smooth part(or fairly horizontal). If there is no smooth
%   section on the curve, it is better you try with other embedding
%   dimensions.
 
% NOTE1: When user do not have any information about tau, she should let 
%   tau equal to zero(0). In this case the code will use autocorrelation up 
%   to orders 10 to select proper embedding lag(tau). The proper lag is the
%   lag before of first decline of autocorrelation value below 
%   exp(-1)=0.367879441.For data with nonlinear dependency autocorrelation 
%   function is not proper and mutual information criteria will be used for
%   selecting proper lag value(tau). when both of criteria , 
%   Autocorrelation and mutual information fail to select tau, tau=1 is 
%   selected automatically.
 
% NOTE2: When user have not any information about proper value of embedding
%   dimension, she should let it zero(0).In this case code automatically
%   selects proper m by FNN( False Nearest Neighbors) or if this method
%   fails due to high noise in data, the code will use another method named
%   symplectic geometry. This method is a graphical in nature however I use
%   F test for selection of m based on variance change of eigenvalues.
 
 
% NOTE3:The code usually will not give any error, however For noisy data,
%  high embedding dimension may cause stop of the code, and error message
%  such as follows:
 
 
%      ??? Attempted to access R(1); index out of bounds because
%      numel(R)=0.
%      Error in ==> regress at 80
%      p = sum(abs(diag(R)) > max(n,ncolX)*eps(R(1)));
%      Error in ==> lyaprosen at 342
%      [betar]=regress(L(1:Tl), [ones(Tl,1) x]);
 
 
%  Please reduce the proposed embedding dimension shown in command window.
%  
 
 
% Ref: 
% -Rosenstein,M. T., J. J. Collins and C. J. De Luca,(1993). A practical 
%  method for calculating largest Lyapunov exponents from small data sets.
%  Physica D.
% -Hai-Feng Liu, Zheng-Hua Dai, Wei-Feng Li, Xin Gong, Zun-Hong Yu(2005)
%  Noise robust estimates of the largest Lyapunov exponent,Physics Letters
%  A 341, 119127
% -Sprott,J. C. (2003). Chaos and Time Series Analysis. Oxford University
%  Press.
% -Lei, M., Wang Z.,  Feng Z.A method of embedding dimension estimation
%  based on symplectic geometry, Physics Letters A 303 (2002) 179189. 
% -Zeng,X., R. Eykholt, and R. A. Pielke (1991)Estimating the 
%  Lyapunov-Exponent Spectrum from Short Time Series of Low Precision,
%  Physical Review Letters, Vol. 66, Number 25.
 
 
 
% Copyright(c) Shapour Mohammadi, University of Tehran, 2009
% shmohammadi@gmail.com
 
% Keywords: Lyapunov Exponents, Chaos, Time Series, Taylor Expansion,
% Direct Method, Full Automatic selection code. Minimum mutual Information,
% Autocorrelation, False nearest neighbors, Symplectic Geometry. 


tic
if m==0;
   
%_________Determination Embeding Dimension: False Nearest Neighbour________
y=y(:);
RT=15;
AT=2;
sigmay=std(y);
[nyr,nyc]=size(y);
%Embeding matrix
maxm=10;

    EMmm=lagmatrix(y,0:maxm-1);

%EM after nan elimination.
EEMmm=EMmm(1+(maxm-1):end,:);
[rEEMmm cEEMmm]=size(EEMmm);

mopt=[];

for k=1:cEEMmm
fnn1=[];
fnn2=[];
   Dmm=dist(EEMmm(:,1:k)'); 
   
   for i=1:rEEMmm-maxm-k
       
       d11mm = min(Dmm(i,1:i-1));
       d12mm=min(Dmm(i,i+1:end));
       Rm=min([d11mm;d12mm]);
       l=find(Dmm(i,1:end)== Rm);
       if Rm>0
       if l+maxm+k-1<nyr 
       fnn1=[fnn1;abs(y(i+maxm+k-1,1)-y(l+maxm+k-1,1))/Rm];
       fnn2=[fnn2;abs(y(i+maxm+k-1,1)-y(l+maxm+k-1,1))/sigmay];
       end 
       end
   end
   Ind1=find(fnn1>RT);
   Ind2=find(fnn2>AT);
   if length(Ind1)/length(fnn1)<.1 && length(Ind2)/length(fnn1)<.1;
   mopt=k;  break
   
   end
end

m=mopt;

is1=isempty(mopt);
if is1==1

%_______Determination Embedding Dimension: Symplectic Geometry Method______

cnt=0;
figure('name','Symplethic Geometry','NumberTitle','off')
for k=3:1:20
cnt=cnt+1;
X=lagmatrix(y,1:k);
X=X(k+1:end,:);
A=X'*X;

[rA cA]=size(A);
HH=ones(cA);

for i=1:cA
    
    S=A(:,i);
    if i>1
        S(1:i-1,1)=0;
    end
    if norm(S(i+1:end,1),2)>0;
    alpha=norm(S,2);
    E=zeros(rA,1);
    E(i,1)=1;
    roh=norm(S-alpha*E,2);
    omega=(1/roh)*(S-alpha*E);
    H=eye(rA)-2*omega*omega';
    A=H*A;
    HH=HH*H;
    end
    
end

lambda1=real(eig(A));
lambda=sort(lambda1,'descend');
sigma=lambda.^2;
SIGMA=log10(sigma/sum(sigma));
for ii=2:length(SIGMA)-1
    Hyp(ii,1)= vartest2(SIGMA(ii:end),SIGMA(1:end),0.05);
end

ind=find(Hyp==1);
if ~isempty(ind)
Embddim(cnt,1)=ind(1,1);
end

plot(SIGMA,'-*b')
hold on
end
emdim=find(Embddim>0);
embddim=emdim(1,1);

m=embddim
title (['Symplectic Geometry for Determination of Embedding Dimension']);
end

end


if tau==0;
    
%___________________Determination of Embeding Lag: tau_____________________

% A: Autocorrelation 

y=y(:);
[nyr,nyc]=size(y);

[ACF,Lags,Bounds] = autocorr(y(:,1),10,[],[]);
ACF=ACF(2:end);
for l=1:10
    if abs(ACF(l))<=exp(-1),tau=l-1; break,end   
end

if tau==0 

% B:  Minimum Mutual Information
pnts=100;
for im=0:10
    z=lagmatrix(y,im);
d=2;
n=length(z(im+1:end));

endp1=ceil(pnts/10);
endp2=ceil(pnts/10);

minz=min(z(im+1:end));maxz=max(z(im+1:end));grz=(maxz-minz)/(pnts-endp1);
miny=min(y(im+1:end));maxy=max(y(im+1:end));gry=(maxy-miny)/(pnts-endp1);

h1z=(4/(3*n))^(1/5)*std(z(im+1:end));
h1y=(4/(3*n))^(1/5)*std(y(im+1:end));

for k=1:pnts
zi(k,1)=minz+grz*(k-endp2);
yi(k,1)=miny+gry*(k-endp2);

fz(k,1)=(1/((2*pi)^0.5*n*h1z))*sum(exp(-((zi(k,1)-...
    z(im+1:end)).^2)/(2*h1z^2)));
fy(k,1)=(1/((2*pi)^0.5*n*h1y))*sum(exp(-((yi(k,1)-...
    y(im+1:end)).^2)/(2*h1y^2)));

pz(k,1)=(1/((2*pi)^0.5*n*h1z))*sum(exp(-((zi(k,1)-...
    z(im+1:end)).^2)/(2*h1z^2)))*grz;
py(k,1)=(1/((2*pi)^0.5*n*h1y))*sum(exp(-((yi(k,1)-...
    y(im+1:end)).^2)/(2*h1y^2)))*gry;

end

[gz gy]=meshgrid(zi,yi);
sigma=((n*var(z(im+1:end))+n*var(y(im+1:end)))/(n+n))^0.5;
h=sigma*(4/(d+2))^(1/(d+4))*(n^(-1/(d+4)));

for i=1:pnts
    for j=1:pnts
       
       fzy(i,j)=(1/(2*pi*n*h^2))*sum(exp(-((gz(i,j)-z(im+1:end)).^2+...
           (gy(i,j)-y(im+1:end)).^2)/(2*h^2)));
       pzy(i,j)=(1/(2*pi*n*h^2))*sum(exp(-((gz(i,j)-z(im+1:end)).^2+...
           (gy(i,j)-y(im+1:end)).^2)/(2*h^2)))*grz*gry;
       I1zy(i,j)= pzy(i,j)*log(pzy(i,j)/(pz(i)*py(j)));
    end
end

Hz=-(pz'*log(pz));
Hy=-(py'*log(py));

MIzy=(sum(sum(I1zy)));
RMIzy1(im+1,1)=2*MIzy/(Hz+Hy);
RMIzy2(im+1,1)=MIzy/(Hz*Hy)^0.5;
RMIzy3(im+1,1)=MIzy/min(Hz,Hy);
end

MIInd=find(RMIzy1(2:end)<exp(-1)*RMIzy1(1,1));
if size(MIInd,2) > 0 & size(MIInd,1) > 0  
  disp(MIInd);
  if MIInd(1,1)>1 
  tauMI=MIInd(1,1)-1;
  else
   tauMI=1;
  end
else 
  tauMI=1;
end

tau=tauMI;

end
end
%___________________________Defining lags for y____________________________
yreg=y(:);
maxlag=m;
[nyr,nyc]=size(yreg);
yLreg=lagmatrix(y,1:maxlag);
yreg=yreg(maxlag+1:end,1);
yLreg=yLreg(maxlag+1:end,:);
[ryLreg cyLreg]=size(yLreg);

% Regressors Up to 3 degree

X1=yLreg;

num1=0;
num2=0;
X2ij=[];
X3ijk=[];
for i=1:cyLreg
     for j=i:cyLreg
           X2ij=[X2ij yLreg(:,i).*yLreg(:,j)];
           Indexij(num1+1,1)=i;
           Indexij(num1+1,2)=j;
           num1=num1+1;
         for k=j:cyLreg
           X3ijk=[ X3ijk yLreg(:,i).*yLreg(:,j).*yLreg(:,k)];
           Indexijk(num2+1,1)=i;
           Indexijk(num2+1,2)=j;
           Indexijk(num2+1,3)=k;
           num2=num2+1;
      
         end 
     end
end

X=[ones(ryLreg,1) X1 X2ij  X3ijk];

beta =inv (X'*X)*X'*yreg;
e=yreg-X*beta;
myreg=yreg-mean(yreg);
R2=1-e'*e/(myreg'*myreg);
if R2<0
        R2=1;
end

%______________________Defining lags for y:tau_____________________________

%Embeding matrix.(time delay)
EM(1:nyr,1:m)=nan;
for lead=0:m-1
EM(1+lead*tau:nyr,lead+1)=y(1:nyr-lead*tau);
end

%EM after nan elimination.
EEM=EM(1+(m-1)*tau:nyr,:);
[rEEM cEEM]=size(EEM);

%_______________________Loop for distance calculations_____________________

dd=pdist(EEM,'chebychev');
dd=squareform(dd);


mad=std(y);
dd=dd+eye(rEEM)*10*mad;


for k=0:20
for n=1:rEEM-k
    
    l1=find(0.05*(1/R2)*mad<dd(n,1:end-k)<0.1*(1/R2)*mad)';
   
    u=dd(l1+k,n+k);
    LL(n,1) = log(mean(u));
   
end
L(k+1,1)=nanmean(LL);
K(k+1,1)=k;

end

lambda=diff(L)./diff(K);

figure('name','Lyapunov Exponent','NumberTitle','off')
plot(K,L,'.');
title(['Lyapunov Exponent'])
%_________________Nonlinear Regression Layapunov Exponents_________________

%Lmax=max(L);
%L0=L(1);
%Lm=L0+0.9*(Lmax-L0);
%Ldiff=abs(L-Lm);

%Tl=find(Ldiff==min(Ldiff));

%x=K(1:Tl);

%[betar]=regress(L(1:Tl), [ones(Tl,1) x]);


%%%%%%%%
%leasqrfunc = @(beta,x) (beta(1)+beta(2) +beta(3)* x./exp(beta(2)*x) );

%%%%%%%%


%for iii=1:100
##beta = nlinfit(K(1:Tl),L(1:Tl),@nonlin1,[betar;randn(1,1)]);
%beta = leasqr(K(1:Tl),L(1:Tl),randn(1,1));
%LLE1(iii,1)=beta(2,1);
%end
%LLE=mean(LLE1);

LLE = max(L);
LLE_mean = mean(L);
LLE_sd = std(L);

toc
end 

%________________________________END_______________________________________
