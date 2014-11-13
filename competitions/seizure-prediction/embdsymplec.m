function  [sigma embdim]=embdsymplec(y,m)

%__________________________________________________________________________
% Usage: Determination of embedding dimension based on symplectic geometry.
%  Let X be lag matrix of y, A=X'X, and A(n) is (n-1)times transformed form
%  of A by  Householder  matrix. If descending sorted eigenvalues A(n) tend
%  to be constant for dimension d+1 then d is the proper embedding
%  dimension.
 
% Inputs:
%  y is a vector of time series values. m is maximum embedding dimension.
%  m: maximum embedding dimension.
 
 
% Outputs:
%   sigma is eigenvalues of A matrix. sigma1=Lambda^2max,...,
%   sigman=Lambda^2min,where n is the dimension of X. If sigma1>sigma2> ...
%   >sigmad>=sigmad+1>=...>=sigman,then d is the embedding dimension of the
%   reconstruction system(Lei etal. 2002).The dimension before horizontal
%   section of log(sigma(i)/tr(sigma(i))plot is the proper dimension.
%   embdim: proper embedding dimension.
 
 
 
% Ref:
%  Min Lei Zhizhong Wang Zhengjin Feng(2002). A method of embedding 
%  dimension estimation based on symplectic geometry. Physics Letters A 303
%  ,pp. 179189
 
% Keywords: Noise; Chaos; Embedding dimension; Symplectic geometry.
 
% Copyright(c) Shapour Mohammadi, University of Tehran, 2009
% shmohammadi@gmail.com
 
%__________________________________________________________________________

cnt=0;
for k=3:1:m
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
    %Hyp(ii,1)= vartest2(SIGMA(ii:end),SIGMA(1:end),0.05);
    Hyp(ii,1)= vartest2(SIGMA(ii:end),SIGMA(1:end),'Tail','right');
end

ind=find(Hyp==1);
if ~isempty(ind)
Embdim(cnt,1)=ind(1,1);


end
plot(SIGMA,'-*b')
hold on
end
emdim=find(Embdim>0);
embdim=emdim(1,1);



%_____________________________END__________________________________________
