function yhat = nonlin1(beta,x)
 b1 = beta(1);
 b2 = beta(2);
 b3 = beta(3);
 yhat =b1+b2*x+b3*x./exp(b2*x);
end 