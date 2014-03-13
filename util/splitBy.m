function [Xs,Xns] = splitBy(X,p) 
 
 if ( size(X,1) != length(p) )
   printf("size(X,1)=%i   length(p)=%i \n",size(X,1),length(p)); 
   error ("the length of criteria vect  must be equal to the length of X!");
 elseif ( sum(p!= 0 & p != 1) > 0 )
   error("p must be a vector of 0' and 1'!")
 endif
 
 Xs = [];
 Xns = [];
 for i =1:length(p)
   if ( p(i) )
     Xs = [Xs; X(i,:)];
   else 
     Xns = [Xns; X(i,:)];
   endif  
 endfor
 
endfunction 
