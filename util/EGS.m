function [X] = EGS(m,n,links,stage)
    X = [];


    v = 1;
    for a = 2:links
      v = [v zeros(1,stage-1) 1];
    endfor
    if (length(v) < n)
      v = [v zeros(1,(n-length(v))) ];
    end

   X = v;
   for j = 1:(m-1)
      X = [X; shift(v,j)];
   end


endfunction 
