function [value] = innerProductMat_Y_1_X(y, x)

 s1 = size(x);
 s2 = size(y);
 if (s1 != s2)
	error("Vector dimension doesnt match \n");
 else
	idy = (y == 1);
	value = sum(x(idy));
 endif

endfunction