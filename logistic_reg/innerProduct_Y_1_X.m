function value = innerProduct_Y_1_X(y, x)

 d1 = length(x);
 d2 = length(y);
 if (d1 != d2)
	error("Vector dimension doesnt match \n");
 else
	idy = (y == 1);
	value = sum(x(idy));
 endif

endfunction