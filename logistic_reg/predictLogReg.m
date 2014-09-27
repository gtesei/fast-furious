function [probs] = predictLogReg(theta, X )

	 probs = sigmoid(X * theta);

endfunction 
