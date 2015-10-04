function [folds] = kfold_bclass(k,y,seed=123)

  if (k>=length(y))
    folds = 1:length(y);
    return; 
  end

  %% seed 
  old_seed = rand("seed");
  rand("seed",seed);

  %% 
  folds = zeros(length(y),1); 
  
  %% class1 
  idx1 = find( y == 1 );
  idx1 = idx1(randperm (length(idx1)));
  folds(idx1) = mod(1:length(idx1) , k);
  offset = mod(1:length(idx1) , k)(end) + 1;    

  %% class 0 
  idx0 = find( y == 0 );
  idx0 = idx0(randperm (length(idx0)));
  folds(idx0) = mod(offset:(length(idx0)+offset-1) , k);
  

  %% 1-based index 
  folds = folds +1;

  %% seed
  rand("seed",old_seed);
 
end 