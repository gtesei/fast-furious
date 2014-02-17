function [X_out,index] = encodeCategoricalFeatures(X)

[m, n] = size(X);

index = struct('key', 'value');
offset = 1; 
for i = 1:n 
  idxHash = struct('key', 'value');
  labels = unique( X(:,i));
  for j = 1:length(labels)
    idxHash = setfield(idxHash , [num2str(i) num2str(labels(j))], offset);
    index = setfield(index , num2str(i), idxHash);
    offset += 1;
  endfor
endfor

X_out = zeros(m,offset-1);

for i = 1:m 
  for j = 1:n
    idxHash = getfield ( index , num2str(j) );
    idx = getfield ( idxHash ,  [num2str(j) num2str(X(i,j))] );
    X_out(i,idx) = 1; 
  endfor 
endfor


endfunction
