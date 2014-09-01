function [bestEpsilon bestF1] = selectThreshold(yval, pval)

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)

    predictions = pval > epsilon;
    true_positives = sum((predictions == 1) & (yval == 1));
    false_positives = sum((predictions == 1) & (yval == 0));
    false_negatives = sum((predictions == 0) & (yval == 1));

    precision = 0;
    if ((true_positives + false_positives) > 0)
      precision = true_positives / (true_positives + false_positives);
    endif
    recall = 0;
    if ( (true_positives + false_negatives)  > 0)
      recall = true_positives / (true_positives + false_negatives);
    endif

    F1 = 0;
    if ((precision + recall) > 0 )
      F1 = 2 * precision * recall / (precision + recall);
    endif

    if (F1 > bestF1)
       bestF1 = F1;
       bestEpsilon = epsilon;
    endif
endfor

end