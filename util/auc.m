function [AUC] = auc (probs , labels, doPlot =0, verbose = 1 ) 
  
  %% check 
  if (size(probs,1) == 1)
    if(size(probs,2) != size(labels,2))
      error("labels has a different dimension than probs")
    end 
  else 
    if(size(probs,1) != size(labels,1))
      error("labels has a different dimension than probs")
    end
  end 

  %% 
  th = linspace( 0.01 , 0.99 , 10000);
  tpr = zeros(size(th,2),1);
  tnr = zeros(size(th,2),1);

  %% 
  for i = 1:size(th,2)
    tpr(i) = sum( (probs >= th(i)) & (labels == 1)  ) / sum((labels == 1));
    tnr(i) = 1 - (sum( (probs < th(i)) & (labels == 0)  ) /sum((labels == 0))); 
  end 

  %% AUC
  height = (tpr(2:end) + tpr(1:(end-1)))/2 ; 
  %width = -diff(fliplr(tnr)) ;
  width = -diff(tnr,1,1);
  AUC = sum(height .* width);  

  if (doPlot)
    plot(tnr, tpr , "linewidth", 3 , "color" , "r" )
    hold on
    plot([0 1] , [0 1] , "linewidth", 0.5 , "color" , "k" , "linestyle" , "--" )
    set(gca, "xlim", [0 1])
    set(gca, "ylim", [0 1])
    set(gca, "xlabel", text("string", "1 - Specificity", "fontsize", 15))
    set(gca, "ylabel", text("string", "Sensitivity", "fontsize", 15))
    set(gca, "title", text("string", "ROC", "fontsize", 17))
    hold off;
  end 

end 