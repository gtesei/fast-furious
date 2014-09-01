function [F1,precision,recall,accuracy] = printClassMetrics (pred_val , yval)

accuracy = mean(double(pred_val == yval)) * 100;
acc_all0 = mean(double(0 == yval)) * 100;
printf("|--> accuracy == %f vs accuracy_all0 == %f \n",accuracy,acc_all0);

actual_positives = sum(yval == 1);
actual_negatives = sum(yval == 0);
true_positives = sum((pred_val == 1) & (yval == 1));
false_positives = sum((pred_val == 1) & (yval == 0));
false_negatives = sum((pred_val == 0) & (yval == 1));
precision = 0; 
if ( (true_positives + false_positives) > 0)
   precision = true_positives / (true_positives + false_positives);
endif 

recall = 0; 
if ( (true_positives + false_negatives) > 0 )
   recall = true_positives / (true_positives + false_negatives);
endif 

F1 = 0; 
if ( (precision + recall) > 0) 
  F1 = 2 * precision * recall / (precision + recall);
endif
  
printf("|-->  true_positives == %i  (actual positive =%i) \n",true_positives,actual_positives);
printf("|-->  false_positives == %i \n",false_positives);
printf("|-->  false_negatives == %i \n",false_negatives);
printf("|-->  precision == %f \n",precision);
printf("|-->  recall == %f \n",recall);
printf("|-->  F1 == %f \n",F1);


end
