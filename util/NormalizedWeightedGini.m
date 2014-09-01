function [ngini] = NormalizedWeightedGini (solution, weights, submission)

  ngini = WeightedGini(solution, weights, submission) / WeightedGini(solution, weights, solution); 

end
