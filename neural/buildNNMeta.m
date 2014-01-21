function [NNMeta] = buildNNMeta(NNArchVect,featureEncodeOnTheFly=0)

  NNMeta.NNArchVect = NNArchVect;
  NNMeta.featureEncodeOnTheFly = featureEncodeOnTheFly;

endfunction