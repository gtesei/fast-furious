function [NNMeta] = buildNNMeta(_NNArchVect,featureEncodeOnTheFly=0)

  NNMeta.NNArchVect = _NNArchVect;
  NNMeta.featureEncodeOnTheFly = featureEncodeOnTheFly;

endfunction