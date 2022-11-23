# iter_Ensemble_Smoother
iES based on regularized Levenburg-Marquardt, see the paper "Iterative Ensemble Smoother as an Approximate Solution to a Regularized Minimum-Average-Cost Problem: Theory
and Applications", by Luo et al., SPE-176023-PA.

This demo contains an PYTHON implementation of the aforementioned iES, which is most of the time used in ensemble-based reservoir data assimilation (also known as history
matching) problems. Our main purpose here is to apply iES to infer the input values given a trained DNN model for fitting a simple polynomial function and the 
corresponding output values of the polynomial function.

This code is based on https://github.com/jamesYu365/iterative-ensemble-smoother. Thanks to jamesYu  for converting 
the iES algorithm from the MATLAB implementation to the PYTHON implementation.
![inv_vis](https://user-images.githubusercontent.com/65839033/203459400-d916e239-8602-40c9-8bcb-cf45199647bb.png)

# Disclaimer
This depository is made available and contributed to under the license that include terms that, for the protection of the contributors, make clear that the depository is offered “as-is”, without warranty, and disclaiming liability for damages resulting from using the depository. This guide is no different. The open content license it is offered under includes such terms.

The code may include mistakes, and can’t address every situation. If there is any question, we encourage you to do your own research, discuss with your community or contact us.

All views expressed here are from the contributors, and do not represent the opinions of any entities with which the contributors have been, are now or will be affiliated.
