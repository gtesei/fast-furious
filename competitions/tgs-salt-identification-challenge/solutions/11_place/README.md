# 11th place solution writeup [0.891518 vs. winner 0.896469]

@alexisrozhkov's pipeline

Model: UNet-like architecture

Backbone: SE ResNeXt-50, pretrained on ImageNet

Decoder features (inspired by Heng’s helpful posts and discussions):

Spatial and Channel Squeeze Excitation gating
Hypercolumns
Deep supervision (zero/nonzero mask)
Tile size adaptation:

Pad 101 -> 128
Mode - replicate (reflect was worse, reflect horizontally+replicate vertically was same, random selection as an augmentation didn’t improve as well, although more tests needed to be made)
Augmentations:

Random invert (unexpected, but visually it doesn’t change input dramatically. Also tried mean subtraction and derivative instead of raw input, but these didn’t work well)
Random cutout (even more unexpected, but some papers indicate that it’s helpful for segmentation, because implicitly it causes model to learn “inpainting” input tiles)
Random gamma-correction (makes sense, since tiles seem to be postprocessed in a way that changes brightness)
Random fixed-size crop
Random horizontal flip
Optimizer (inspired by Peter’s writeups):

SGD: momentum 0.9, weight decay 0.0001
Batch size: 16
Starting LR determined using procedure similar to LR find from fast.ai course - 5e-2
LR schedule - cosine annealing from maximum LR, cycle length - 50 epochs, 10 cycles per experiment
Best snapshots according to metric were saved independently for each cycle, final solution uses 2 best cycles per fold
Loss:

Since I used deep supervision it’s a combination of 3 losses - classification loss, pure segmentation loss (using output from an earlier layer, empty samples excluded), total loss (segmentation loss over all samples)
Classification loss - BCE
Segmentation losses - BCE + Lavasz Hinge*0.5
Segmentation loss was evaluated after cropping masks and predictions back to 101 - this seems to “fix” cases where a few-pixel corner mask was predicted on a padded mask with imperfect alignment, but was cropped afterwards, resulting in 0 metric per sample.
Cross-validation

5-fold random split
Tried depth, area and “type”-based stratification, but it seemed to degrade result
Didn’t have enough time to synchronize folds with Ivan to allow cross-validation of ensemble results
@sawseen’s pipeline

Model: Modified Unet

Backbone: SE-ResNeXt-50, pretrained on ImageNet

Decoder features:

Dilated convolutions with dilation from 1 to 5
Hypercolumns
ASP_OC_Module before last Convolution layer
Deep supervision (zero/nonzero mask, nonzero mask segmentation)
Dropout
Tile size adaptation

Replication up to 128
Augmentations:

Random Rotation up to 10 degree
Random Crop and Scale
Random Horizontal flip
Optimizer:

SGD: momentum = 0.9, weight decay = 0.0001
Batch size: 16
Lr schedule. Pretrain for 32 epochs with lr = 0.01. Then SGDR was applied for 4 cycles with cosine annealing: lr from 0.01 to 0.0001. Each cycle lasts for 64 epochs.
Loss:

Segmentation loss
Pretrain: 0.1 * BCE loss + 0.9 * lovasz loss (Elu + 1)
SGDR cycles: 0.9 * BCE loss + 0.1 * lovasz loss (Elu + 1)
Optimization loss = 0.05 * BCE(zero/nonzero mask classification) + 4 * 0.12 * segmentation loss per deep supervised nonzero masks + 1.0 * result segmentation loss
Cross-validation:

5 fold without stratification. For each fold 2 best snapshots were chosen for result predictions
Ensembling

Each of us trained multiple models (5 folds x 2 best cycles), which were used to predict masks. These 10 predictions were averaged per fold, and then a threshold was applied. Ivan chose a more conservative value of 0.5 for each fold, Alexey was more adventurous and selected a threshold that was resulting in a best metric - even though it might lead to overfitting to validation set. Then ranges were aligned and predictions were averaged again; resulting predictions were thresholded and used for submission.

Postprocessing

Few days before competition deadline we’ve decided to arrange predictions according to publicly shared csv for mosaics. This was the “eureka” moment of the competition - ground truth annotation immediately became clear, and we started to look for ways to “help” the network solve the task better, taking the annotation specifics into account.

Unlike some other teams, we think that ground truth annotation made sense. It seems that the way the data was acquired makes it hard to get reasonable measurements underneath the salt domes, and also it seems that salt only forms a cap and doesn’t take the whole volume under this cap (I hope that people with domain expertise will correct if that’s a wrong assumption). In this case it wasn’t physically-sound to label block underneath salt as salt (because probably they were not). Next question is why to label the salt dome thickness this particular way? Our hypothesis is that it’s hard to make this prediction given tiles without boundary, so they were marked as “not containing salt”, and the rest of tiles were marked properly. Probably the tile size was chosen to match the expected thickness of the salt dome, but it’s just a guess.

Surprisingly, it turned out that our networks were already doing a very reasonable job in most cases and were just lacking enough context - a “bird’s eye view” should’ve been able to fix that. 2 days before the deadline we realized a way to do this - pass the tile ids along the pipeline and then apply post-processing heuristics using tile neighbor information to improve consistency of result, even though it was a bit risky (different depth values for neighboring samples probably imply that tiles were taken from different “slices” of a scan volume, so not necessarily they had aligned ground truth values).

We have used both ground truth and predicted masks to construct few rules to improve the segmentation results:

For almost-vertical ground truth tiles (the ones with last 30 rows same, but not fully covered) - extrapolate last row downwards till the end of mosaic
For “caps” in ground truth tiles (the ones with fully-covered last row) - zero all tiles below
For tiles vertically “sandwiched” between predicted or ground truth masks - replace mask with last row of tile above, if it will cause increase of mask area (ideally it should’ve been a trapezoid from last row of tile above to first row of tile below, but didn’t have time to do that, and potential score gain is small)
For tiles below predicted “caps” - do the same as for ground truth case
For tiles below almost-vertical predicted tiles - extract downward.
It is interesting that although these heuristics improved public lb score significantly, the effect on private lb score was negligible (~0.1 increase vs 0.001). Some rules didn’t make a difference on public lb, but deteriorated result on private lb significantly (last described rule caused 0.003 drop on private lb).

Another observation - seems that for majority of teams private lb score is higher than public nonetheless. It leads us to hypothesize that public/private split wasn’t random and was cherry-picked to meet organizers’ demands in a better way. One hypothesis for why private lb score was usually higher than public - organizers excluded few-pixel masks (which were hard to predict, incurred high penalty and probably weren’t important from business standpoint) from private set.

Anyway - thanks to organizers for interesting contest with a “plot twist”, to Heng, Peter and other people on Kaggle forum and ods.ai community that sparked enlightening discussions, and to all participants for making it a very competitive experience!


