UNet for segmenting salt deposits from seismic images with PyTorch.

## General

We, [tugstugi](https://github.com/tugstugi) and [xuyuan](https://github.com/xuyuan), have participated
in the Kaggle competition [TGS Salt Identification Challenge](https://www.kaggle.com/c/tgs-salt-identification-challenge)
and reached the 9-th place. This repository contains a simplified and cleaned up version of our team's code partially based
on the ideas of [Heng Cherkeng's discussion](https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65933)
on the Kaggle discussion board.

We have used a single UNet model with a SENet154 encoder which has a single fold score of 0.882.
With 10 folds using reflective padding and another 10 folds with resizing, we got 0.890.
The final private LB score 0.892 was achieved by post processing on the model's output.


## Features

* single [UNet](https://arxiv.org/abs/1505.04597) model with a [Squeeze-and-Excitation network](https://arxiv.org/abs/1709.01507) encoder
* no ensembling, no pseudo labeling
* [object context](https://arxiv.org/abs/1809.00916) in the decoders and in the base
* symmetric extension of the [Lovasz hinge loss](https://arxiv.org/abs/1705.08790) function (+0.02 private LB improvement):
```python
def symmetric_lovasz(outputs, targets):
    return (lovasz_hinge(outputs, targets) + lovasz_hinge(-outputs, 1 - targets)) / 2
```

## Training
1. Download and extract the [dataset](https://www.kaggle.com/c/tgs-salt-identification-challenge/data)
    * copy `train.csv` into `datasets/`
    * copy train images and masks into `datasets/train/`
    * copy test images into `datasets/test/`
2. Train SENet154-Unet for 250 epochs on 2x P100: `python train.py --vtf --pretrained imagenet --loss-on-center --batch-size 32 --optim adamw --learning-rate 5e-4 --lr-scheduler noam --basenet senet154 --max-epochs 250 --data-fold fold0 --log-dir runs/fold0 --resume runs/fold0/checkpoints/last-checkpoint-fold0.pth`
    * tensorboard logs, checkpoints and models are saved under `runs/`
    * start tensorboard with `tensorboard --logdir runs`
    * training log of a LB0.883 model is provided under `runs/lb0.883_fold0/`
3. Do [SWA](https://arxiv.org/abs/1803.05407) on the best loss, accuracy and kaggle metrics models: `python swa.py --input runs/fold0/models --output fold0_swa.pth`
4. Create a Kaggle submission: `python test.py --tta fold0_swa.pth --output-prefix fold0`
    * a submission file `fold0-submission.csv` should be created now


