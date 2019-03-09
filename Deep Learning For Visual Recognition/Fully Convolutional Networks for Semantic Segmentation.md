# Fully Convolutional Networks for Semantic Segmentation

*Long J, Shelhamer E, Darrell T. Fully convolutional networks for semantic segmentation[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2015: 3431-3440.*

> 参考 https://zhuanlan.zhihu.com/p/22976342

## Abstract
Key insight: fully convolutional networks that take input of arbitrary size and produce correspondingly-sized output with efficient inference
and learning 全卷积网络

## 7. Conclusion
Fully convolutional networks are a rich class of models that address many **pixelwise** tasks. 精确到像素点的识别

FCNs for semantic segmentation dramatically improve accuracy by transferring pre-trained classifier weights, fusing different layer representations, and learning end-to-end on whole images. 

End-to-end, pixel-to-pixel operation simultaneously simplifies and
speeds up learning and inference.

## 1. Introduction
提出问题：Make a prediction at every pixel

Contribution: the first work to train FCNs ene-to-end (1) for pixelwise prediction and (2) from supervised pre-training

Semantic segmentation是在semantics和location之间权衡：
global information - *what*; 
local information - *where*

Cast pre-trained networks into FCN and augment with a skip architecture -> local-to-global pyramid

## 3. Fully Convolutional Networks
3.2和3.3讲的是经过了3.1之后，生成了coarse output，我们应该怎么用这种coarse output得到更精确的pixelwise的结果。

### 3.1 Adapting classifiers for dense prediction
传统的网络take fixed-sized inputs and produce non-spatial outputs。它们的全连接层往往有固定的维度并且会丢失空间坐标。但是事实上，我们也可以把全连接层换成卷积层，这样的话，网络就可以接受任意大小的input然后生成spatial output maps。

另一个优点是the computation is highly amortized over the overlapping regions of those patches. (??)

适合dense problems。

网络会通过subsampling来reduce output dimensions.

### 3.2 Shift-and-stitch is filter dilation
Dense predictions can be obtained from coarse outputs by stitching together outputs from shifted versions of the input.

### 3.3 Upsampling is (fractionally strided) convolution
Another way to connect coarse outputs to dense pixels is interpolation.

### 3.4 Patchwise training is loss sampling

## 4. Segmentation Architecture