# Rich feature hierarchies for accurate object detection and semantic segmentation

*Girshick R, Donahue J, Darrell T, et al. Rich feature hierarchies for accurate object detection and semantic segmentation[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2014: 580-587.*

## Key contents
### Two key insights
1. 使用CNN来localize和segment objects：one can apply high-capacity convolutional neural networks (CNNs) to bottom-up region proposals in order to localize and segment objects.
2. 提出一个方法解决labeled data较少的问题：when labeled training data is scarce, supervised pre-training for an auxiliary task, followed by domain-specific fine-tuning, yields a significant performance boost. 首先针对一个有很多labeled traning data的辅助任务进行pre-training，然后再根据目标任务进行调整。

## Following the paper structure
### Abstract
A simple and scalable object detection algorithm.<br>
Two key insights.

### 6. Conclusion
重提两个insights.<br>
Using a combination of classifical tools from computer vision (bottom-up region proposals) and deep learning (CNN).

### 1. Introduction
hierarchical, multi-stage processes for computing features<br>
bridging the gap between image classification and object detection<br>
解决localizing objects within an image的问题：operating within the "recognition using regions" paradigm.<br>
(Figure 1)系统概述:(1)input image(2)提取近2000个bottom-up region proposals(3)针对每一个proposal使用CNN计算features(4)使用class-specific linear SVM为每个region进行分类。<br>
解决labeled data不够训练CNN的问题：supervised pre-training on a large auxiliary dataset (ILSVRC), followed by domain specific fine-tuning on a small dataset (PASCAL).<br>
Efficient<br>

### 2. Object detection with R-CNN
3 modules: (1)category-independent region proposals，定义了可用的candidate detections(2)从每个region中提取一个固定长度的feature vector的large CNN(3)a set of class-specific linear SVMs.
#### 2.1 Module design
**Region proposals**: While R-CNN is agnostic to the particular region proposal method, we use selective search to enable a controlled comparison with prior detection work. 虽然R-CNN与特定区域提议方法无关，但我们使用选择性搜索来实现与先前检测工作的受控比较。(?)
**Feature extraction**: 4096-dimensional feature vector. forward propagating a mean-subtracted RGB image through five con $227 \times 227$
convolutional layers and two fully connected layers.<br>
需要把图像转为标准的$227 \times 227$：dilate the tight bounding box so that at the warped size there are exactly $p$ pixels of warped image context around the original box (we use $p = 16$) 然后再warp all pixels in a tight bounding box around it to the required size.
#### 2.2 Test-time detection
#### 2.3 Training

### 4. The ILSVRC2013 detection dataset
这个数据集没有PASCAL VOC那么homogeneous，所以这里需要介绍一下choices about how to use it.
#### 4.1 Dataset overview
train，val，test三个数据集。200 classes。<br>
val和test中，所有的图里所有的instance都有label。train中，有一些没有label。<br>
由此产生一些问题：哪里找negative examples？train images的数据与val和test不同，train应该用到什么程度？


