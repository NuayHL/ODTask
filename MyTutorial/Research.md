## **Terminology** 
### **FFN**: Feed Forward Network
- 前馈神经网络。最简单的神经网络模型，一般由多层感知机（perceptron）组成。
### **FCN**: Fully Conventional Network
- 全卷积神经网络。所有层均为卷积层，不存在池化和全连接。
### **BPR**: Best Possible Recall
- 可能的最好召回率

## **Focal Loss**
来源于17年的文章，主要是为了平衡正负样本在训练时候的分布。
什么是正负样本

## **需要做到**
    
    定义网络
    定义数据
    定义损失函数和优化器
    可视化各种指标 
    计算在验证集上的指标

## **mAP**
### Average Prcision


## **Label Assignment**
Label Assignment 指的是目标检测算法在训练阶段对生成的feature map上的不同位置赋予合适的Gt或者其他学习目标的过程。
最主要的体现在于根据Label Assignment的选择计算每一次训练的Loss。具体而言，就是为算法预测出的每一个目标选择与之匹配的GT。

Label Assign是训练过程的一个关键步骤，和目标函数一起决定了detector的学习目标，进而决定了detector的好坏。

这也可以看做是一个映射，将每张图的GT及其背景信息通过函数转换为这个网络应该输出的feature map

**评价指标：减少歧义**

**一般Label Assignment分为两种：**
- one-to-many
- one-to-one

使用哪一种Assign决定了这个检测算法在inference（推理）的时候是否使用NMS

### - Retina Net(Faster-Rcnn)
    直接根据GT与所有Anchor的IOU选出最合适的匹配。所有Anchor指的是所有scale下的Anchor，因此这个阶段也自动把GT分配给了不同scale的Anchor

    "Since anchor boxes and ground-boxes are associated by their IoU scores, this enables different FPN feature levels to handle different scale objects." -- FCOS

### - FCOS
    使用anchor point
    对于ambiguous sample，选择最小的bounding box与之对应
    只有不是背景的point才参与计算回归框的损失
    
### - ATSS
    动态threshold。
    每层按距离选取前k个sample，全部计算IOU的mean和std。计算threshold=mean+std


## **Dense & Sparse Detector**
## **Data Argumentation**
### 1. Color Jitter
### 2. Horizontal Filp
### 3. 

## **ROI Pooling & ROI Align**
对ROI快速池化，或者更加精细的池化（例如双线性差值避免损失数据）
Old fashioned method used in Fast-Rcnn series.
