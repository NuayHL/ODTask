# Hard Work!

其实写给自己看的每日进度。。。不要摸鱼！
### 17th Mar.
git总是出问题
### 18th Mar.
git Ok了。开始搭Darknet53
### 19th Mar.
查到了具体的网络结构然后复现了Darknet53，需要进一步测试。

终于有点进展了。
### 20th Mar.
darknet53没有问题. 需要把detector和neck结合在一起.

### 22nd Apr.
回到工作，目前的任务是Label Assignment

### 23rd Apr.
I need to prepare for the presentation next week. Only then I will continue the coding.

Two things need to be settled:
1. label assignment module
2. data processing pipeline

### 28rd Apr.
Things becomes urgent.

Maybe today can finish a primitive datatrans function.

COCOapi, COCOdataset format

拉倒吧，还是用中文纪录一下。今天把coco数据集的格式搞明白了，并且完成了odgt到coco类型annotation的转化。

明天要干三件事情
1. 用cv2写一个可视化，能显示gt的bbox
2. 完成crowdhuman的pytorch dataset构建
3. 新建一个文档用于纪录我的开发细节（其实是怕忘记

### 29rd Apr.
新建了一个MyTutorial存放我的一些开发经过，随项目上传github。

关于就是dataset的输出是否需要转化为tensor，目前是np.ndarray的样式，且img不是model的输入样式。

今天的进度完成。

yolo是有检测头的，明天先完成对yolo的注释，然后开始Result class的设计，这一部分参考下其他项目的
代码。然后在此基础上完成Label Assign的模块。之后还有Loss的编写，然后应该就可以开始训练了。哦对了
还有nms的问题。

估计还要一个星期的时间。

### 30rd Apr.
我觉得未来应该要用yaml文件导入模型参数，需要看看这是怎么写的

似乎现在还用不到result class可以直接写出来Label Assign。比较好奇的是现在我需不需要用cuda写一些算子。

今天一天都是在忙ubuntu的环境，不得不说实验室有线的校园网真是太离谱了。我至今没遇到过这么无语的环境。

明天的任务主要是把training这一块的labelassign解决，先写一个RetinaNet用的最简单的吧。然后就可以开始训练了。

### 1st May.
Ubuntu的环境终于配置好了。太稳定了。

开始写iou

### 2nd May.
欲速则不达。在编写label assignment和iou的时候在代码上如何实现还是个问题。

要保证模型速度和torch的bp。这几天先看看torch中loss这个类。

今天主要是发现了这个问题，先看了一下FCOS中的assign是怎么写的。

### 7rd May.
我决定还是先自己写一个初步的好了

### 9th May.
今天感觉还是有所突破的，仔细阅读了RetinaNet中关于Anchor的代码。那么今天回去可以
看看写出来初步的Anchor的代码。

这个人写的很好 https://www.zhihu.com/column/c_1249719688055193600

后续就是还原最简单的Label Assignl了。之后加上loss就可以去初步训练了。

对了，还有图片的预处理阶段。主要这周有两个大作业，估计正式训练还是要等一周的样子。