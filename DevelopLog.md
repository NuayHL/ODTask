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

### 12nd May.
今天把最简单的assign写完了，anchor和iou都测试无误。有个问题对于gtbbox没能分配到
anchor的话现在我还没写解决方案。

明天看看能不能把loss写了。之后就可以正式开始第一次的训练。 

后续还有两件事情，一个是是nms，另一个是评价方案。

### 25th May.
中间有一周没有记录，上周我在继续写loss，这周终于loss正常了，而且assign保证了每个annotation bbox
能有至少一个anchor匹配。

loss上周遇到的问题在于输出的值异常和loss的反向传播出错。错误的原因在于将需要求导的tensor提前改变了数据 。
可以用torch.clone()解决，这个以后写的时候也要小心。

目前就是编写模型的输出相关模块，一个是从结果解码bbox，然后输入nms；另一个是模型的评价系统，利用coco的api
来实践一下。

### 31st May.
终于，莫名其妙的loss震荡解决了。原因在于我在对annotation的bbox进行缩放的时候没有深拷贝
导致bbox被修改。每遍历一次dataset，也就是一个epoch，bbox都会漂移，导致训练的loss逐渐增大

目前我的网络的backbone有resnet和darknet，detector采用yolov3的。

目前等这个训练完毕后编写关于训练评估的代码。完成coco数据集编写。然后复现baseline，这一步可以直接用别人的代码train。
最后就是在此基础上编写关于正负样本划分的问题。

### 4nd Jun.
第一次70epoch的训练跑完了，有以下几个问题。

第一是感觉训练时间有些长，不知道是不是分辨率的问题，也可能是代码的问题。

第二是效果感觉不是很好，似乎没能正确的预测。

未来有以下几个方向
1. 利用别人的网络跑一次训练
2. 学习使用python装饰器和argparse的用法。
3. 编写coco载入的数据集。
4. 构建从p3到p7的金字塔型网络。

### 6nd Jun.
摸鱼两天后居然一下从loss那里看出了问题，是BCEloss写反了我日你妈。顺带找出来好几个很离谱的
错误，bbox regression应该为smooth l1 loss我写成了l2导致不好收敛， 二是对clsloss平均的时候
除以了全部样本（不包含忽略）导致clsloss权重降低，我想这也是为什么BCE写错了也能勉勉强强收敛的原因。

今天的工作主要还有重新写了dataset那里，将dataset的transform与dataset分离操作。这样提高代码的
封装性。而且在这里我又发现一个错误，我定义config的时候将width和height写反了，直接导致anchorbox
在w和h的数量上是反的，这样对图片覆盖不全，这一部分也改了，对cv2的坑又一步摸熟悉了tmd。

回顾retinanet的源代码，对于tensor的操作确实惊为天人，还是要多学习个。

trian的差不多之后稍微看下结果，编写一下评价代码以及了解下tensorbroad，多读几篇论文，然后再决定
下一步做什么。

不着急不着急。

### 7nd Jun.
其实使用DDP的时候一直还是有问题的，1卡的显存会被0卡的进程占用2g到3g。不知道是为什么

*17nd 补充* 在程序的其他地方is_initialized()是False,所以不会传入对应的GPU，原因未知。已经通过传递device解决 
### 17nd Jun.
第一阶段结束了。
根据FCOS的数据，目前的模型在CrowdHuman val中的表现仅超过了yolov2。

有两篇很重要的论文需要读一下。

需要做的事情：
1. 对测试时候bbox的type写一个修改函数
2. 阅读论文写出新的neck/detector/assignment/nms/loss (...)
3. 增加不同的数据集，需要创建不同的读取方式
4. 在训练的时候做一些可视化，并随着训练进行，增加测试的环节，并且记录到log里面。

### 21st Jun.
关于评价里的小目标在cocotool里面是怎么算的，这是一个问题。先去查cocodataset的网站看看

### 25th Jun.
1. 重新改写annotation文件中的area，因为评价指标是按照area的像素多少算的，但是输入的图片大小不一，这导致比例
不一样。
2. 把fpn改到5层去训练
3. 把nms改成softnms，尝试编写新的loss函数
4. anchor也许要改改？

### 29th Jun.
要改的东西还挺多的。

1. 修改dataset，加入ignore的功能
2. 进而修改assignment，在训练的时候不对ignore区域赋负样本。
3. 采用新的assignment的方法，例如ota，free anchor
4. 进而编写loss，在retinanet的基础上对crowdhuman进行训练。
5. 最后一步，挑选nms进行最后的组合，例如Adaptive NMS

### 6th July.
之前写的RetinaNet似乎是有些问题
1. 看yolov6的argumentation并应用
2. 编写softnms
3. 尝试对ECPdataset编写cocostyle annotation
4. 学习mmdetection做大规模的训练
5. 学习attention的机制

### 8th July.
softnms 倒是不急，我需要制定策略后再决定。

当务之急，是先能在自己的平台上复现接近sota的结果。？凌晨胡话（

### 14th July.
组会结束了，目前的想法是将这个项目重构一遍。

主要的问题在于