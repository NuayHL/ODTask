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