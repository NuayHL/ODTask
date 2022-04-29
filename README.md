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

yolo是有检测头的，明天先完成对yolo的注释，然后开始Result class的设计，这一部分参考下其他项目的
代码。然后在此基础上完成Label Assign的模块。之后还有Loss的编写，然后应该就可以开始训练了。

估计还要一个星期的时间。