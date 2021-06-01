## GoogLeNet

​	$GoogLeNet$在2014年由$Google$团队提出。斩获当年$ImageNet$竞赛中$Classification\ Task$(分类任务)第一名。

### 网络亮点

1. ​	引入了$Inception$结构(融合不同尺度的特征信息)
2. ​	使用$1\times 1$的卷积核进行降维以及映射处理
3. ​	添加两个辅助分类器帮助训练(AlexNet和VGG都只有一个输出层，GoogLeNet有三个输出层)
4. ​	丢弃全连接层，使用平均池化层(大大减少模型参数)。

### 网络结构

![img](https://img-blog.csdn.net/20160225155414702)

<img src="https://gitee.com/shen_wenxin0510/readme-pictures/raw/master/cv/image-20210526100023140.png" style="zoom:67%;" />

1. GoogLeNet采用了模块化的结构（Inception结构）
2. 网络最后采用了average pooling（平均池化）来代替全连接层，该想法来自NIN（Network in Network），事实证明这样可以将准确率提高0.6%。
3. 虽然移除了全连接，但是网络中依然使用了Dropout ;
4. 为了避免梯度消失，网络额外增加了2个辅助的softmax用于向前传导梯度（辅助分类器）这里的辅助分类器只是在训练时使用，在正常预测时会被去掉。辅助分类器促进了更稳定的学习和更好的收敛，往往在接近训练结束时，辅助分支网络开始超越没有任何分支的网络的准确性，达到了更高的水平。

#### Inception结构

##### 引入原因

​	一昧的增加网络的深度和宽度会带来很多问题：

1. 参数太多，如果训练数据集有限，很容易产生过拟合；

2. 网络越大、参数越多，计算复杂度越大，难以应用；

3. 网络越深，容易出现梯度弥散问题（梯度越往后穿越容易消失），难以优化模型。

   提出Inception方法。

​    Inception就是把多个卷积或池化操作，放在一起组装成一个网络模块，设计神经网络时以模块为单位去组装整个网络结构。在未使用这种方式的网络里，我们一层往往只使用一种操作，比如卷积或者池化，而且卷积操作的卷积核尺寸也是固定大小的。但是，在实际情况下，在不同尺度的图片里，需要不同大小的卷积核，这样才能使性能最好，或者或，对于同一张图片，不同尺寸的卷积核的表现效果是不一样的，因为他们的感受野不同。所以，我们希望让网络自己去选择，Inception便能够满足这样的需求，一个Inception模块中并列提供多种卷积核的操作，网络在训练的过程中通过调节参数自己去选择使用，同时，由于网络中都需要池化操作，所以此处也把池化层并列加入网络中。



![image-20210526100305548](https://gitee.com/shen_wenxin0510/readme-pictures/raw/master/cv/image-20210526100305548.png)

​	流程：将$previous \ layer$输出的特征矩阵同时输入四个分支，得到结果之后按深度进行拼接。

​	(b)中引入了$1\times 1\ convolutions$降维的功能。减少深度，从而减少卷积参数，从而减少计算量。 

#### 辅助分类器

<img src="https://gitee.com/shen_wenxin0510/readme-pictures/raw/master/cv/image-20210526105812226.png" style="zoom:50%;" />

第一层是平均池化下采样层($AveragePool$  $filter\ size:5\times 5 ,\ stride:\ 3$)。该层连在$Inception-4a(output\_size:14\times 14\times 512)$/$Inception-4d(output\_size:14\times 14\times 528)$之后。经过$AveragePool$之后，输出结果为:$4\times 4\times 512/ 4\times 4\times 518$

 第二层是卷积层($1\times 1\ convolution\ with\  128\ filters$)用来降低维度。同时采用了$ReLU$激活函数。

第三层是节点个数为1024的全连接层。同样也采用ReLU激活函数。

在全连接层和全连接层之间采用$Dropout\ $按照70%的比例随机失活神经元。

第四层：输出层，输出节点个数对应类别个数。

之后输出通过softmax，得到概率分布。



### 