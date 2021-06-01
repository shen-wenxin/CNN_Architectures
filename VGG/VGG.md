## VGG

​	VGG在2014年有牛津大学著名研究组VGG(Visual Geometry Group)提出，斩获该年ImageNet竞赛中Localization Task（定位任务）第一名和Classification Task（分类任务）第二名。

### 网络亮点

##### 小卷积核和多卷积子层

​	通过堆叠多个$$3\times 3$$的卷积核来替代大尺度卷积核(减少所需参数)。

​	论文中提到，可以通过堆叠两个$$3\times 3$$的卷积核来替代$$5\times 5$$的卷积核，堆叠三个$$3\times 3$$的卷积核来替代$$7\times 7$$的卷积核。他们拥有相同的感受野。

##### 小池化核

​	相比AlexNet的3x3的池化核，VGG全部采用2x2的池化核。

##### 通道数多

​	VGG网络第一层的通道数为64，后面每层都进行了翻倍，最多到512个通道，通道数的增加，使得更多的信息可以被提取出来。

##### **层数更深、特征图更宽**

​	由于卷积核专注于扩大通道数、池化专注于缩小宽和高，使得模型架构上更深更宽的同时，控制了计算量的增加规模。

### 概念拓展

#### CNN感受野

​	在卷积神经网络中，决定某一层输出结果中一个元素所对应的输入层的区域大小，被称作感受野。通俗的解释是，输出$$feature\  map$$上的一个单元对应输入层上的区域大小。

<img src="https://gitee.com/shen_wenxin0510/readme-pictures/raw/master/cv/image-20210524022542458.png" alt="image-20210524022542458" style="zoom:50%;" />

​						图1(没排号)

​	结合下图进行分析，在最下面$$9\times9\times 1$$的网络，通过一个$Conv1$。根据计算公式$out_{size} = (in_{size} - F_{size} + 2P)/S + 1$，计算可得到$$4\times 4\times 1$$大小的网络。将这$4\times4\times 1$的网络通过$MaxPool1$，可得到一个大小为$2\times2\times1$的网络。

<img src="https://gitee.com/shen_wenxin0510/readme-pictures/raw/master/cv/image-20210524143021258.png" style="zoom: 25%;" />

​	所以，对于图一中，第一层的一个单元，对应第二层的一个感受野就是$2\times2$的区域大小，对应第三层的一个感受野大小就是$5\times5$的区域大小。

##### 感受野计算公式

​	$F(i) = (F(i + 1) - 1)\times Stride + Ksize$

​	参数解释：$F(i)$为第$i$层感受野，$Stride$为第$i$层的步距，$Ksize$为卷积核或池化核尺寸。

​	以图1为例：

​	$Feature\  map:F = 1$

​	$Pool1:	F = (1 - 1)\times 2 + 2 = 2$

​	$Conv1:	F = (2-1)\times2+3=5$

​	假设堆叠3个$3\times 3$的卷积核：

​	$Feature\ map:F = 1$

​	$Conv3\times3(3):\ F = (1-1)\times 1 + 3 = 3$

​	$Conv3\times 2(2):\ F = (3 - 1)\times 1 + 3 = 5$

​	$Conv3\times3(1):\ F = (5 - 1)\times 1 + 3 = 7$

​	以上式子表明，$Feature\ map$上的一个单位，在原图上的感受野是一个$7\times 7$的大小。

​	论文中提到:可以通过**堆叠两个3$\times$3的卷积核来替代5$\times$5的卷积核，堆叠三个3$\times$3的卷积核来替代7$\times$7的卷积核**.

​	通过堆叠替代，可以大大减少所需的参数。使用$7\times7$卷积核所需的参数为$7\times7\times C\times C = 49C^2$,堆叠三个$3\times 3$卷积核所需参数为$3\times (3\times3\times C\times C) = 27C^2$。（假设输入输出的channel为C）

### 网络分析

​	其网络结构如图所示，实际中多用D这一类网络：

<img src="https://gitee.com/shen_wenxin0510/readme-pictures/raw/master/cv/image-20210524004952259.png" style="zoom:50%;" />

​	参数补充：$conv$的$stride$为1,$padding$为1。

​			$maxpool$的$size$为2,$stride$为2。

​	以下为一个图片处理的例子：

<img src="https://gitee.com/shen_wenxin0510/readme-pictures/raw/master/cv/image-20210524151820913.png" style="zoom:50%;" />

```python
class VGG(nn.Module):
    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
```

