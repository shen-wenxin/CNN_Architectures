## AlexNet 

### 意义

1. AlexNet首次在大规模图像数据集实现了深层卷积神经网络结构。
2. 其在ImageNet LSVRC-2012目标识别的top-5 error为15.3%，同期第二名仅为26.2%。
3. 得益于GPU计算性能的提升，以及大规模数据集的出现，之后每年的ImageNet LSVRC挑战赛都被深度学习模型霸榜。

### 创新点

1. 训练出当前最大规模的卷积神经网络，此前的LeNet-5网络仅为3个卷积层和一个全连接层。

2. 实现高效的GPU卷积运算结构，也使得此后GPU成为深度学习的主要工具。

3. 使用了ReLU激活函数，而不是传统的Sigmoid激活函数以及Tanh激活函数。（Sigmoid函数的缺点：在求导过程中求导比较麻烦，当训练网络比较深的时候会出现梯度消失的现象。)

4. 使用了LRN局部响应归一化

5. 在全连接层的前两层中使用了Dropout 随即失活神经元操作，以减少过拟合。

   

### 网络分析

#### 术语解释

##### 1 softmax

​	假设有一个数组，$$V_i$$表示$$V$$中的第$$i$$个元素，那么这个元素的$$softmax$$值就是：

​				$$S_i = \frac{e^{V_i}}{\sum_je^V_j}$$

##### 2 计算公式

​	经卷积后的矩阵尺寸大小的计算公式为：

​				$$N = (W - F + 2P)/S + 1$$

1. ​	输入图片大小$$W\times W$$
2. ​	$$Filter$$大小为$$F\times F$$
3. ​	步长$$S$$
4. ​	$$padding$$的像素数$$P$$	

#### 网络结构



![image-20210519104301337](https://gitee.com/shen_wenxin0510/readme-pictures/raw/master/image-20210519104301337.png)

​	网络包含8个带权重的层。

​	前5层是卷积层，剩下3层是全连接层。最后一层全连接层的输出是1000维$$softmax$$的输入，$$sofrnax$$会产生1000类标签的分布。

##### Conv1

<img src="https://gitee.com/shen_wenxin0510/readme-pictures/raw/master/image-20210522171714138.png" style="zoom:50%;" />

​	分析图跟论文得到conv1的信息如下：

​		$$48*2 = 96$$个卷积核

​		$$kernel\_size:$$ $$11 $$

​		$$padding:[1,2]$$(意思是在特征矩阵的左边加一列0，右边加两列0，在上面加一列0，下面加两列0)

​		$$stride:4$$

​		$$input\_size:[224,224,3]$$

​		$$N = \frac{(W - F + 2P)}{S} + 1=\frac{224-11+(1+2)}{4} + 1 = 55$$

​		$output\_size:[55,55,96]$

##### Maxpool1

<img src="https://gitee.com/shen_wenxin0510/readme-pictures/raw/master/image-20210522172147994.png" style="zoom:50%;" />

​	经过查阅资料：

​		$$kernel\_size:3$$

​		$$padding:0$$

​		$stride:2$

​		$$input\_size:[55,55,96]$$

​		$$N = \frac{(W - F + 2P)}{S} + 1=\frac{55-3}{2} + 1 = 27$$

​		$output\_size:[27,27,96]$

$$Conv2$$

<img src="https://gitee.com/shen_wenxin0510/readme-pictures/raw/master/image-20210522172650952.png" style="zoom:50%;" />

​		$$128*2 = 256$$个卷积核

​		$$kernel\_size:$$ $$5 $$

​		$$padding:[2,2]$$

​		$$stride:1$$

​		$$input\_size:[27,27,96]$$

​		$$N = \frac{(W - F + 2P)}{S} + 1=\frac{27-5+4}{1} + 1 = 27$$

​		$output\_size:[27,27,256]$

##### Maxpool2

<img src="https://gitee.com/shen_wenxin0510/readme-pictures/raw/master/image-20210522172952060.png" style="zoom:50%;" />

​	经过查阅资料：

​		$$kernel\_size:3$$

​		$$padding:0$$

​		$stride:2$

​		$$input\_size:[27,27,256]$$

​		$$N = \frac{(W - F + 2P)}{S} + 1=\frac{27-3}{2} + 1 = 13$$

​		$output\_size:[13,13,256]$

##### Conv3

<img src="https://gitee.com/shen_wenxin0510/readme-pictures/raw/master/image-20210522173201592.png" style="zoom:50%;" />

​		$$192*2 = 384$$个卷积核

​		$$kernel\_size:$$ $$3 $$

​		$$padding:[1,1]$$

​		$$stride:1$$

​		$$input\_size:[13,13,256]$$

​		$$N = \frac{(W - F + 2P)}{S} + 1=\frac{13-3+2}{1} + 1 = 13$$

​		$output\_size:[13,13,384]$

##### Conv4

<img src="https://gitee.com/shen_wenxin0510/readme-pictures/raw/master/image-20210522173450183.png" style="zoom:50%;" />

​		$$192*2 = 384$$个卷积核

​		$$kernel\_size:$$ $$3 $$

​		$$padding:[1,1]$$

​		$$stride:1$$

​		$$input\_size:[13,13,384]$$

​		$$N = \frac{(W - F + 2P)}{S} + 1=\frac{13-3+2}{1} + 1 = 13$$

​		$output\_size:[13,13,384]$

##### Conv5

<img src="https://gitee.com/shen_wenxin0510/readme-pictures/raw/master/image-20210522191800596.png" style="zoom:50%;" />

​		$$128*2 = 256$$个卷积核

​		$$kernel\_size:$$ $$3 $$

​		$$padding:[1,1]$$

​		$$stride:1$$

​		$$input\_size:[13,13,384]$$

​		$$N = \frac{(W - F + 2P)}{S} + 1=\frac{13-3+2}{1} + 1 = 13$$

​		$output\_size:[13,13,256]$

##### Maxpool3

<img src="https://gitee.com/shen_wenxin0510/readme-pictures/raw/master/cv/image-20210519114803518.png" style="zoom:50%;" />

​	经过查阅资料：

​		$$kernel\_size:3$$

​		$$padding:0$$

​		$stride:2$

​		$$input\_size:[13,13,256]$$

​		$$N = \frac{(W - F + 2P)}{S} + 1=\frac{27-3}{2} + 1 = 13$$

​		$output\_size:[6,6,256]$

#### 网络创新点分析

##### Dropout

​	$$AlexNet$$使用$$Dropout$$的方式在网络的正向传播过程中随机失活一部分神经元。

<img src="" style="zoom:50%;" />

#### ReLU

​	没有激活函数的神经网络就是一个线性回归模型。只是单纯的线性关系，这样的网络结构有很大的局限性：即使用很多这样结构的网络层叠加，其输出和输入仍然是线性关系，无法处理有非线性关系的输入输出。因此，对每个神经元的输出做个非线性的转换也就是，将上面就加权求和$$\sum w_ix_i + b$$的结果输入到一个非线性函数，也就是激活函数中。 这样，由于激活函数的引入，多个网络层的叠加就不再是单纯的线性变换，而是具有更强的表现能力。

<img src="https://gitee.com/shen_wenxin0510/readme-pictures/raw/master/cv/image-20210522232818998.png" style="zoom:50%;" />

​	

​	在这之前的神经元输出一般使用$$tanh$$或$$sigmoid$$作为激活函数。

​	$$tanh(x) = \frac{sinhx}{coshx} = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

​	$$sigmoid:f(x) = \frac{1}{1+e^{-x}}$$

​	采用$$sigmoid$$等函数，算激活函数(指数运算)时，计算量大，采用Relu激活函数，整个过程的计算量会减少很多。同时对于深层网络，$$sigmoid$$在反向传播时，很容易出现梯度消失的情况(在sigmoid接近饱和区时，变换太缓慢，导数趋于0，这种情况会造成信息丢失),从而无法完成深层网络的训练。ReLu会使一部分神经元的输出为0，这样就造成了网络的稀疏性，并且减少了参数的相互依存关系，缓解了过拟合问题的发生。

​	论文中提到，经过实验表明$$AlexNet$$采用$$ReLU$$作为激活函数，相比于$$tanh$$，训练速度有大幅度的提升。在CIFAR-10数据集上，一个四层的$$CNN$$网络如果用$$ReLU$$作为激活函数，会比用$$tanh$$快大概六倍。

<img src="https://gitee.com/shen_wenxin0510/readme-pictures/raw/master/cv/image-20210523182746453.png" style="zoom: 67%;" />

#### 局部响应归一化 LRN

​	归一化有助于快速收敛，同时对局部神经元的活动创建竞争机制，使得其中响应比较大的值变得相对更大，并抑制其他反馈较小的神经元，增强了模型的泛化能力。

​	ReLU具有让人满意的特性，它不需要通过输入归一化来防止饱和。如果至少一些训练样本对ReLU产生了正输入，那么那个神经元上将发生学习。然而，我们仍然发现接下来的局部响应归一化有助于泛化。

​	$$\alpha ^i_{x,y}$$表示神经元激活，通过在$$(x,y)$$的位置应用核$$i$$,然后应用$$ReLU$$非线性来计算，响应归一化激活$$b^i_{x,y}$$通过下式给定：

​	$$b^i_{x,y} = a^i_{x,y}/(k+\alpha\sum^{min(N-1,i+n/2)}_{j = max(0,i - n / 2)}(a^j_{x,y})^2)^{\beta}$$

​	参数解释：

1. ​	$$b^i_{x,y}$$是归一化后的值,$$i$$是通道的位置，代表更新第几个通道的值，$$x$$与$$y$$代表待更新像素的位置
2. ​	$$a^i_{x,y}$$是输入值，是激活函数$$ReLU$$的输出值
3. ​	$$k,\alpha,\beta,n/2$$是自定义系数
4. ​	$$N$$是总的通道数

### 代码 pytorch

```python
class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 1000) -> None:
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

