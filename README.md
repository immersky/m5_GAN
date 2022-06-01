个人学习GAN


最原始的GAN论文重要的不是代码的构成，而是其推导和思想，m5_GAN.py内含一个生成MINST数字的 GAN,其中判别器和生成器都是很简单的几层全连接神经网络，因此我就不自己写和推理了，转载一个别人的代码,后续在复现DCGAN和CGAN再关注一下代码

代码转载自https://blog.csdn.net/jizhidexiaoming/article/details/96485095



论文原文:https://arxiv.org/pdf/1406.2661.pdf

参考：https://blog.csdn.net/qq_30091945/article/details/101079255,https://zhuanlan.zhihu.com/p/117529144.https://blog.csdn.net/qikaihuting/article/details/84950947,https://blog.csdn.net/qq_45849192/article/details/123323982,https://blog.csdn.net/jizhidexiaoming/article/details/96485095,https://blog.csdn.net/qinglingLS/article/details/91480550

GAN理论推导参考https://blog.csdn.net/stalbo/article/details/79283399



以下包含个人理解，水平有限，如有错误恳求指正。

# 1.GAN简介

GAN（Generative Adversarial Network）全名叫做**对抗生成网络**或者生成对抗网络。GAN内部包含两个主要网络结构，一个称作生成器G，一个乘坐判别器D。

生成器G：输入随机生成的高斯噪声数据，输出生成图像

判别器D：输入图像，判别输入图像究竟是真实数据，还是G生成的。

定义特殊的损失函数，每一次梯度下降中，将G，D的权重更新，迭代到最优时，D对于G生成的数据，判断其真实和生成的概率均为0.5。

# 2.GAN目标函数与数学原理

为了理解GAN的目标函数定义和迭代过程，必须掌握一些基础数学原理

论坛在给出目标函数前，再次解释了生成器G和判别器D的作用，为了让生成器G（注意，生成器G是一个神经网络，模拟某个目标函数，本身也代表一个函数）学习让他生成的数据$P_g$向数据X靠拢，作者定义了对于噪声数据的概率$P_z(z)$，然后用生成器 函数G(z;$\theta _g$)，噪声数据z映射到真实数据x的数据空间,G可微，z为随机生成的噪声数据，$\theta _g$是多层神经网络G的各种参数，比如权重。D是另一个多层神经网络，D($x;\theta_d$),x为输入数据，另一个是网络的参数,D(x)显示x是原始数据的概率。

论文中的目标函数定义：
$\underset {G} {min} \underset {D}{max}V(D,G)=E_{x\sim{P_{data}(x)}}[logD(x)]+E_{z\sim P_z(x)}[log(1-D(G(z)))]$

+号左边的项是判别器 D 判别样本是不是从服从$P_{data}(x)$分布的总体中取出来的样本，其中E是指取期望，最大化这一项相当于令判别器 D 在 x 服从于 data 的概率密度时能准确地预测 D(x)=1

$D(x)=1 \ when \ x\sim P_{panda}(x)$

另外一项是企图欺骗D的G，与上面的描述相似

对于 D 而言要尽量使公式最大化（识别能力强），而对于 G 又想使之最小（生成的数据接近实际数据）。整个训练是一个迭代过程，在给定 G 的情况下先最大化 V(D,G)，而取 D(就像对某一个变量求偏导就把其它变量当常数)，然后固定 D，并最小化 V(D,G) 而得到 G。其中，给定 G，最大化 V(D,G)评估了 生成数据和实际数据之间的差异或距离。

训练：对于G，要使目标函数最小，对于D，要使目标函数最大，且他们使用同一个目标函数

论文提到，让D在训练中迭代一下就达到全局最优是不可能的，且在有限数据集往往过拟合，所以，在训练用的循环语句中，结构如下

![train](https://user-images.githubusercontent.com/74494790/171400558-092b06e3-e679-432c-a1ce-8a16a238114e.png)



让D,G交替迭代循环，如图

![loss](https://user-images.githubusercontent.com/74494790/171400605-47501f6f-de9d-430c-99d8-239eaef2d58b.png)


论文提到，实践中,上面那个目标函数可能没有提供足够的梯度供G学习，训练迭代步数较少时，D可以随便区别，这个时候应该把重心放到训练G，让G能够最大化$logD(G(z))$




论文给了个例子，图中，黑色曲线是真实样本的概率分布函数，绿色曲线是虚假样本的概率分布函数，蓝色曲线是判别器D的输出，它的值越大表示这个样本越有可能是真实样本。最下方的平行线是随机生成的噪声z，它映射到了x。

可见,随着训练的推进，虚假样本的分布逐渐与真实样本重合，D虽然也在不断更新，

最后，黑线和绿线最后几乎重合，模型达到了最优状态，这时 D 的输出对于任意样本都是 0.5。

为了能够更好的理解目标函数的构造，来学习以下概率论知识，下文主要转载自https://blog.csdn.net/stalbo/article/details/79283399

## 2.1.KL散度的定义

KL散度（KL divergence），这是统计中的一个概念，是衡量两种概率分布的相似程度，其越小，表示两种概率分布越接近。

对于离散随机变量，定义如下：
![lisan](https://user-images.githubusercontent.com/74494790/171400808-29adf3a3-0be9-4e2e-8593-7748ae99baac.jpg)



对于连续随机变量，定义如下:

![lianxu](https://user-images.githubusercontent.com/74494790/171400842-3aab6050-9d29-4b58-818e-cc42b00178c6.jpg)



对于GAN，我们想要将一个随机高斯噪声z通过一个生成网络G得到一个和真的数据分布 $P_{data}(x)$ 差不多的生成分布 $P_G(x;θ)$，其中的参数 θ是网络的参数决定的，我们希望找到 θ使得两个分布尽可能接近



## 2.2.最大似然估计

最大似然估计用于估计总体分布的未知参数，以下是教科书中关于最大似然估计的内容


![1 ](https://user-images.githubusercontent.com/74494790/171400894-3fc6f1d3-5525-4b53-a555-ffe6a2bf5614.jpg)

![2](https://user-images.githubusercontent.com/74494790/171401045-ab081047-ba27-4e01-816f-d9bbe5248b5c.jpg)

![3](https://user-images.githubusercontent.com/74494790/171400955-43f459f4-c770-4072-b2f5-bacfcc81a0df.jpg)

![4](https://user-images.githubusercontent.com/74494790/171400885-de1eb44a-3fef-44b4-9ca0-43b7913c43b4.jpg)


对于GAN，

似然函数为
![likehood](https://user-images.githubusercontent.com/74494790/171400932-5307501b-f35a-4b51-83db-e2d6ee1abd06.jpg)


则要找到$\theta^*$使似然函数达到最大值,如下式



![max](https://user-images.githubusercontent.com/74494790/171400978-6db63ba0-0366-4ba2-bedb-4139b1a7753b.jpg)

在上面的推导中，我们希望最大化似然函数 LL。若对似然函数取对数，那么累乘 ∏∏就能转化为累加 ∑，并且这一过程并不会改变最优化的结果，随后将式子转化为求令$logP_G(x;\theta)$期望最大化的$\theta$,而期望根据概率论公式，展开为积分形式$\int P_{data}(x)logP_G(x;\theta)dx$,而这个最优化过程针对$\theta$，添加一项不含$\theta$的积分并不影响结果（$ \theta $的值），则添加一项$-\int P_{data}(x)logP_{data}(x)dx$,构造KL散度

$\underset{\theta} {arg\ max}\int P_{data}(x)log\frac {P_G(x;\theta)}{P_{data}(x)}dx$

而

$P_G(x)=\int_z P_{prior}(z)I_{|G(z)=x|}dz$

其中

$I_{G(z)=x}=\begin{cases}0 &G(z)\ne x\ \ 1&G(z)=x \end{cases}$

可见，实际上$P_G（x）$不可求出

## 2.3.GAN最优化问题可解的数学证明

详见原论文**4.1 Global Optimality of pg = pdata**部分。https://blog.csdn.net/stalbo/article/details/79283399 
根据原论文说的也非常好



