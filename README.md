# A sound event classification task
50 sound classes

The competition comes from https://paper.yanxishe.com

[I think the most important thing for sequence classification task is rnn's last state](https://blog.csdn.net/zfh19941994/article/details/79981753?utm_source=blogxgwz9)



### 通过一个小项目玩转深度学习那些事
#### 学习率的调整
#### 不同优化器
#### 不同网络结构


为了防止过拟合的问题，需要在网络中加入dropout和l1,l2正则化等操作
不同的优化器所对应的初始学习率是不同的，表现出的效果也是不同的

给的用于分类的音频数据只有1600条，还要分出300条作为交叉验证集，如果直接将全特征作为训练数据，会出现无论怎样还特征测试集的准确率都无法超过50%这个瓶颈，训练集的准确率有时会90%多，有时在80%。
因此，对于这样需要大量数据的网络，首先要将音频进行分割，然后衍生出更多的数据。
