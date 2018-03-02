### RNN_LSTM



### 一、word embedding 部分

* word embedding的理解：将宋词中的出现频次前5000个汉字生成字典后，wordembedding将单个字，映射到高维的空间中。根据提供的数据集，在新的空间上找到字和字之间的向量关系。
* word embading的内容

​      同类型的词的向量比较接近   东西南北方向的

​      有相关性的词 向量也会比较接近  声、调和听、闻就比较接近  

​      向量之间的距离也有一定线性关系   十、百、千、万

![tsne1](F:\GitHub\quiz_RNN_LSTM\tsne1.png)


### 二、RNN训练部分

RNN训练部分完成的内容：
- train.py 训练

  train.py中写了feed_dict的内容，用sess.run执行了模型参数。

- utils.py 数据处理

  utils.py中实现了函数def get_train_data(vocabulary, batch_size, num_steps)，函数根据数据集内容vocabulary、字典dictionary、batch_size、num_steps生成data和target。

- model.py 网络

  model.py中实现了rnn_cell的结构和softmax的计算。

#### 结果

tinymind上运行结结果：https://www.tinymind.com/executions/f11qw68i

tinymind上不能显示中文，需要从unicode转码为中文显示。

unicode:

```2018-03-01 00:35:35,471 - DEBUG - sample.py:81 - ==============[\u6c5f\u795e\u5b50]==============2018-03-01 00:35:35,471 - DEBUG - sample.py:82 - \u6c5f\u795e\u5b50\u4e00\u7b11\u4e00\u676f\u540c\u9189\u3002\u4e00\u7b11\u4e00\u676f\uff0c\u4e00\u676f\u4e00\u9189\uff0c\u4e00\u7b11\u4e00\u676f\u9152\u3002
2018-03-01 00:35:35,696 - DEBUG - sample.py:81 - ==============[\u8776\u604b\u82b1]==============
2018-03-01 00:35:35,696 - DEBUG - sample.py:82 - \u8776\u604b\u82b1\u4e00\u7b11\u4e00\u679d\u6625\u8272\u3002

\u4e00\u679d\u82b1\u4e0b\u6625\u98ce\uff0c\u4e00\u679d\u6625\u8272\u3002\u6625\u98ce\u4e0d\u65ad\u6625\u98ce\u6076\u3002\u4e00\u58f0\u4e00\u6795\uff0c\u4e00\u58f0\u58f0\u65ad\uff0c\u4e00\u58f0\u58f0\u65ad\u3002

\u4e00\u58f0\u5439\u5c3d\u897f\u6e56\u96e8\u3002\u53c8\u4e0d\u662f\u3001\u6625\u98ce\u96e8\u3002\u4e00\u70b9
2018-03-01 00:35:35,920 - DEBUG - sample.py:81 - ==============[\u6e14\u5bb6\u50b2]==============
2018-03-01 00:35:35,920 - DEBUG - sample.py:82 - \u6e14\u5bb6\u50b2\u4e00\u58f0\u4e0d\u5230\u6c5f\u5357\u6c34\u3002\u4e00\u7247\u6c5f\u5357\uff0c\u4e00\u7247\u6c5f\u5357\u6c34\u3002





\u5218UNK

\u6c34\u8c03\u6b4c\u5934\uff08\u5bff\u5218\u5b88\uff09

\u4e00\u7b11\u4e00\u756a\u9152\uff0c\u4e00\u7b11\u4e00\u676f\u4e2d\u3002\u4e00\u676f\u4e00\u9189\uff0c\u4e00\u7b11\u4e00\u7b11\u4e0d\u80fd\u916c
```

中文

```2018-03-01 00:35:35,471 - DEBUG - sample.py:81 - ==============[江神子]==============
2018-03-01 00:35:35,696 - DEBUG - sample.py:81 - ==============[蝶恋花]==============
2018-03-01 00:35:35,696 - DEBUG - sample.py:82 - 蝶恋花一笑一枝春色。

一枝花下春风，一枝春色。春风不断春风恶。一声一枕，一声声断，一声声断。

一声吹尽西湖雨。又不是、春风雨。一点
2018-03-01 00:35:35,920 - DEBUG - sample.py:81 - ==============[渔家傲]==============
2018-03-01 00:35:35,920 - DEBUG - sample.py:82 - 渔家傲一声不到江南水。一片江南，一片江南水。





刘UNK

水调歌头（寿刘守）

一笑一番酒，一笑一杯中。一杯一醉，一笑一笑不能酬
```



### 遇到的问题&心得体会：

问题1：num_steps、init_state、final_state、output_tensor没理解对，导致模型创建错误，训练结果不好。

心得：final_state是output_tensor的最后一个state，而且需要在每个训练step后再传给下一个step的init_state，让模型可以学习到数据间的相关性，发挥循环网络的作用。



问题2：训练时报错，ValueError: Shape [-1,32] has negative dimensions。

tensorflow的placehold中定义尺寸参数含有None时，在初始化如果没有传入对应参数，None就会被初始化成-1，改正feed_dict解决问题。