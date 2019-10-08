# text-classification-rnn-tf
Multi-label text classification with RNN(LSTM) model


##数据集
使用THUCNews的一个子集进行训练与测试，数据集请自行到[THUCTC：一个高效的中文文本分类工具包](http://thuctc.thunlp.org/)下载，请遵循数据提供方的开源协议。

本次训练使用了其中的10个分类，每个分类6500条数据。类别如下：

```
体育, 财经, 房产, 家居, 教育, 科技, 时尚, 时政, 游戏, 娱乐
```

这个子集可以在此下载：链接: https://pan.baidu.com/s/1hugrfRu 密码: qfud

数据集划分如下：

- 训练集（cnews.train.txt）: 5000*10
- 验证集（cnews.dev.txt）: 500*10
- 测试集（cnews.test.txt）: 1000*10

下载的数据集放在'./data/'目录。

## 数据预处理
程序在data_utils.py

1.对句子分词，分好词的文件保存(我保存在'./data/seg_data/'目录下)，可避免重复操作\
2.去除停用词，标点、数字等\
3.根据数据集创建字典，并保存到vocabulary.txt\
4.使用预训练的词向量,根据词向量文件和字典生成字典中每个word对于的向量，并保存。\
5.文字转数字

## BiLSTM网络

### 参数配置

```python

# Model Hyperparameters
tf.flags.DEFINE_integer("vocab_size", 5000, "Number of words in vocabulary")
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of word embedding(default: 100)")
tf.flags.DEFINE_integer("hidden_size", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("learning_rate", 0.001, "learning rate (default: 0.001)")
tf.flags.DEFINE_integer("max_sent_len", 600, "The max length of sentence (default: 100)")
tf.flags.DEFINE_integer("label_nums", 10, "Number of categories")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 20)")

```

### 网络模型
具体参看`bilstm_model.py`的实现。模型结构主要如下：
1. embedding_lookup
2. conv + pool
3. full connection

### word embedding
可以使用随机初始化的word embedding，也可以用word2vec预训练word embedding，我是将一个预训练好的文件下载下来，当然也可以自自己训练，训练好的词向量需要放在'./data/embed/'目录下。

### 训练与评估

运行 `python run_bilstm.py train`，可以开始训练。

结果显示，在验证集上的最佳效果为95.76%。

### 测试

运行 `python run_cnn.py test` 在测试集上进行测试。

```
test_loss:0.1996, test_acc:95.81%
              precision    recall  f1-score   support

          体育       1.00      1.00      1.00      1000
          财经       0.94      0.99      0.97      1000
          房产       0.87      0.89      0.88      1000
          家居       0.96      0.84      0.90      1000
          教育       0.95      0.95      0.95      1000
          科技       0.95      0.98      0.96      1000
          时尚       0.95      0.98      0.97      1000
          时政       0.96      0.94      0.95      1000
          游戏       0.98      0.98      0.98      1000
          娱乐       0.99      0.98      0.98      1000

   micro avg       0.95      0.95      0.95     10000
   macro avg       0.95      0.95      0.95     10000
weighted avg       0.95      0.95      0.95     10000   0.10      0.10      0.10     10000
```
在测试集上的准确率达到了95.81%，且各类的precision, recall和f1-score都超过了0.9。

