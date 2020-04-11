# Named Entity Recognition implemented by PyTorch

采用PyTorch框架实现命名实体识别，目前已经实现了**BiLSTM**和**BiLSTM_CRF**两种神经网络，含batch。



## 项目结构说明

###数据

**data**目录下存放训练和测试用的数据集，目前使用的是人民日报数据集。

renmin.txt文件夹里存放的是未经处理的raw_data，数据处理脚本是data/ren_min_newspaper目录下的data_renmin_word.py文件，其中renmin2.txt, renmin3.txt, renmin4.txt都是数据处理脚本产生的。dev和train这两个文件是神经网络的数据输入，train是训练数据集，dev是测试/评估数据集。

###模型参数

**models**文件夹中存放的是**BiLSTM**和**BiLSTM_CRF**模型的初始化参数，输入数据和训练好的模型参数。

BiLSTM_config.yml中存放的是BiLSTM模型初始化参数: embedding_dim, hidden_dim, batch_size, dropout, tags。

BiLSTM_model_params.pkl中存放的是训练好的BiLSTM模型的参数。

BiLSTM_data.pkl中存放的是BiLSTM模型的数据: batch_size, word_to_ix_size, word_to_ix, tag_to_ix, ix_to_word, ix_to_tag。

### 模型本身

**BiLSTM**模型:

BiLSTM.py为BiLSTM的模型，BiLSTM_model.py是对BiLSTM模型的封装，包含了训练、测试等功能。

**BiLSTM_CRF**模型：

BiLSTM_CRF.py为BiLSTM_CRF模型, BiLSTM_CRF_model.py是对BiLSTM_CRF模型的封装，同样包含了训练测试等功能。



##运行项目

### 准备数据/数据预处理

运行**data/ren_min_newspaper/data_renmin_word.py**文件即可。如果不想一次把所有数据都扔进神经网络训练的话，可以修改data_remin_word.py中的sentence2split()函数中的控制条件，比如如果只想拿2000条数据作为训练+测试的集合:

```python
def sentence2split():  # renmin4.txt中有2172条数据
    with open('./renmin3.txt', 'r') as inp, codecs.open('./renmin4.txt', 'w', 'utf-8') as outp:
        # texts = inp.read().decode('utf-8')
        texts = inp.read()  # 所有文本
        # sentences = re.split('[，。！？、‘’“”:]/[O]'.decode('utf-8'), texts)
        sentences = re.split('[，。！？、‘’“”:]/[O]', texts)

        line_num = 0  # renmin4.txt中文本的行数
        for sentence in sentences:
            if sentence != " ":
                outp.write(sentence.strip() + '\n')
                line_num += 1
                if line_num >= 2000:  # 拿2000条数据作为train和test的集合
                    break
```



###训练模型:

以**BiLSTM_CRF**模型为例说明训练和测试模型的方法 （**BiLSTM**模型同理）

进入到**BiLSTM_CRF_model.py**文件中，如果是想在我训练的基础上继续对模型进行训练的话，需要在 \_\_init \_\_函数的最后注释掉self.save_model()，并保留self.restore_model()。

如果不想保留已经训练好的模型，想重新开始训练的话，就把\_\_init_\_函数最后的self.restore_model()注释掉，保留self.save_model()。

接下来在BiLSTM_CRF_model.py文件中运行

```python
if __name__ == "__main__":
    start_time = time.time()
    bilstm_crf_model = BiLSTM_CRF_Model("train")
    bilstm_crf_model.train()
    print('训练总用时:{}s'.format(time.time() - start_time))
```



### 测试模型:

还是在BiLSTM_CRF_model.py文件中，不用管上面训练模型中的self.save_model()和self.restore_model()。直接在BiLSTM_CRF_model.py文件中运行

```python
if __name__ == "__main__":
    bilstm_crf_model = BiLSTM_CRF_Model('test')
    bilstm_crf_model.test()
```

测试结果如下所示:

```
/Users/yinyukang/Codes/python/NER-PyTorch/BiLSTM_CRF_model.py:74: YAMLLoadWarning: calling yaml.load() without Loader=... is deprecated, as the default Loader is unsafe. Please read https://msg.pyyaml.org/load for full details.
  config = yaml.load(fopen)
data_manager.DataManager.load_data():
train数据集中有30339个[sentence, target]
train数据集 vocab size(len(word_to_ix)): 3788
train数据集 unique tag(len(tag_to_ix)含START STOP): 12


data_manager.DataManager.load_data():
dev数据集中有7585个[sentence, target]
dev数据集 vocab size(len(word_to_ix)): 3788
dev数据集 unique tag(len(tag_to_ix)含START STOP): 12


/Users/yinyukang/opt/anaconda3/envs/NLP/lib/python3.7/site-packages/torch/nn/modules/rnn.py:51: UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.5 and num_layers=1
  "num_layers={}".format(dropout, num_layers))
BiLSTM_CRF model restore success!
----------------------------------------------------------------------
| end of test | Accuracy:  0.90 | Recall  0.90 | F1     0.90 | len(extracted_entities): 10781 | len(correct_entities): 10729
----------------------------------------------------------------------
```



## 模型的性能

训练全部的数据，每次训练都是30个epoch

|            | Accuracy | Recall | F1   | 训练时长 |
| ---------- | -------- | ------ | ---- | -------- |
| BiLSTM     | 0.91     | 0.75   | 0.82 | 40-50min |
| BiLSTM_CRF | 0.90     | 0.90   | 0,90 | 100min   |



## 待办事项

HMM模型，CRF模型，Bert模型



## Reference

感谢下面几个仓库对我的帮助

[buppt / ChineseNER](https://github.com/buppt/ChineseNER) 入门和提供数据

[yanwii / ChinsesNER-pytorch](https://github.com/yanwii/ChinsesNER-pytorch) 结构化设计、封装

感谢PyTorch官方能提供命名实体识别方便的例子

[PyTorch Tutorial](https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html?highlight=advanced)





