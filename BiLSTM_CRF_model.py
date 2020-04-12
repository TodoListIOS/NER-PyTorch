# -*- coding:utf-8 -*-
'''
@Author: yanwii
@Date: 2018-10-31 10:00:03
'''
import os
import pickle
import sys
import time
from copy import deepcopy
import yaml

import torch
import torch.optim as optim
from data_manager import DataManager
from BiLSTM_CRF import BiLSTM_CRF
from utils import get_tags, format_result, path_to_entity


class BiLSTM_CRF_Model(object):

    def __init__(self, entry="train"):
        self.model_config_path = os.path.join('models', 'BiLSTM_CRF_config.yml')
        self.model_param_path = os.path.join('models', 'BiLSTM_CRF_model_params.pkl')
        self.data_map_path = os.path.join('models',
                                          'BiLSTM_CRF_data.pkl')  # 存batch_size, word_to_ix_size, word_to_ix, tag_to_ix
        self.load_config()  # emb_size, hid_size, batch_size, drop_out, model_path, tags
        self._best_loss = 1e18
        if entry == "train":
            self.train_manager = DataManager(batch_size=self.batch_size, tags=self.tags, data_type='train')
            self.total_size = len(self.train_manager.batch_data)  # 多少批数据
            data = {
                "batch_size": self.train_manager.batch_size,  # 一批数据的量，初始化神经网络的参数
                "word_to_ix_size": self.train_manager.word_to_ix_size,  # word_to_ix的长度,初始化神经网络的参数
                "word_to_ix": self.train_manager.word_to_ix,  # 初始化神经网络的参数
                "tag_to_ix": self.train_manager.tag_to_ix,  # 初始化神经网络的参数
                "ix_to_word": self.train_manager.ix_to_word,
                "ix_to_tag": self.train_manager.ix_to_tag,
            }
            self.save_data_map(data)
            self.dev_manager = DataManager(batch_size=30, data_type="dev", data_map_path=self.data_map_path)

            self.model = BiLSTM_CRF(
                tag_to_ix=self.train_manager.tag_to_ix,  # 训练数据的tag_to_ix
                batch_size=self.batch_size,  # load_config中 默认128
                vocab_size=len(self.train_manager.word_to_ix),  # 训练数据的len(word_to_ix)
                embedding_dim=self.embedding_dim,  # load_config中 默认300
                hidden_dim=self.hidden_dim,  # load_config中 默认128
                dropout=self.dropout,  # load_config中 默认0.5
                use_gpu=False,
                use_crf=True,  # 使用crf层
            )
            # self.restore_model()
            self.save_model()
        elif entry == 'test':
            self.train_manager = DataManager(batch_size=self.batch_size, tags=self.tags, data_type='train')
            self.dev_manager = DataManager(batch_size=30, data_type='dev', data_map_path=self.data_map_path)
            # data_map = self.load_data_map()
            self.model = BiLSTM_CRF(
                tag_to_ix=self.train_manager.tag_to_ix,
                batch_size=self.batch_size,
                vocab_size=self.train_manager.word_to_ix_size,
                embedding_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim,
                dropout=self.dropout,
                use_gpu=False,
                use_crf=True,
            )
            self.restore_model()

    def load_config(self):
        try:
            fopen = open(self.model_config_path)
            config = yaml.load(fopen)
            fopen.close()
        except Exception as error:
            print("Load config failed, using default config {}".format(error))
            fopen = open(self.model_config_path, "w")
            config = {
                "embedding_size": 300,
                "hidden_size": 150,
                "batch_size": 128,
                "dropout": 0.5,
                "tags": ["nr", "ns", "nt"]
            }
            yaml.dump(config, fopen)
            fopen.close()
        self.embedding_dim = config.get("embedding_dim")  # 初始化神经网络的参数 300
        self.hidden_dim = config.get("hidden_dim")  # 初始化神经网络的参数 150
        self.batch_size = config.get("batch_size")  # 初始化神经网络的参数 128
        self.dropout = config.get("dropout")  # 初始化神经网络的参数 0.5
        self.tags = config.get("tags")  # ['nr', 'ns', 'nt'] 读取数据时使用
        debug = 1

    def save_model(self):
        try:
            torch.save(self.model.state_dict(), self.model_param_path)
            print('BiLSTM_CRF model save success!')
        except Exception as error:
            print('BiLSTM_CRF model save failed! {}'.format(error))

    def restore_model(self):
        try:
            self.model.load_state_dict(torch.load(self.model_param_path))
            print("BiLSTM_CRF model restore success!")
        except Exception as error:
            print("BiLSTM_CRF model restore faild! {}".format(error))

    def save_data_map(self, data_map):
        """保存训练数据的 batch_size, word_to_ix_size, word_to_ix, tag_to_ix, ix_to_word, ix_to_tag"""
        with open(self.data_map_path, "wb") as outp:
            pickle.dump(data_map, outp)

    def load_data_map(self):
        """加载训练数据的 batch_size, word_to_ix_size, word_to_ix, tag_to_ix, ix_to_word, ix_to_tag"""
        with open(self.data_map_path, "rb") as inp:
            data_map = pickle.load(inp)
        return data_map

    def train(self):
        # optimizer = optim.Adam(self.model.parameters())
        optimizer = optim.SGD(self.model.parameters(), lr=0.005, weight_decay=1e-4)  # 官方版

        for epoch in range(10):
            epoch_start_time = time.time()
            index = 0
            epoch_loss = 0  # 一个epoch的loss
            for batch in self.train_manager.get_batch():  # yield生成器
                index += 1
                self.model.zero_grad()

                # sentence{tuple:20}为填充后的, tags{tuple:20}为填充后的, length{tuple:20}为填充前的长度
                sentences, tags, lengths = zip(*batch)
                sentences_tensor = torch.tensor(sentences, dtype=torch.long)  # shape: torch.Size([batch_size, max_sentence_len])
                tags_tensor = torch.tensor(tags, dtype=torch.long)  # shape: torch.Size([batch_size, max_sentence_len])
                lengths_tensor = torch.tensor(lengths, dtype=torch.long)  # shape:batch_size
                batch_size = sentences_tensor.shape[0]  # 例:128

                # 一批数据的loss
                loss = self.model.neg_log_likelihood(sentences_tensor, tags_tensor, lengths_tensor)
                # loss的格式 tensor([10.8672], grad_fn=<SubBackward0>

                # 把loss转化为数字形式
                loss_to_num = loss.cpu().tolist()[0]
                epoch_loss += loss_to_num

                # 一批数据中平均每条数据的loss
                avg_loss = loss_to_num / batch_size

                # index / self.total_size == len / 25 ==> len = (index * 25) / self.total_size
                # progress = ("█" * int(index * 25 / self.total_size)).ljust(25)  # 指定字符长度为25
                # print("""epoch [{}] |{}| {}/{}\n\tloss {:.2f}""".format(
                #     epoch, progress, index, self.total_size, loss.cpu().tolist()[0]
                # ))

                print('| epoch {:2d} | {:2d}/{:2d} batches | '
                      'batch loss {:5.2f} | avg loss {:5.2f}'.format(
                    epoch, index, self.total_size, loss_to_num, avg_loss))

                loss.backward()
                optimizer.step()

            # 一次训练结束后更新模型
            if self._best_loss > epoch_loss:
                self._best_loss = epoch_loss
                print('epoch {} 更新模型'.format(epoch))
                self.save_model()
                # self.best_model = deepcopy(self.model)

            print('| end of epoch {:2d} | Training time: {:5.2f} |'.format(epoch, time.time() - epoch_start_time))
            # 每个epoch结束的时候使用dev数据集对模型进行评估
            self.evaluate(epoch)

    def evaluate(self, epoch=0):
        extracted_entities = []  # 由模型抽取出来的所有实体
        correct_entities = []  # 正确的所有实体

        for batch in self.dev_manager.get_batch():  # yield生成器
            sentences, labels, length = zip(*batch)
            # sentences:{tuple:30} labels:{label"30} length:{tuple:30}
            # length中存着每个sentence填充之前的长度

            _, predict_paths = self.model(sentences, length)  # 一批数据前向传播

            unpadded_sentences = []  # 去掉填充的部分
            unpadded_labels = []  # 去掉填充的部分

            assert len(sentences) == len(labels)
            assert len(sentences) == len(length)
            for ix in range(len(sentences)):
                unpadded_sentence = sentences[ix][:length[ix]]
                unpadded_label = labels[ix][:length[ix]]
                unpadded_sentences.append(unpadded_sentence)
                unpadded_labels.append(unpadded_label)

            for unpadded_sentence, correct_path, predict_path in zip(unpadded_sentences, unpadded_labels, predict_paths):
                extracted_entities = path_to_entity(seq_of_word=unpadded_sentence, seq_of_tag=predict_path,
                                                    ix_to_word=self.train_manager.ix_to_word,
                                                    ix_to_tag=self.train_manager.ix_to_tag,
                                                    res=extracted_entities)
                correct_entities = path_to_entity(seq_of_word=unpadded_sentence, seq_of_tag=correct_path,
                                                  ix_to_word=self.train_manager.ix_to_word,
                                                  ix_to_tag=self.train_manager.ix_to_tag,
                                                  res=correct_entities)

        # 模型抽取出的实体和正确的实体之间的交集
        intersection_entities = [i for i in extracted_entities if i in correct_entities]

        print('-' * 70)

        if len(intersection_entities) != 0:
            accuracy = float(len(intersection_entities)) / len(extracted_entities)
            recall = float(len(intersection_entities)) / len(correct_entities)
            f1 = (2 * accuracy * recall) / (accuracy + recall)
            if epoch == -1: # test
                print('| end of test | Accuracy: {:5.2f} | Recall {:5.2f} | '
                      'F1 {:8.2f} | len(extracted_entities): {:5d} | len(correct_entities): {:5d}'
                      .format(accuracy, recall, f1, len(extracted_entities), len(correct_entities)))
            else:
                print('| end of epoch {:2d} | Accuracy: {:5.2f} | Recall {:5.2f} | '
                      'F1 {:8.2f} | len(extracted_entities): {:5d} | len(correct_entities): {:5d}'
                      .format(epoch, accuracy, recall, f1, len(extracted_entities), len(correct_entities)))
        else:
            if epoch == -1:  # test
                print('| end of test | Accuracy: {:5.2f}'.format(0))
            else:
                print('| end of epoch {:2d} | Accuracy: {:5.2f}'.format(epoch, 0))
        print('-' * 70)

    def test(self):
        self.evaluate(epoch=-1)

    # def predict(self, input_str=""):
    #     if not input_str:
    #         input_str = input("请输入文本: ")
    #     input_vec = [self.word_to_ix.get(i, 0) for i in input_str]
    #     # convert to tensor
    #     sentences = torch.tensor(input_vec).view(1, -1)
    #     _, paths = self.model(sentences)
    #
    #     entities = []
    #     for tag in self.tags:
    #         tags = get_tags(paths[0], tag, self.tag_to_ix)
    #         entities += format_result(tags, input_str, tag)
    #     return entities


if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("menu:\n\ttrain\n\tpredict")
    #     exit()
    # if sys.argv[1] == "train":
    #     cn = ChineseNER("train")
    #     cn.train()
    # elif sys.argv[1] == "predict":
    #     cn = ChineseNER("predict")
    #     print(cn.predict())

    #####
    # start_time = time.time()
    # bilstm_crf_model = BiLSTM_CRF_Model("train")
    # bilstm_crf_model.train()
    # print('训练总用时:{}s'.format(time.time() - start_time))

    bilstm_crf_model = BiLSTM_CRF_Model('test')
    bilstm_crf_model.test()

