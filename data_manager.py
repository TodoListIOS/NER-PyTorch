import copy
import os
import pickle as cPickle
import torch


class DataManager(object):
    def __init__(self, max_length=100, batch_size=20, data_type='train', data_map_path='', tags=[]):
        self.tags = []  # 数据集中出现的所有tag
        self.word_to_ix_size = 0  # word_to_ix的长度
        self.batch_size = batch_size
        self.max_length = max_length
        self.data_type = data_type
        self.data = []  # [[第0句的word_list, 第0句的tag_list], [第1句的word_list, 第1句的tag_list], ...]
        self.batch_data = []
        self.word_to_ix = {"unk": 0, "pad": 1}
        self.ix_to_word = {}
        self.data_map_path = data_map_path  # 仅当data_type='dev'时使用,data_type='train'时不使用

        self.tag_to_ix = {"O": 0, "START": 1, "STOP": 2}
        self.ix_to_tag = {}

        if data_type == "train":
            assert tags, Exception("请指定需要训练的tag类型，如[\"ORG\", \"PER\"]")
            self.generate_tags(tags)
            self.data_path = os.path.join('data', 'ren_min_newspaper', 'train')
        elif data_type == "dev":
            self.data_path = os.path.join('data', 'ren_min_newspaper', 'dev')
            self.load_data_map()
        elif data_type == "test":
            self.data_path = "data/test"
            self.load_data_map()

        self.load_data()
        self.prepare_batch()

    def generate_tags(self, tags):
        """不含START STOP，但是tag_to_ix中含有这2个，不影响"""
        for tag in tags:
            for prefix in ["B_", "M_", "E_"]:
                self.tags.append(prefix + tag)
        self.tags.append("O")

    def load_data_map(self):
        with open(self.data_map_path, "rb") as f:
            self.data_map = cPickle.load(f)
            self.word_to_ix = self.data_map.get("word_to_ix", {})  # word_to_ix
            self.tag_to_ix = self.data_map.get("tag_to_ix", {})  # tag_to_ix

    def load_data(self):
        # load data
        # add vocab
        # covert to one-hot
        sentence = []  # 一句话中的所有word
        target = []  # 一句话中的所有tag

        with open(self.data_path) as f:
            ix = -1
            for line in f:
                ix += 1  # 正在处理第ix行数据,便于后续DEBUG
                line = line[:-1]  # 去掉一行末尾的'\n'

                if line == '':  # 一句话结束了
                    # 当数据类型为train时, sentence中不应该有0
                    if self.data_type == 'train':
                        assert 0 not in sentence
                    self.data.append([sentence, target])
                    sentence = []
                    target = []
                    continue

                try:
                    word, tag = line.split(" ")  # 以空格划分word和tag
                except Exception:  # line为空时发生异常
                    continue
                if word not in self.word_to_ix and self.data_type == "train":  # word_to_ix只对应训练数据集
                    self.word_to_ix[word] = max(self.word_to_ix.values()) + 1
                if tag not in self.tag_to_ix and self.data_type == "train" and tag in self.tags:  # tag_to_ix也只对训练数据集
                    self.tag_to_ix[tag] = len(self.tag_to_ix.keys())

                # self.data_type == 'dev'时，有可能word不在self.word_to_ix中
                # 此时约定这个word为'unk',对应的word_id为0
                sentence.append(self.word_to_ix.get(word, 0))  # 在这一步直接word->number了
                target.append(self.tag_to_ix.get(tag, 0))  # tag->number

        self.word_to_ix_size = len(self.word_to_ix.values())
        self.tag_to_ix_size = len(self.tag_to_ix.values())

        for k, v in self.word_to_ix.items():
            self.ix_to_word[v] = k

        for k, v in self.tag_to_ix.items():
            self.ix_to_tag[v] = k

        print('data_manager.DataManager.load_data():')
        print("{}数据集中有{}个[sentence, target]".format(self.data_type, len(self.data)))  # 34131
        print("{}数据集 vocab size(len(word_to_ix)): {}".format(self.data_type, self.word_to_ix_size))  # 3869
        print("{}数据集 unique tag(len(tag_to_ix)含START STOP): {}".format(self.data_type, len(self.tag_to_ix.values())))  # 9
        print('\n')

    # def convert_tag(self, data):
    #     # add E-XXX for tags
    #     # add O-XXX for tags
    #     _, tags = data
    #     converted_tags = []
    #     for _, tag in enumerate(tags[:-1]):
    #         if tag not in self.tag_to_ix and self.data_type == "train":
    #             self.tag_to_ix[tag] = len(self.tag_to_ix.keys())
    #         converted_tags.append(self.tag_to_ix.get(tag, 0))
    #     converted_tags.append(0)
    #     data[1] = converted_tags
    #     assert len(converted_tags) == len(tags), "convert error, the list dosen't match!"
    #     return data

    def prepare_batch(self):
        """prepare data for batch"""
        index = 0
        while True:
            if index + self.batch_size >= len(self.data):
                # 最后一批数据长度不足self.batch_size时，倒着取长度为self.batch的数据(和上一次取的有重复)
                pad_data = self.pad_data(self.data[-self.batch_size:])
                self.batch_data.append(pad_data)
                break
            else:
                pad_data = self.pad_data(self.data[index:index + self.batch_size])
                index += self.batch_size
                self.batch_data.append(pad_data)

    def pad_data(self, data):
        c_data = copy.deepcopy(data)
        max_length = max([len(i[0]) for i in c_data])  # 一批句子中的最大长度
        for i in c_data:
            # append之前i[0]:word2id(list), i[1]:tag2id(list)
            # append之后i[0]:word2id, i[1]:tag2id, i[2]:len(i[0])
            i.append(len(i[0]))  # len(i[0])为这句话的真实长度
            i[0] = i[0] + (max_length - len(i[0])) * [1]  # word2id(list)中缺少的部分补1 self.word_to_ix中'pad'对应1
            i[1] = i[1] + (max_length - len(i[1])) * [0]  # tag2id(list)中缺少的部分也补0 self.tag_to_ix中'O'对应0
        return c_data

    # def iteration(self):
    #     idx = 0
    #     while True:
    #         yield self.batch_data[idx]
    #         idx += 1
    #         if idx > len(self.batch_data) - 1:
    #             idx = 0

    def get_batch(self):
        for data in self.batch_data:
            yield data
