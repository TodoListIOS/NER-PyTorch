import os
import pickle

import torch
import yaml

from HMM import HMM
from data_manager import DataManager
from utils import path_to_entity


class HMM_Model(object):

    def __init__(self, entry='train'):
        self.data_map_path = os.path.join('models', 'HMM_data.pkl')
        self.model_config_path = os.path.join('models', 'HMM_config.yml')
        self.model_param_path = os.path.join('models', 'HMM_model_params.pkl')
        self.load_config()  # self.embedding_dim, self.hidden_dim, self.batch_size, self.drop_out, self.tags
        if entry == 'train':
            self.train_manager = DataManager(data_type='train', tags=self.tags)
            data_map = {
                "word_to_ix_size": self.train_manager.word_to_ix_size,  # word_to_ix的长度,初始化HMM模型
                "tag_to_ix_size": self.train_manager.tag_to_ix_size,  # tag_to_ix的长度,初始化HMM模型
                "word_to_ix": self.train_manager.word_to_ix,
                "tag_to_ix": self.train_manager.tag_to_ix,
                "ix_to_word": self.train_manager.ix_to_word,
                "ix_to_tag": self.train_manager.ix_to_tag,
            }
            self.save_data_map(data_map)
            self.dev_manager = DataManager(data_type='dev', data_map_path=self.data_map_path)

            self.model = HMM(hidden_state_num=self.train_manager.tag_to_ix_size,
                             observable_state_num=self.train_manager.word_to_ix_size)

            self.save_model()
            # self.restore_model()
        elif entry == 'test':
            self.train_manager = DataManager(tags=self.tags, data_type='train')
            self.dev_manager = DataManager(data_type='dev', data_map_path=self.data_map_path)
            self.model = HMM(hidden_state_num=self.train_manager.tag_to_ix_size,
                             observable_state_num=self.train_manager.word_to_ix_size)
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
                "tags": ["nr", "ns", "nt"]
            }
            yaml.dump(config, fopen)
            fopen.close()
        self.tags = config.get("tags")  # ['nr', 'ns', 'nt'] 读取数据时使用
        debug = 1

    def save_data_map(self, data_map):
        """保存训练数据的 batch_size, word_to_ix_size, word_to_ix, tag_to_ix, ix_to_word, ix_to_tag"""
        with open(self.data_map_path, "wb") as outp:
            pickle.dump(data_map, outp)

    def save_model(self):
        model_param = {
            'transitions': self.model.transitions,
            'observable_matrices': self.model.observable_matrices,
            'Pi': self.model.Pi,
        }
        with open(self.model_param_path, 'wb') as outp:
            pickle.dump(model_param, outp)

    def restore_model(self):
        try:
            with open(self.model_param_path, 'rb') as inp:
                model_param = pickle.load(inp)
                self.model.transitions = model_param.get('transitions', None)
                self.model.observable_matrices = model_param.get('observable_matrices', None)
                self.model.Pi = model_param.get('Pi', None)
            print('HMM model restore success.')
        except Exception as error:
            print('HMM model restore failed. {}'.format(error))

    def train(self):
        self.model.forward(self.train_manager.data)
        self.save_model()

    def test(self):
        extracted_entities = []
        correct_entities = []
        for dev_data in self.dev_manager.data:
            word_seq, tag_seq = dev_data
            pred_tag_seq = self.model.viterbi_decode(word_seq, self.dev_manager.tag_to_ix)

            extracted_entities = path_to_entity(word_seq, tag_seq, self.dev_manager.ix_to_word,
                                                self.dev_manager.ix_to_tag,
                                                res=extracted_entities)
            correct_entities = path_to_entity(word_seq, pred_tag_seq, self.dev_manager.ix_to_word,
                                              self.dev_manager.ix_to_tag,
                                              res=correct_entities)

        # 模型抽取出的实体和正确的实体之间的交集
        intersection_entities = [i for i in extracted_entities if i in correct_entities]

        print('-' * 70)
        if len(intersection_entities) != 0:
            accuracy = float(len(intersection_entities)) / len(extracted_entities)
            recall = float(len(intersection_entities)) / len(correct_entities)
            f1 = (2 * accuracy * recall) / (accuracy + recall)
            print('| end of test | Accuracy: {:5.2f} | Recall {:5.2f} | '
                  'F1 {:8.2f} | len(extracted_entities): {:5d} | len(correct_entities): {:5d}'
                  .format(accuracy, recall, f1, len(extracted_entities), len(correct_entities)))
        else:
            print('| end of test | Accuracy: {:5.2f}'.format(0))
        print('-' * 70)


if __name__ == '__main__':
    # hmm_model = HMM_Model(entry='train')
    # hmm_model.train()
    hmm_model = HMM_Model(entry='test')
    hmm_model.test()
