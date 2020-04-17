import torch
import torch.nn.functional as F
from torch import nn

START_TAG = "START"
STOP_TAG = "STOP"


class BiLSTM_CRF(nn.Module):

    def __init__(self,
                 tag_to_ix,
                 batch_size,
                 vocab_size,  # len(word_to_ix)
                 embedding_dim,  # 句中每个word的特征维度
                 hidden_dim,
                 dropout=0.5,
                 use_gpu=False,  # 没有添加GPU
                 use_crf=True,  # BiLSTM+CRF网络，使用CRF
                 ):
        super(BiLSTM_CRF, self).__init__()
        self.tag_to_ix = tag_to_ix  #
        self.batch_size = batch_size
        self.vocab_size = vocab_size  #
        self.embedding_dim = embedding_dim  #
        self.hidden_dim = hidden_dim  #
        self.dropout = dropout
        self.use_gpu = use_gpu
        self.use_crf = use_crf
        assert self.use_crf
        # 官方版的4个初始化参数：
        # vocab_size, tag_to_ix, embedding_dim, hidden_dim

        self.tag_size = len(tag_to_ix)  #

        self.word_embeddings = nn.Embedding(vocab_size, self.embedding_dim)
        # self.word_embeds = nn.Embedding(vocab_size, embedding_dim) 官方

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True, dropout=self.dropout)
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
        #                             num_layers=1, bidirectional=True) 官方

        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)
        # self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size) 官方

        self.hidden = self.init_hidden()

        if self.use_crf:
            self.transitions = nn.Parameter(
                torch.randn(self.tag_size, self.tag_size))

            # self.transitions.data[self.tag_map[START_TAG], :] = -10000
            # self.transitions.data[:, self.tag_map[STOP_TAG]] = -10000
            self.init_transitions(print_transitions=False)

    def init_hidden(self):
        return (torch.randn(2, self.batch_size, self.hidden_dim // 2),
                torch.randn(2, self.batch_size, self.hidden_dim // 2))
        # return (torch.randn(2, 1, self.hidden_dim // 2), 官方
        #         torch.randn(2, 1, self.hidden_dim // 2))

    def init_transitions(self, print_transitions=False):
        self.transitions.data[self.tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, self.tag_to_ix[STOP_TAG]] = -10000

        tag_list = ['nr', 'ns', 'nt']

        for tag in tag_list:
            # 以tag='nr'为例
            self.transitions.data[self.tag_to_ix['O'], self.tag_to_ix['M_' + tag]] = -10000  # M_nr -> O
            self.transitions.data[self.tag_to_ix['O'], self.tag_to_ix['B_' + tag]] = -10000  # B_nr -> O
            self.transitions.data[self.tag_to_ix[STOP_TAG], self.tag_to_ix['M_' + tag]] = -10000  # M_nr -> STOP
            self.transitions.data[self.tag_to_ix[STOP_TAG], self.tag_to_ix['B_' + tag]] = -10000  # B_nr -> STOP

        for to_tag in tag_list:
            # to_tag='nr'为例
            for from_tag in tag_list:
                # B_nr -> B_nr, B_ns -> B_nr, B_nt -> B_nr
                self.transitions.data[self.tag_to_ix['B_' + to_tag], self.tag_to_ix['B_' + from_tag]] = -10000
                # M_nr -> B_nr, M_ns -> B_nr, M_nt -> B_nr
                self.transitions.data[self.tag_to_ix['B_' + to_tag], self.tag_to_ix['M_' + from_tag]] = -10000

        for to_tag in tag_list:
            # to_tag='nr'为例
            self.transitions.data[self.tag_to_ix['M_' + to_tag], self.tag_to_ix['O']] = -10000  # O -> M_nr
            self.transitions.data[self.tag_to_ix['M_' + to_tag], self.tag_to_ix[START_TAG]] = -10000  # START -> M_nr
            self.transitions.data[self.tag_to_ix['M_' + to_tag], self.tag_to_ix['E_' + to_tag]] = -10000  # E_nr -> M_nr
            # 对于其他两个实体
            for from_tag in tag_list:
                if from_tag == to_tag:
                    continue
                # B_ns -> M_nr B_nr -> M_nr
                self.transitions.data[self.tag_to_ix['M_' + to_tag], self.tag_to_ix['B_' + from_tag]] = -10000
                # M_ns -> M_nr M_nr -> M_nr
                self.transitions.data[self.tag_to_ix['M_' + to_tag], self.tag_to_ix['M_' + from_tag]] = -10000
                # E_ns -> M_nr E_nr -> M_nr
                self.transitions.data[self.tag_to_ix['M_' + to_tag], self.tag_to_ix['E_' + from_tag]] = -10000

        for to_tag in tag_list:
            # 以to_tag='nr'为例
            self.transitions.data[self.tag_to_ix['E_' + to_tag], self.tag_to_ix['O']] = -10000  # O -> E_nr
            self.transitions.data[self.tag_to_ix['E_' + to_tag], self.tag_to_ix[START_TAG]] = -10000  # START -> E_nr
            self.transitions.data[self.tag_to_ix['E_' + to_tag], self.tag_to_ix['E_' + to_tag]] = -10000  # E_nr -> E_nr
            for from_tag in tag_list:
                if from_tag == to_tag:
                    continue
                # B_ns -> E_nr B_nt -> E_nr
                self.transitions.data[self.tag_to_ix['E_' + to_tag], self.tag_to_ix['B_' + from_tag]] = -10000
                # M_ns -> E_nr M_nt -> E_nr
                self.transitions.data[self.tag_to_ix['E_' + to_tag], self.tag_to_ix['M_' + from_tag]] = -10000
                # E_ns -> E_nr E_nt -> E_nr
                self.transitions.data[self.tag_to_ix['E_' + to_tag], self.tag_to_ix['E_' + from_tag]] = -10000

        if print_transitions:
            print('初始化后的转移矩阵为')
            print(' ' * 10, end='')
            for tag, ix in self.tag_to_ix.items():
                print('{:>10}'.format(tag), end='')  # 右对齐
            print()
            for to_tag, j in self.tag_to_ix.items():
                print('{:>10}|'.format(to_tag), end='')  # 一行开始
                for from_tag, i in self.tag_to_ix.items():
                    elem = self.transitions.data[j, i].item()  # 由i -> j的转移值
                    print('{:9.3}|'.format(elem), end='')
                print()  # 一行结束 换行

    def _get_lstm_features(self, sentences):
        """BiLSTM的输出（还没经过CRF层）"""
        self.hidden = self.init_hidden()
        length = sentences.shape[1]  # sentence:torch.Size([batch_size, max_sentence_len])  length为最长数据的长度
        embeddings = self.word_embeddings(sentences).view(self.batch_size, length, self.embedding_dim)
        # embeddings的shape: torch.Size([batch_size, max_sentence_len, embedding_dim])

        # 当lstm的batch_first=True时，
        # input tensor的输入形式变成（batch_size, seq_len, input_size） input_size为每个word的特征维度(embedding_dim)
        # output tensor的输出形式变成 (batch_size, seq_len, hidden_size*num_directions)
        # 这里的input_size是官方文档里的叫法，其实就是embedding_size（表示一个word的向量的长度）
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)
        # 例lstm_out: torch.Size([128, 47, 150]) 128为batch_size, 47为seq_len, 150为hidden_size(75)*num_directions(2)

        lstm_out = lstm_out.view(self.batch_size, -1, self.hidden_dim)  # shape:torch.Size([128, 47, 150])
        lstm_feats = self.hidden2tag(lstm_out)  # Linear(in_features=150, out_features=12, bias=True)
        return lstm_feats  # torch.Size([20, 332, 9])即torch.Size([batch_size, seq_max_len, tag_size])

    def _score_sentence(self, feats, label):
        """
        计算正确标注对应的得分
        :param feats: [len_sent * tag_size] 发射矩阵
        :param label: [1 * len_sent] 标注正确的tag序列
        :return: gold_score 正确标注对应的得分
        """
        score = torch.zeros(1)  # tensor([0.])
        label = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), label])  # 先把开始标签放入原label
        for index, feat in enumerate(feats):
            emission_score = feat[label[index + 1]]
            transition_score = self.transitions[label[index + 1], label[index]]
            score += emission_score + transition_score
        score += self.transitions[self.tag_to_ix[STOP_TAG], label[-1]]  # label中最后一个标签->STOP TAG的分数
        return score

    def _forward_alg(self, feats):
        """
        所有tag序列的得分之和
        :param feats: [len_sent * tag_size] 发射矩阵
        :return: score = log(e^Score(X, y0) + e^Score(X, y1) + e^Score(X, y2) + ......)
        """
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full([self.tag_size], -10000.)
        # START_TAG has all of the score.
        init_alphas[self.tag_to_ix[START_TAG]] = 0.  # word开始的log_sum_exp

        # Wrap in a variable so that we will get automatic backprop
        # Iterate through the sentence
        forward_var = init_alphas

        for feat_index in range(feats.shape[0]):  # feats.shape[0]=sentence_len
            previous = torch.stack([forward_var] * feats.shape[1])
            emit_scores = torch.unsqueeze(feats[feat_index], 0).transpose(0, 1)
            next_tag_var = previous + emit_scores + self.transitions
            forward_var = torch.logsumexp(next_tag_var, dim=1)  # 按行求log_sum_exp
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        terminal_var = torch.unsqueeze(terminal_var, 0)
        alpha = torch.logsumexp(terminal_var, dim=1)[0]  # 按行求log_sum_exp
        return alpha

    def neg_log_likelihood(self, sentences, tags, lengths):
        """
        计算Loss
        :param sentences: shape为torch.Size([batch_size, max_sentence_len])
        :param tags: shape为torch.Size([batch_size, max_sentence_len])
        :param length: shape为batch_size，存着每个sentence的被填充前的真实长度
        :return:
        """
        self.batch_size = sentences.size(0)

        featss = self._get_lstm_features(sentences)  # BiLSTM的输出（还没经过CRF层）
        # feats的shape为torch.Size([batch_size, max_sentence_len, tagset_size])

        gold_score = torch.zeros(1)  # loss的后半部分S(X,y)的结果 tensor([0.])
        forward_score = torch.zeros(1)  # loss的log部分的结果 tensor([0.])
        for feats, tag, length in zip(featss, tags, lengths):
            # feats shape:torch.Size([max_sentence_len, tagset_size])
            # tag shape: max_sentence_len
            # length为这个sentence被填充的前的真实长度
            feats = feats[:length]  # 取有效值，忽略填充位置的结果
            tag = tag[:length]  # 去有效值，忽略填充位置的结果
            gold_score += self._score_sentence(feats, tag)
            forward_score += self._forward_alg(feats)
        return forward_score - gold_score

    def forward(self, sentences, lengths=None):
        """
        :params sentences sentences to predict
        :params lengths represent the ture length of sentence, the default is sentences.size(-1)
        """
        sentences = torch.tensor(sentences, dtype=torch.long)  # torch.Size([30, 332])
        if not lengths:
            lengths = [i.size(-1) for i in sentences]
        self.batch_size = sentences.size(0)  # 30
        logits = self._get_lstm_features(sentences)  # torch.Size([30, 332, 9])
        scores = []
        paths = []
        for logit, leng in zip(logits, lengths):
            # logit的shape torch.Size([332, 9])
            logit = logit[:leng]  # 取有效值
            score, path = self._viterbi_decode(logit)
            scores.append(score)
            paths.append(path)
        return scores, paths

    def _viterbi_decode(self, feats):
        backpointers = []
        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tag_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars  # torch.Size([1, 9])

        for feat_index in range(feats.shape[0]):  # 0-168
            forward_vars = torch.stack([forward_var] * feats.shape[1])  # 扩展成torch.Size([9, 1, 9])
            forward_vars = torch.squeeze(forward_vars)  # torch.Size([9, 9])
            next_tag_vars = forward_vars + self.transitions
            viterbivar_s_t, bptr_s_t = torch.max(next_tag_vars, dim=1)  # 按行取最大值
            # viterbivar_s_t的shape为9, bptr_s_t的shape为9

            feat_s_t = torch.unsqueeze(feats[feat_index], 0)  # torch.Size(1, 9)
            forward_var = torch.unsqueeze(viterbivar_s_t, 0) + feat_s_t  # 更新word的得分

            backpointers.append(bptr_s_t.tolist())

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]  # 最后加上到STOP_TAG的转移分数
        best_tag_id = torch.argmax(terminal_var).tolist()
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path
