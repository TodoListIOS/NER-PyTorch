import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(self,
                 tag_to_ix,
                 batch_size,
                 vocab_size,  # len(word_to_ix)
                 embedding_dim,  # 句中每个word的特征维度
                 hidden_dim,
                 dropout=0.5,
                 use_gpu=False,  # 没有添加GPU
                 use_crf=False  # 纯BiLSTM网络，不使用crf
                 ):
        super(BiLSTM, self).__init__()
        self.tag_to_ix = tag_to_ix  #
        self.batch_size = batch_size
        self.vocab_size = vocab_size  #
        self.embedding_dim = embedding_dim  #
        self.hidden_dim = hidden_dim  #
        self.dropout = dropout
        self.use_gpu = use_gpu
        self.use_crf = use_crf

        self.tag_size = len(self.tag_to_ix)

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True, dropout=self.dropout)

        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size)

        self.hidden = self.init_hidden()

        # if use_crf:
        #     self.transitions = nn.Parameter(torch.zeros(self.tag_size, self.tag_size))
        #     self.transitions.data[self.tag_to_idx[START], :] = -10000.0
        #     self.transitions.data[:, self.tag_to_idx[END]] = -10000.0

    def init_hidden(self):
        return (torch.randn(2, self.batch_size, self.hidden_dim // 2),
                torch.randn(2, self.batch_size, self.hidden_dim // 2))

    def _get_lstm_features(self, sentences):
        self.hidden = self.init_hidden()
        length = sentences.shape[1]  # sentence:torch.Size([batch_size, max_sentence_len])  length为最长数据的长度

        embeddings = self.word_embeddings(sentences).view(self.batch_size, length, self.embedding_dim)

        # 当lstm的batch_first=True时
        # input tensor的输入形式变成[batch_size, seq_len, input_size] input_size为每个word的特征维度(embedding_dim)
        lstm_out, self.hidden = self.lstm(embeddings, self.hidden)

        lstm_out = lstm_out.view(self.batch_size, -1, self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def neg_log_likelihood(self, sentences, tags, lengths):
        self.batch_size = sentences.size(0)

        featss = self._get_lstm_features(sentences)  # 纯BiLSTM的输出 featss的shape: torch.Size([batch_size, max_sentence_len, tagset_size])
        scores = torch.zeros(1)  # scores=tensor([0.])

        # if self.use_crf:
        #     forward_score = self._forward_alg(feats)
        #     score = self._score_sentence(feats, tags)
        #     return forward_score - score
        # else:
        #     scores = nn.functional.cross_entropy(feats, tags)
        #     return scores

        for feats, tag in zip(featss, tags):
            # feats shape:torch.Size([max_sentence_len, tagset_size])
            # tag shape: max_sentence_len
            scores += nn.functional.cross_entropy(feats, tag)
            # 例:score: tensor(2.4926, grad_fn=<NllLossBackward>)
        return scores

    def forward(self, sentences, lengths=None):
        """
        前向传播
        :param sentences: {tuple: batch_size} [batch_size * max_sentence_length]
        :param lengths: [sentence0填充前的长度, sentence1填充前的长度, ...]
        :return:
        """
        sentences = torch.tensor(sentences, dtype=torch.long)  # torch.Size([batch_size, max_sentence_len])
        if not lengths:
            lengths = [i.size(-1) for i in sentences]
        self.batch_size = sentences.size(0)
        featss = self._get_lstm_features(sentences)  # 一批句子的feats, shape: torch.Size([batch_size, max_sentence_len, tagset_size])
        scores = []
        paths = []
        for feats, length in zip(featss, lengths):
            # feats: shape:torch.Size([max_sentence_len, tagset_size])
            # length: int
            feats = feats[:length]  # 取有效值
            score, path = torch.max(feats, dim=1)  # 按行取最大值
            # 例score tensor([float0, float2, float3, float4, ...])
            # 例 path [tensor(int0), tensor(int1), tensor(int2)] (Note: int0表示第0个整数)

            path = path.cpu().data.tolist()
            scores.append(score)
            paths.append(path)

        # if self.use_crf:
        #     score, tag_seq = self.viterbi_decode(feats)
        # else:
        #     score, tag_seq = torch.max(feats, 1)
        #     tag_seq = list(tag_seq.cpu().data)
        return scores, paths
