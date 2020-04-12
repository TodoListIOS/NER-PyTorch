import torch


class HMM(object):
    def __init__(self, hidden_state_num, observable_state_num):
        """

        :param hidden_state_num: 隐藏状态数，这里对应存在的标注的种类
        :param observable_state_num: 可观测状态数，这里对应有多少个word
        """
        self.hidden_state_num = hidden_state_num
        self.observable_state_num = observable_state_num

        # 状态转移矩阵 transitions[i][j]表示从状态j转移到状态i的概率
        self.transitions = torch.zeros(hidden_state_num, hidden_state_num)

        # 观测概率矩阵 observable_matrices[i][j]表示tag_i是word_j的概率
        self.observable_matrices = torch.zeros(hidden_state_num, observable_state_num)

        # 初始状态概率 Pi[i]表示初始时刻为tag_i的概率
        self.Pi = torch.zeros(hidden_state_num)

    def forward(self, train_data_list):
        """
        HMM的训练，即根据训练语料对模型参数进行估计,
        因为我们有观测序列以及其对应的状态序列，所以我们
        可以使用极大似然估计的方法来估计隐马尔可夫模型的参数
        :param train_data_list:
        :return:
        """
        word_seqs = []
        tag_seqs = []
        for train_data in train_data_list:
            word_seq, tag_seq = train_data
            word_seqs.append(word_seq)
            tag_seqs.append(tag_seq)
        assert len(word_seqs) == len(tag_seqs)

        # 估计转移矩阵概率
        for tag_seq in tag_seqs:
            tag_seq_len = len(tag_seq)
            for ix in range(tag_seq_len - 1):
                from_tag_id = tag_seq[ix]
                to_tag_id = tag_seq[ix + 1]
                self.transitions[to_tag_id][from_tag_id] += 1
        # 问题：如果某元素没有出现过，该位置为0，这在后续的计算中是不允许的
        # 解决方法：我们将等于0的概率加上很小的数
        self.transitions[self.transitions == 0.] = 1e-10
        self.transitions = self.transitions / self.transitions.sum(dim=0, keepdim=True)  # 按列求平均值

        # 估计观测概率矩阵
        for tag_seq, word_seq in zip(tag_seqs, word_seqs):
            assert len(tag_seq) == len(word_seq)
            for tag, word in zip(tag_seq, word_seq):
                self.observable_matrices[tag][word] += 1
        self.observable_matrices[self.observable_matrices == 0.] = 1e-10
        self.observable_matrices = self.observable_matrices / self.observable_matrices.sum(dim=1, keepdim=True)  # 按行求平均值

        # 估计初始状态概率
        for tag_seq in tag_seqs:
            first_tag = tag_seq[0]
            self.Pi[first_tag] += 1
        self.Pi[self.Pi == 0.] = 1e-10
        self.Pi = self.Pi / self.Pi.sum()
        debug = 1

    def viterbi_decode(self, word_seq, tag_to_ix):
        """
        使用维特比算法对给定观测序列求状态序列， 这里就是对字组成的序列,求其对应的标注。
        维特比算法实际是用动态规划解隐马尔可夫模型预测问题，即用动态规划求概率最大路径（最优路径）
        这时一条路径对应着一个状态序列
        :param word_seq: 观测序列
        :param tag_to_ix:
        :return: 最优路径
        """
        # 问题:整条链很长的情况下，十分多的小概率相乘，最后可能造成下溢
        # 解决办法：采用对数概率，这样源空间中的很小概率，就被映射到对数空间的大的负数
        # 同时相乘操作也变成简单的相加操作
        transitions = torch.log(self.transitions)
        observable_matrices = torch.log(self.observable_matrices)
        Pi = torch.log(self.Pi)

        # 初始化维特比矩阵viterbi 它的维度为[状态数, 观测序列长度]
        # 其中viterbi[i, j]表示观测序列的第j个标注为i的所有单个序列(i_1, i_2, ..i_j)出现的概率最大值
        word_seq_len = len(word_seq)
        viterbi = torch.zeros(self.hidden_state_num, word_seq_len)

        # backpointer是跟viterbi一样大小的矩阵
        # backpointer[i, j]存储的是观测序列的第j个标注为i时，第j-1个标注的id
        # 等解码的时候，我们用backpointer进行回溯，以求出最优路径
        back_pointer = torch.zeros(self.hidden_state_num, word_seq_len).long()

        start_word_id = word_seq[0]

        # 转置前 observable_matrices:[tag_size * word_seq_len]
        # 转移后 t_observable_matrices: [word_seq_len * tag_size]
        # 所以 t_observable_matrices[ix]表示word_seq[ix]对应所有tag的概率
        t_observable_matrices = observable_matrices.t()  # 转置

        # 第0个word对应所有tag的概率
        start_word_to_tags_probs = t_observable_matrices[start_word_id]

        # 以标签只有'B', 'I', 'O'三种为例
        # Pi[0]为log(P(初始tag为B))
        # start_word_to_tags_probs[0]为log(P(word0|B))
        # 推导如下:
        # start_word_to_tags_probs[0] + Pi[0] = log(P(word0|'B') * P('B'|初始状态))
        #                                     = log(P(word0的标签为'B'))
        # start_word_to_tags_probs[1] + Pi[1] = log(P(word1|'I') * P('I'|初始状态))
        #                                     = log(P(word1的标签为'I'))
        # start_word_to_tags_probs[2] + Pi[2] = log(P(word2|'O') * P('O'|初始状态))
        #                                     = log(P(word2的标签为'O'))
        viterbi[:, 0] = start_word_to_tags_probs + Pi  # word[0]对应的viterbi的得分
        back_pointer[:, 0] = -1

        for step in range(1, word_seq_len):
            word_id = word_seq[step]

            # word_seq[step]的对应所有tag的概率
            word_to_tags_probs = t_observable_matrices[word_id]
            for tag_id in range(len(tag_to_ix)):
                # word_seq[step - 1]转移到tag_id上的最大概率，最大概率对应的id
                max_prob, max_id = torch.max(viterbi[:, step - 1] + transitions[tag_id], dim=0)  # 一维向量，默认max为按列取

                # 最大概率max_prob可以理解为：log(Pmax(word_seq[step]的tag_id标签|前序节点word_seq[step - 1]))
                #   max_prob + word_tags_probs[tag_id]
                # = log(P(word_seq[step]|word_seq[step]的tag_id标签) * Pmax(word_seq[step]的tag_id标签|前序节点word_seq[step - 1]))
                # = log(Pmax(word_seq[step]的标签为tag_id标签))
                viterbi[tag_id, step] = max_prob + word_to_tags_probs[tag_id]
                back_pointer[tag_id, step] = max_id

        # 终止
        best_path_score, best_path_id = torch.max(viterbi[:, word_seq_len - 1], dim=0)  # 一维向量取最大值，默认dim=0 按列取

        # 回溯
        best_path_id = best_path_id.item()
        best_path = [best_path_id]
        for back_step in range(word_seq_len - 1, 0, -1):
            best_path_id = back_pointer[best_path_id, back_step]
            best_path_id = best_path_id.item()
            best_path.append(best_path_id)

        assert len(best_path) == word_seq_len
        best_path = [best_path_id for best_path_id in reversed(best_path)]
        return best_path
