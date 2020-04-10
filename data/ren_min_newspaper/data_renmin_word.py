"""
if tag == 'nr' or tag == 'ns' or tag == 'nt':  # 如果tag是人名 地名 机构团体中的
在人民日报的文件中总共有3个实体类：nr 人名, ns 地名, nt 机构团体名, O 什么实体也不是
"""
import codecs
import re
import pdb
import pandas as pd
import numpy as np
import collections
import pickle

from sklearn.model_selection import train_test_split


def origin_handle():
    line_num = 0
    with open('./renmin.txt', 'r') as inp, open('./renmin2.txt', 'w') as outp:
        for line in inp.readlines():
            line = line.split('  ')  # txt文档中word/tag的组合是以两个空格分隔的
            i = 1  # 第0个word/tag的组合是日期，就忽略了，所以从1开始算起
            while i < len(line) - 1:  # 每一句话最后一个字符都是\n，所以处理line的时候不处理最后一个
                if line[i][0] == '[':  # 如果line中第i个单词的第0个字符是'['，表示开始一个组合词
                    outp.write(line[i].split('/')[0][1:])  # 单词部分从'['开始,把word写入文件
                    i += 1  # 继续下一个word/tag
                    while i < len(line) - 1 and line[i].find(']') == -1:  # 如果没有扫描到这个组合词的结尾
                        if line[i] != '':
                            outp.write(line[i].split('/')[0])  # 如果word/tag不为空，就把word部分写入文件
                        i += 1
                    # 扫描到了组合词的结尾('[日本/ns', '政府/n]nt')
                    outp.write(line[i].split('/')[0].strip() + '/' + line[i].split('/')[1][-2:] + ' ')  # -2表示取组合词的tag
                elif line[i].split('/')[1] == 'nr':  # 如果tag是人名
                    word = line[i].split('/')[0]  # 人名
                    i += 1  # 继续下一个word/tag
                    if i < len(line) - 1 and line[i].split('/')[1] == 'nr':  # 如果下一个word/tag的tag也代表人名('王/nr', '林昌/nr')
                        outp.write(word + line[i].split('/')[0] + '/nr ')
                    else:
                        outp.write(word + '/nr ')
                        continue
                else:
                    outp.write(line[i] + ' ')
                i += 1
            outp.write('\n')
            line_num += 1
            # if line_num != 0 and line_num % 200 == 0:
            #     print('处理完第{}行'.format(line_num))


def origin_handle2():
    with codecs.open('./renmin2.txt', 'r', 'utf-8') as inp, codecs.open('./renmin3.txt', 'w', 'utf-8') as outp:
        for line in inp.readlines():
            line = line.split(' ')
            i = 0  # 没有了表示时间的word/tag，i可以从0开始计算了
            while i < len(line) - 1:
                if line[i] == '':
                    i += 1
                    continue
                word = line[i].split('/')[0]
                tag = line[i].split('/')[1]
                if tag == 'nr' or tag == 'ns' or tag == 'nt':  # 如果tag是人名 地名 机构团体中的
                    outp.write(word[0] + "/B_" + tag + " ")
                    for j in word[1:len(word) - 1]:
                        if j != ' ':
                            outp.write(j + "/M_" + tag + " ")
                    outp.write(word[-1] + "/E_" + tag + " ")
                else:
                    for wor in word:
                        outp.write(wor + '/O ')  # 'O'表示不是nr ns nt中的任何一个
                i += 1
            outp.write('\n')


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
                # if line_num >= 2000:  # 拿2000条数据作为train和test的集合
                #     break


def flatten(x):
    result = []
    for el in x:
        if isinstance(x, collections.Iterable) and not isinstance(el, str):
            result.extend(flatten(el))
        else:
            result.append(el)
    return result


def check_list(x):
    for el in x:
        if not isinstance(el, str):
            print(el)


def my_data2pkl():
    # datas = []  # [[第一行的所有单字], [第二行的所有单字], [第三行的所有单字], .....]
    document_words = []  # [[第一行的所有word], [第二行的所有word], [第三行的所有word], .....]
    # labels = []  # [[第一行的所有tag], [第二行的所有tag], [第三行的所有tag], ...]
    document_tags = []  # [[第一行的所有tag], [第二行的所有tag], [第三行的所有tag], ...]
    # linedata = []
    # linelabel = []
    tags = set()  # 存储整个文本的所有tag
    # tags.add('')
    input_data = codecs.open('renmin4.txt', 'r', 'utf-8')
    line_num = 0

    for line in input_data.readlines():
        # line = line.split()  # 把一行文本切成单个字/tag的形式
        # 把一行文本切成单个字/tag的形式
        list_of_word_tag = line.split()  # ['迈/O', '向/O', '充/O', '满/O', '希/O', '望/O', '的/O', '新/O', '世/O', '纪/O', ...]

        # linedata = []  # 存放所有的单字
        # linelabel = []  # 存放所有的标签
        list_of_word_in_line = []  # 存放所有的word
        list_of_tag_in_line = []  # 存放所有的tag
        numNotO = 0  # 一行文本中非'O'的实体的数量
        for word_tag in list_of_word_tag:
            # word = word.split('/')
            _word = word_tag.split('/')[0]
            _tag = word_tag.split('/')[1]
            # linedata.append(word[0])
            # linelabel.append(word[1])
            list_of_word_in_line.append(_word)  # 把分离出来的一个word放入存储每行word的列表
            list_of_tag_in_line.append(_tag)  # # 把分离出来的一个tag放入存储每行tag的列表
            # tags.add(word[1])  # 存放所有tag的列表中加入该字对应的tag
            tags.add(_tag)  # 存放整个文本tag的set中加入分离出来的tag
            # if word[1] != 'O':
            #     numNotO += 1
            if _tag != 'O':  # 如果word对应的tag不是'O'
                numNotO += 1
        if numNotO != 0:  # 如果一行文本中存在实体
            # datas.append(linedata)
            document_words.append(list_of_word_in_line)  # 把本行的word列表加入存放整个文本word的列表
            # labels.append(linelabel)
            document_tags.append(list_of_tag_in_line)  # # 把本行的tag列表加入存放整个文本tag的列表
        line_num += 1

    input_data.close()
    print('document_words的长度是{}'.format(len(document_words)))  # 638 使用完整的renmin4.txt时是37924行
    print('document_tags的长度是{}'.format(len(document_tags)))  # 638 使用完整的renmin4.txt时是37924行
    print('renmin4.txt总共有{}行'.format(line_num))  # 2172
    print('所有文本中出现过的tag有 {}'.format(str(tags)))  # {'M_nr', 'M_ns', 'E_ns', 'B_nt', 'B_nr', 'M_nt', 'E_nt', 'O', 'B_ns', 'E_nr'}

    word_to_ix = {}  # {'中': 0, '共': 1, '央': 2, ......}
    for words in document_words:
        # words为列表，存放每行的word
        for word in words:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    print('字典形式的word_to_ix的长度是{}'.format(len(word_to_ix)))

    ix_to_word = {}
    for key, val in word_to_ix.items():
        ix_to_word[val] = key
    print('字典形式的ix_to_word的长度是{}'.format(len(ix_to_word)))

    tag_to_ix = {}  # {'B_nt': 0, 'M_nt': 1, ......}
    for tags in document_tags:
        # tags为列表，存放每行的tag
        for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)

    # 加入开始和结束标签
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    tag_to_ix[START_TAG] = len(tag_to_ix)
    tag_to_ix[STOP_TAG] = len(tag_to_ix)

    print('字典形式的tag_to_ix的长度是{}'.format(len(tag_to_ix)))
    print(tag_to_ix)

    ix_to_tag = {}
    for key, val in tag_to_ix.items():
        ix_to_tag[val] = key
    print('字典形式的ix_to_tag的长度是{}'.format(len(ix_to_tag)))
    print(ix_to_tag)

    #
    X_train, X_test, y_train, y_test = train_test_split(document_words, document_tags, test_size=0.1, random_state=1)

    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    print('训练数据的长度: {}'.format(len(X_train)))  # 选取前2000条时：574 完整的：34131
    print('测试数据的长度: {}'.format(len(X_test)))  # 选取前2000条时：64 完整的：3793

    training_data = []
    for ix in range(len(X_train)):
        list_of_word_in_line = X_train[ix]
        list_of_tag_in_line = y_train[ix]
        training_data.append((list_of_word_in_line, list_of_tag_in_line))
    print('training_data的类型是 {} 长度是 {}'.format(str(type(training_data)), str(len(training_data))))  # list, 574

    testing_data = []
    for ix in range(len(X_test)):
        list_of_word_in_line = X_test[ix]
        list_of_tag_in_line = y_test[ix]
        testing_data.append((list_of_word_in_line, list_of_tag_in_line))
    print('testing_data的类型是 {} 长度是 {}'.format(str(type(testing_data)), str(len(testing_data))))  # list, 64

    with open('../renmindata.pkl', 'wb') as outp:
        pickle.dump(word_to_ix, outp)
        pickle.dump(ix_to_word, outp)
        pickle.dump(tag_to_ix, outp)
        pickle.dump(ix_to_tag, outp)
        pickle.dump(training_data, outp)
        pickle.dump(testing_data, outp)
    print('** Finish saving data to renmindata.pkl.')


def data_to_train_dev():
    with codecs.open('./renmin4.txt', 'r', 'utf-8') as inp, \
            codecs.open('./dev', 'w', 'utf-8') as dev_out, \
            codecs.open('./train', 'w', 'utf-8') as train_out:

        word_lists = []  # 存储所有行的word:[[第0行中所有的word], [第1行中所有的word], [第2行中所有的word], .....]
        tag_lists = []  # 存储所有行的tag:[[第0行中所有的tag], [第1行中所有的tag], [第2行中所有的tag], .....]

        line_num = 0  # 所有行数
        line_with_entity_num = 0  # 包含实体的行数
        for line in inp.readlines():
            word_list = []  # 存储一行中的所有word
            tag_list = []  # 存储一行中的所有tag
            line = line.strip()
            if line == '':  # 跳过空行
                continue

            num_not_o = 0  # 一行文本中非'O'的实体的数量
            word_tag_list = line.split(' ')
            for word_tag in word_tag_list:
                try:
                    word, tag = word_tag.split('/')
                    word_list.append(word)
                    tag_list.append(tag)
                    if tag != 'O':
                        num_not_o += 1
                except:
                    pass
            if num_not_o != 0:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                line_with_entity_num += 1
            line_num += 1
        inp.close()

        print('renmin4.txt中包含实体的行有 {}/{}'.format(line_with_entity_num, line_num))  # 37924/153337
        print('word_lists的长度是:{}'.format(len(word_lists)))  # 37924
        print('tag_lists的长度是:{}'.format(len(tag_lists)))  # 37924

        word_train, word_dev, tag_train, tag_dev = train_test_split(word_lists, tag_lists, test_size=0.2,
                                                                    random_state=1)

        print('从{}条数据中划分train和dev数据集, test_size={}\n'
              '划分后train_data有{}条，dev_data有{}条'.format(line_with_entity_num, 0.2, len(word_train), len(word_dev)))
        # train_data有34131条，dev_data有3793条

        # 写入人民日报的train文件
        assert len(word_train) == len(tag_train)
        for ix in range(len(word_train)):
            word_list = word_train[ix]  # 一句话中的所有word
            tag_list = tag_train[ix]  # 一句话中所有word对应的所有tag

            assert len(word_list) == len(tag_list)
            for word, tag in zip(word_list, tag_list):
                train_out.write(word + ' ' + tag + '\n')
            train_out.write('\n')  # 两句话之间以空行作为分隔符

        # 写入人民日报的dev文件
        assert len(word_dev) == len(tag_dev)
        for ix in range(len(word_dev)):
            word_list = word_dev[ix]  # 一句话中的所有word
            tag_list = tag_dev[ix]  # 一句话中所有word对应的所有tag

            assert len(word_list) == len(tag_list)
            for word, tag in zip(word_list, tag_list):
                dev_out.write(word + ' ' + tag + '\n')
            dev_out.write('\n')  # 两句话之间以空行作为分隔符


origin_handle()
origin_handle2()
sentence2split()
# my_data2pkl()
data_to_train_dev()
