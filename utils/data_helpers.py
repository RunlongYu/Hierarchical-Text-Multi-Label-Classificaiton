# -*- coding:utf-8 -*-
__author__ = 'Randolph'

import os
import multiprocessing
import gensim
import logging
import json
import numpy as np

from collections import OrderedDict
from pylab import *
from gensim.models import word2vec
from tflearn.data_utils import pad_sequences

TEXT_DIR = '../data/content.txt'
METADATA_DIR = '../data/metadata.tsv'


def logger_fn(name, input_file, level=logging.INFO):
    tf_logger = logging.getLogger(name)
    tf_logger.setLevel(level)
    log_dir = os.path.dirname(input_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(input_file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    tf_logger.addHandler(fh)
    return tf_logger


def create_prediction_file(output_file, data_id, all_labels, all_predict_labels, all_predict_values):
    """
    Create the prediction file.

    Args:
        output_file: The all classes predicted scores provided by network
        data_id: The data record id info provided by class Data
        all_labels: The all origin labels
        all_predict_labels: The all predict labels by threshold
        all_predict_values: The all predict values by threshold
    Raises:
        IOError: If the prediction file is not a .json file
    """
    if not output_file.endswith('.json'):
        raise IOError("✘ The prediction file is not a json file."
                      "Please make sure the prediction data is a json file.")
    with open(output_file, 'w') as fout:
        data_size = len(all_predict_labels)
        for i in range(data_size):
            predict_labels = [int(i) for i in all_predict_labels[i]]
            predict_values = [round(i, 4) for i in all_predict_values[i]]
            labels = [int(i) for i in all_labels[i]]
            data_record = OrderedDict([
                ('testid', data_id[i]),
                ('labels', labels),
                ('predict_labels', predict_labels),
                ('predict_values', predict_values)
            ])
            fout.write(json.dumps(data_record, ensure_ascii=False) + '\n')


def get_label_using_scores_by_threshold(scores, threshold=0.5):
    """
    Get the predicted labels based on the threshold.
    If there is no predict value greater than threshold, then choose the label which has the max predict value.

    Args:
        scores: The all classes predicted scores provided by network
        threshold: The threshold (default: 0.5)
    Returns:
        predicted_labels: The predicted labels
        predicted_values: The predicted values
    """
    predicted_labels = []
    predicted_values = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        count = 0
        index_list = []
        value_list = []
        for index, predict_value in enumerate(score):
            if predict_value > threshold:
                index_list.append(index)
                value_list.append(predict_value)
                count += 1
        if count == 0:
            index_list.append(score.index(max(score)))
            value_list.append(max(score))
        predicted_labels.append(index_list)
        predicted_values.append(value_list)
    return predicted_labels, predicted_values


def get_label_using_scores_by_topk(scores, top_num=1):
    """
    Get the predicted labels based on the topK number.

    Args:
        scores: The all classes predicted scores provided by network
        top_num: The max topK number (default: 5)
    Returns:
        The predicted labels
    """
    predicted_labels = []
    predicted_values = []
    scores = np.ndarray.tolist(scores)
    for score in scores:
        value_list = []
        index_list = np.argsort(score)[-top_num:]
        index_list = index_list[::-1]
        for index in index_list:
            value_list.append(score[index])
        predicted_labels.append(np.ndarray.tolist(index_list))
        predicted_values.append(value_list)
    return predicted_labels, predicted_values


def cal_metric(predicted_labels, labels):
    """
    Calculate the metric(recall, precision).

    Args:
        predicted_labels: The predicted_labels
        labels: The true labels
    Returns:
        The value of metric
    """
    label_no_zero = []
    for index, label in enumerate(labels):
        if int(label) == 1:
            label_no_zero.append(index)
    count = 0
    for predicted_label in predicted_labels:
        if int(predicted_label) in label_no_zero:
            count += 1
    recall = count / len(label_no_zero)
    precision = count / len(predicted_labels)
    return recall, precision


def cal_F(recall, precision):
    """
    Calculate the metric F value.

    Args:
        recall: The recall value
        precision: The precision value
    Returns:
        The F value
    """
    F = 0.0
    if (recall + precision) == 0:
        F = 0.0
    else:
        F = (2 * recall * precision) / (recall + precision)
    return F


def create_metadata_file(embedding_size, output_file=METADATA_DIR):
    """
    Create the metadata file based on the corpus file(Use for the Embedding Visualization later).

    Args:
        embedding_size: The embedding size
        output_file: The metadata file (default: 'metadata.tsv')
    Raises:
        IOError: If word2vec model file doesn't exist
    """
    word2vec_file = '../data/word2vec_' + str(embedding_size) + '.model'

    if not os.path.isfile(word2vec_file):
        raise IOError("✘ The word2vec file doesn't exist."
                      "Please use function <create_vocab_size(embedding_size)> to create it!")

    model = gensim.models.Word2Vec.load(word2vec_file)
    word2idx = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    word2idx_sorted = [(k, word2idx[k]) for k in sorted(word2idx, key=word2idx.get, reverse=False)]

    with open(output_file, 'w+') as fout:
        for word in word2idx_sorted:
            if word[0] is None:
                print("Empty Line, should replaced by any thing else, or will cause a bug of tensorboard")
                fout.write('<Empty Line>' + '\n')
            else:
                fout.write(word[0] + '\n')


def create_word2vec_model(embedding_size, input_file=TEXT_DIR):
    """
    Create the word2vec model based on the given embedding size and the corpus file.

    Args:
        embedding_size: The embedding size
        input_file: The corpus file
    """
    word2vec_file = '../data/word2vec_' + str(embedding_size) + '.model'

    sentences = word2vec.LineSentence(input_file)
    # sg=0 means use CBOW model(default); sg=1 means use skip-gram model.
    model = gensim.models.Word2Vec(sentences, size=embedding_size, min_count=0,
                                   sg=0, workers=multiprocessing.cpu_count())
    model.save(word2vec_file)


def load_vocab_size(embedding_size):
    """
    Return the vocab size of the word2vec file.

    Args:
        embedding_size: The embedding size
    Returns:
        The vocab size of the word2vec file
    Raises:
        IOError: If word2vec model file doesn't exist
    """
    word2vec_file = '../data/word2vec_' + str(embedding_size) + '.model'

    if not os.path.isfile(word2vec_file):
        raise IOError("✘ The word2vec file doesn't exist."
                      "Please use function <create_vocab_size(embedding_size)> to create it!")

    model = word2vec.Word2Vec.load(word2vec_file)
    return len(model.wv.vocab.items())


def data_word2vec(input_file, num_classes_list, word2vec_model):
    """
    Create the research data tokenindex based on the word2vec model file.
    Return the class Data(includes the data tokenindex and data labels).

    Args:
        input_file: The research data
        num_classes_list: <list> The number of classes
        word2vec_model: The word2vec model file
    Returns:
        The class Data(includes the data tokenindex and data labels)
    Raises:
        IOError: If the input file is not the .json file
    """
    num_classes_list = list(map(int, num_classes_list.split(',')))
    vocab = dict([(k, v.index) for (k, v) in word2vec_model.wv.vocab.items()])

    def _token_to_index(content):
        result = []
        for item in content:
            word2id = vocab.get(item)
            if word2id is None:
                word2id = 0
            result.append(word2id)
        return result

    def _create_onehot_labels(labels_index, num_labels):
        label = [0] * num_labels
        for item in labels_index:
            label[int(item)] = 1
        return label

    if not input_file.endswith('.json'):
        raise IOError("✘ The research data is not a json file. "
                      "Please preprocess the research data into the json file.")
    with open(input_file) as fin:
        id_list = []
        title_index_list = []
        abstract_index_list = []
        labels_list = []
        onehot_labels_list = []

        labels_bind_list = []
        total_line = 0
        for eachline in fin:
            data = json.loads(eachline)
            patent_id = data['id']
            title_content = data['title']
            abstract_content = data['abstract']
            first_labels = data['section']
            second_labels = data['subsection']
            third_labels = data['group']

            id_list.append(patent_id)
            # title_index_list.append(_token_to_index(title_content))
            abstract_index_list.append(_token_to_index(abstract_content))
            labels_list.append(third_labels)

            labels_tuple = (_create_onehot_labels(first_labels, num_classes_list[0]),
                            _create_onehot_labels(second_labels, num_classes_list[1]),
                            _create_onehot_labels(third_labels, num_classes_list[2]))

            onehot_labels_list.append(labels_tuple)

            if 'labels_bind' in data.keys():
                labels_bind_list.append(data['labels_bind'])

            total_line += 1

    class _Data:
        def __init__(self):
            pass

        @property
        def number(self):
            return total_line

        @property
        def patent_id(self):
            return id_list

        @property
        def title_tokenindex(self):
            return title_index_list

        @property
        def abstract_tokenindex(self):
            return abstract_index_list

        @property
        def labels(self):
            return labels_list

        @property
        def onehot_labels(self):
            return onehot_labels_list

        @property
        def labels_bind(self):
            if labels_bind_list:
                return labels_bind_list
            else:
                return None

    return _Data()


def data_augmented(data, drop_rate=1.0):
    """
    Data augmented.

    Args:
        data: The Class Data()
        drop_rate: The drop rate
    Returns:
        aug_data
    """
    aug_num = data.number
    aug_patent_id = data.patent_id
    aug_title_tokenindex = data.title_tokenindex
    aug_abstract_tokenindex = data.abstract_tokenindex
    aug_labels = data.labels
    aug_onehot_labels = data.onehot_labels

    if data.labels_bind:
        aug_labels_bind = data.labels_bind
    else:
        aug_labels_bind = None

    for i in range(len(data.aug_abstract_tokenindex)):
        data_record = data.tokenindex[i]
        if len(data_record) == 1:  # 句子长度为 1，则不进行增广
            continue
        elif len(data_record) == 2:  # 句子长度为 2，则交换两个词的顺序
            data_record[0], data_record[1] = data_record[1], data_record[0]
            aug_patent_id.append(data.patent_id[i])
            aug_title_tokenindex.append(data.title_tokenindex[i])
            aug_abstract_tokenindex.append(data_record)
            aug_labels.append(data.labels[i])
            aug_onehot_labels.append(data.onehot_labels[i])

            if data.labels_bind:
                aug_labels_bind.append(data.labels_bind[i])
            else:
                aug_labels_bind = None

            aug_num += 1
        else:
            data_record = np.array(data_record)
            for num in range(len(data_record) // 10):  # 打乱词的次数，次数即生成样本的个数；次数根据句子长度而定
                # random shuffle & random drop
                data_shuffled = np.random.permutation(np.arange(int(len(data_record) * drop_rate)))
                new_data_record = data_record[data_shuffled]

                aug_patent_id.append(data.patent_id[i])
                aug_title_tokenindex.append(data.title_tokenindex[i])
                aug_abstract_tokenindex.append(list(new_data_record))
                aug_labels.append(data.labels[i])
                aug_onehot_labels.append(data.onehot_labels[i])

                if data.labels_bind:
                    aug_labels_bind.append(data.labels_bind[i])
                else:
                    aug_labels_bind = None

                aug_num += 1

    class _AugData:
        def __init__(self):
            pass

        @property
        def number(self):
            return aug_num

        @property
        def patent_id(self):
            return aug_patent_id

        @property
        def title_tokenindex(self):
            return aug_title_tokenindex

        @property
        def abstract_tokenindex(self):
            return aug_abstract_tokenindex

        @property
        def labels(self):
            return aug_labels

        @property
        def onehot_labels(self):
            return aug_onehot_labels

        @property
        def labels_bind(self):
            return aug_labels_bind

    return _AugData()


def load_word2vec_matrix(vocab_size, embedding_size):
    """
    Return the word2vec model matrix.

    Args:
        vocab_size: The vocab size of the word2vec model file
        embedding_size: The embedding size
    Returns:
        The word2vec model matrix
    Raises:
        IOError: If word2vec model file doesn't exist
    """
    word2vec_file = '../data/word2vec_' + str(embedding_size) + '.model'

    if not os.path.isfile(word2vec_file):
        raise IOError("✘ The word2vec file doesn't exist. "
                      "Please use function <create_vocab_size(embedding_size)> to create it!")
    model = gensim.models.Word2Vec.load(word2vec_file)
    vocab = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    vector = np.zeros([vocab_size, embedding_size])
    for key, value in vocab.items():
        if key is not None:
            vector[value] = model[key]
    return vector


def load_data_and_labels(data_file, num_classes_list, embedding_size, data_aug_flag):
    """
    Load research data from files, splits the data into words and generates labels.
    Return split sentences, labels and the max sentence length of the research data.

    Args:
        data_file: The research data
        num_classes_list: <list> The number of classes
        embedding_size: The embedding size
        data_aug_flag: The flag of data augmented
    Returns:
        The class Data
    """
    word2vec_file = '../data/word2vec_' + str(embedding_size) + '.model'

    # Load word2vec model file
    if not os.path.isfile(word2vec_file):
        create_word2vec_model(embedding_size, TEXT_DIR)

    model = word2vec.Word2Vec.load(word2vec_file)

    # Load data from files and split by words
    data = data_word2vec(data_file, num_classes_list, word2vec_model=model)
    if data_aug_flag:
        data = data_augmented(data)

    # plot_seq_len(data_file, data)

    return data


def pad_data(data, pad_seq_len):
    """
    Padding each sentence of research data according to the max sentence length.
    Return the padded data and data labels.

    Args:
        data: The research data
        pad_seq_len: The max sentence length of research data
    Returns:
        pad_seq: The padded data
        labels: The data labels
    """
    abstract_pad_seq = pad_sequences(data.abstract_tokenindex, maxlen=pad_seq_len, value=0.)
    onehot_labels_list = data.onehot_labels
    return abstract_pad_seq, onehot_labels_list


def plot_seq_len(data_file, data, percentage=0.98):
    """
    Visualizing the sentence length of each data sentence.

    Args:
        data_file: The data_file
        data: The class Data (includes the data tokenindex and data labels)
        percentage: The percentage of the total data you want to show
    """
    data_analysis_dir = '../data/data_analysis/'
    if 'train' in data_file.lower():
        output_file = data_analysis_dir + 'Train Sequence Length Distribution Histogram.png'
    if 'validation' in data_file.lower():
        output_file = data_analysis_dir + 'Validation Sequence Length Distribution Histogram.png'
    if 'test' in data_file.lower():
        output_file = data_analysis_dir + 'Test Sequence Length Distribution Histogram.png'
    result = dict()
    for x in data.abstract_tokenindex:
        if len(x) not in result.keys():
            result[len(x)] = 1
        else:
            result[len(x)] += 1
    freq_seq = [(key, result[key]) for key in sorted(result.keys())]
    x = []
    y = []
    avg = 0
    count = 0
    border_index = []
    for item in freq_seq:
        x.append(item[0])
        y.append(item[1])
        avg += item[0] * item[1]
        count += item[1]
        if count > data.number * percentage:
            border_index.append(item[0])
    avg = avg / data.number
    print('The average of the data sequence length is {0}'.format(avg))
    print('The recommend of padding sequence length should more than {0}'.format(border_index[0]))
    xlim(0, 400)
    plt.bar(x, y)
    plt.savefig(output_file)
    plt.close()


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    含有 yield 说明不是一个普通函数，是一个 Generator.
    函数效果：对 data，一共分成 num_epochs 个阶段（epoch），在每个 epoch 内，如果 shuffle=True，就将 data 重新洗牌，
    批量生成 (yield) 一批一批的重洗过的 data，每批大小是 batch_size，一共生成 int(len(data)/batch_size)+1 批。

    Args:
        data: The data
        batch_size: The size of the data batch
        num_epochs: The number of epochs
        shuffle: Shuffle or not (default: True)
    Returns:
        A batch iterator for data set
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
