import os
import collections
import nltk
from nltk.corpus import stopwords
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import copy
np.random.seed(42)
import string
from nltk import word_tokenize

from text_normalizer import factory

def clean_up(line):
    a = factory.EngLowercase()
    lines = a.lowercase(line)
    stopset = set(stopwords.words('english')) | set(string.punctuation)
    cleanup = " ".join(filter(lambda word: word not in stopset, lines.split()))
    return cleanup


def read_data(directory_name):
    pos_contents = ""
    neg_contents = ""
    pos_sentence_list = []
    neg_sentence_list = []
    for _, _, file in os.walk(directory_name + "/pos"):
        for filename in file:
            with open("../data/sentiment_analysis/train/pos/" + filename, "r", encoding = "latin-1") as f:
                for line in f:
                    if not line.isspace():
                        clean_up_line = clean_up(line)
                        pos_contents += clean_up_line
                        pos_sentence_list.append(line)
    for _, _, file in os.walk(directory_name + "/neg"):
        for filename in file:
            with open("../data/sentiment_analysis/train/neg/" + filename, "r", encoding = "latin-1") as f:
                for line in f:
                    if not line.isspace():
                        clean_up_line = clean_up(line)
                        neg_contents += clean_up_line
                        neg_sentence_list.append(line)
    return pos_contents, neg_contents, pos_sentence_list,neg_sentence_list

def get_unigram_counts(pos_contents, neg_contents):
    unigram_count_most_common = {}
    data = pos_contents + neg_contents
    data = data.split()
    unigram_counts = collections.Counter(data)
    unigram_count_most_common_tuples = unigram_counts.most_common(10000)
    temp_array = np.array(unigram_count_most_common_tuples).T[0]
    most_common_elements = (temp_array.tolist())
    # for key, value in unigram_counts.items():
    #     if key in most_common_elements_list:
    #         unigram_count_most_common[key] = value
    return most_common_elements

def get_matrix(sentence, most_common_elements, most_common_elements_set):
    sentence_list = sentence.split()
    matrix = np.zeros(len(most_common_elements))
    for word in sentence_list:
        if(word in most_common_elements_set):
            index = most_common_elements.index(word)
            matrix[index] += 1
    return matrix

def get_matrix_form(pos_sentences_list,neg_sentences_list, most_common_elements):
    pos_matrix = []
    neg_matrix = []
    most_common_elements_set = set(most_common_elements)
    for i in range(len(pos_sentences_list)):
        pos_matrix.append(get_matrix(pos_sentences_list[i], most_common_elements, most_common_elements_set))
    for i in range(len(neg_sentences_list)):
        neg_matrix.append(get_matrix(neg_sentences_list[i], most_common_elements, most_common_elements_set))
    return np.array(pos_matrix), np.array(neg_matrix)

def get_test_matrix(test_sentences_list, most_common_elements):
    test_sentence = []
    most_common_elements_set = set(most_common_elements)
    for i in range(len(test_sentences_list)):
        test_sentence.append(get_matrix(test_sentences_list[i], most_common_elements, most_common_elements_set))
    return np.array(test_sentence)


def add_labels(pos_matrix, neg_matrix):
    pos_matrix = np.c_[pos_matrix, np.full(pos_matrix.shape[0], 1)]
    neg_matrix = np.c_[neg_matrix, np.full(neg_matrix.shape[0], -1)]
    data = np.concatenate((pos_matrix, neg_matrix), axis = 0)
    return data

def build_model(data, test_data, file_list):
    np.random.shuffle(data)
    X_train = data[:, :-1]
    Y_train = data[:, -1]

    classifier = MLPClassifier(hidden_layer_sizes=(100,10), random_state=42, activation = 'relu', max_iter= 400,
                                learning_rate_init=0.001)
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_train)
    y_pred_test = classifier.predict(test_data)
    return accuracy_score(Y_train, y_pred)

def read_test_data(directory_name):
    sentence_list = []
    file_list = []
    sentences = ""
    for _, _, file in os.walk(directory_name):
        for filename in file:
            with open("../data/sentiment_analysis/test/" + filename, "r", encoding = "latin-1") as f:
                for line in f:
                    if not line.isspace():
                        clean_up_line = clean_up(line)
                        sentences += clean_up_line
                        sentence_list.append(line)
                file_list.append(filename)
    return sentences, sentence_list, file_list

def run():
    pos_contents, neg_contents,pos_sentences_list, neg_sentences_list = read_data("../data/sentiment_analysis/train")
    test_sentences, test_sentences_list, file_list = read_test_data("../data/sentiment_analysis/test")
    most_common_elements = get_unigram_counts(pos_contents, neg_contents)
    pos_matrix, neg_matrix = get_matrix_form(pos_sentences_list,neg_sentences_list, most_common_elements)
    test_data = get_test_matrix(test_sentences_list, most_common_elements)
    data = add_labels(pos_matrix, neg_matrix)
    accuracy = build_model(data, test_data, file_list)
    print(accuracy)
run()














