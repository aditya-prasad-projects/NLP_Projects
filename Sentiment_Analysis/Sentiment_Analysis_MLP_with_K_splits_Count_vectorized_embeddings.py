import os
import collections

from nltk.corpus import stopwords
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

np.random.seed(42)
import string


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
    data = pos_contents + neg_contents
    data = data.split()
    unigram_counts = collections.Counter(data)
    unigram_count_most_common_tuples = unigram_counts.most_common(5000)
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

def add_labels(pos_matrix, neg_matrix):
    pos_matrix = np.c_[pos_matrix, np.full(pos_matrix.shape[0], 1)]
    neg_matrix = np.c_[neg_matrix, np.full(neg_matrix.shape[0], -1)]
    data = np.concatenate((pos_matrix, neg_matrix), axis = 0)
    return data

def build_model(data, k_split):
    np.random.shuffle(data)
    n = np.array_split(data, k_split)
    accuracy = 0
    best_accuracy = 0
    for i in range(k_split):
        data = n[i]
        X_test = data[:, :-1]
        Y_test = data[:, -1]
        counter = 0
        for j in range(k_split):
            if (j != i):
                if (counter == 0):
                    data = n[j]
                    counter += 1
                else:
                    data = np.concatenate((data, n[j]), axis=0)
        X_train = data[:, :-1]
        Y_train = data[:, -1]
        classifier = MLPClassifier(hidden_layer_sizes=(100,10), random_state=42, activation = "relu", max_iter= 400)
        classifier.fit(X_train, Y_train)
        y_pred = classifier.predict(X_test)
        temp_accuracy = accuracy_score(Y_test, y_pred)
        if(temp_accuracy > best_accuracy):
            best_accuracy = temp_accuracy
        accuracy +=temp_accuracy
    return accuracy / k_split, best_accuracy

def write_to_file(average_accuracy,maximum_accuracy):
    with open("q_3_1_result.txt", "w") as f:
        f.write("Vocabulary size = 5000\n")
        f.write("Optimal Parameters are: " + "\n")
        f.write("Number of nodes in first layer: 100" + "\n")
        f.write("Number of iterations = 200\n")
        f.write("learning_rate_init = 0.005\n")
        f.write("activation_function = relu\n\n")
        f.write("The maximum accuracy across the 10 folds is: " + str(maximum_accuracy))
        f.write("\nThe average accuracy over all the 10 folds  is: " + str(average_accuracy))


def run():
    pos_contents, neg_contents,pos_sentences_list, neg_sentences_list = read_data("../../train")
    most_common_elements = get_unigram_counts(pos_contents, neg_contents)

    pos_matrix, neg_matrix = get_matrix_form(pos_sentences_list,neg_sentences_list, most_common_elements)
    data = add_labels(pos_matrix, neg_matrix)
    average_accuracy, maximum_accuracy= build_model(data,10)
    print("accuracy = ",average_accuracy)
    #write_to_file(average_accuracy,maximum_accuracy)

run()














