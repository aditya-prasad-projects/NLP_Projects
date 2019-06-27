import gensim
from text_normalizer import factory
import os
import collections
import nltk
from nltk.corpus import stopwords
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import string

np.random.seed(0)


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
            with open("../data/sentiment_analysis/train/pos/" + filename, "r", encoding="latin-1") as f:
                for line in f:
                    if not line.isspace():
                        clean_up_line = clean_up(line)
                        pos_contents += clean_up_line
                        pos_sentence_list.append(line)
    for _, _, file in os.walk(directory_name + "/neg"):
        for filename in file:
            with open("../data/sentiment_analysis/train/neg/" + filename, "r", encoding="latin-1") as f:
                for line in f:
                    if not line.isspace():
                        clean_up_line = clean_up(line)
                        neg_contents += clean_up_line
                        neg_sentence_list.append(line)
    return pos_contents, neg_contents, pos_sentence_list, neg_sentence_list


def get_unigram_counts(pos_contents, neg_contents):
    data = pos_contents + neg_contents
    data = data.split()
    unigram_counts = collections.Counter(data)
    unigram_count_most_common_tuples = unigram_counts.most_common(1000)
    temp_array = np.array(unigram_count_most_common_tuples).T[0]
    most_common_elements = (temp_array.tolist())
    return most_common_elements


def get_matrix(sentence, most_common_elements_set, most_common_elements_vectorized):
    sentence_list = sentence.split()
    temp_matrix = np.zeros(300).reshape(1, 300)
    count = 0
    for i in range(len(sentence_list)):
        if (sentence_list[i] in most_common_elements_set):
            count += 1
            temp_matrix = np.concatenate((temp_matrix, most_common_elements_vectorized[i].reshape(1, 300)), axis=0)
    tokenized = nltk.word_tokenize(sentence)
    tagged = nltk.pos_tag(tokenized)
    tags = [tag for word, tag in tagged
     if tag.startswith('NN') or tag.startswith('VB') or tag.startswith('JJ') or tag.startswith('MD') or tag.startswith('RB')]
    tag_counter = collections.Counter(tags)
    tag_most_common = tag_counter.most_common(4)
    tags_most = np.array(tag_most_common).T[0]
    tag_matrix = []
    for i in range(len(tags_most)):
        tag_matrix.append(tag_counter[tags_most[i]] / len(tags))
    # return np.mean(temp_matrix[1:, :],axis = 0)
    matrix = np.c_[temp_matrix[1:, :].mean(axis=0).reshape(1,300), np.array(tag_matrix).reshape(1,4)]
    return matrix



def add_labels(pos_matrix, neg_matrix):
    pos_matrix = np.c_[pos_matrix, np.full(pos_matrix.shape[0], 1)]
    neg_matrix = np.c_[neg_matrix, np.full(neg_matrix.shape[0], -1)]
    data = np.concatenate((pos_matrix, neg_matrix), axis=0)
    return data


def get_matrix_form(pos_sentences_list, neg_sentences_list, most_common_elements, most_common_elements_vectorized):
    pos_matrix = np.zeros(304).reshape(1, 304)
    neg_matrix = np.zeros(304).reshape(1, 304)
    most_common_elements_set = set(most_common_elements)
    for i in range(len(pos_sentences_list)):
        matrix = get_matrix(pos_sentences_list[i], most_common_elements_set,
                                                            most_common_elements_vectorized)
        pos_matrix = np.concatenate((pos_matrix, matrix), axis=0)
    for i in range(len(neg_sentences_list)):
        matrix = get_matrix(neg_sentences_list[i], most_common_elements_set,
                                                            most_common_elements_vectorized)
        neg_matrix = np.concatenate((neg_matrix, matrix), axis=0)
    return pos_matrix[1:, :], neg_matrix[1:, :]


def build_model(data):
    np.random.shuffle(data)
    X_train = data[:,:-1]
    Y_train = data[:,-1]
    classifier = MLPClassifier(hidden_layer_sizes=(200, 70, 10), random_state=42, activation = 'relu', max_iter= 400,
                                learning_rate_init=0.001, learning_rate = 'constant')
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_train)
    accuracy = accuracy_score(Y_train, y_pred)
    return accuracy


def write_to_file(most_common_elements):
    with open("./most_common_elements.txt", "w") as f:
        for i in range(len(most_common_elements)):
            f.write(most_common_elements[i] + " ")


def get_vectors_from_file(most_common_elements):
    v = np.zeros(300).reshape(1, 300)
    with open("../../most_common_elements_vectorized.txt", "r") as f:
        contents = f.read()
        contents = contents.split("$")
        for i in range(len(contents) - 1):
            temp = contents[i].split(" ")
            temp_matrix = []
            for j in range(len(temp) - 1):
                temp_matrix.append(float(temp[j]))
            v = np.concatenate((v, np.array(temp_matrix).reshape(1, 300)), axis=0)
        v = v[1:, :]
    with open("../../most_common_elements_2.txt", "r") as f:
        contents_common_elements = f.read()
        contents_common_elements = contents_common_elements.split(" ")
        contents_common_elements = contents_common_elements[:-1]
    return contents_common_elements, v


def get_vectors(most_common_elements):
    model = gensim.models.KeyedVectors.load_word2vec_format('../data/sentiment_analysis/GoogleNews-vectors-negative300.bin', binary=True)
    vocab_not_in_model = []
    vectors = np.zeros(300).reshape(1, 300)
    for i in range(len(most_common_elements)):
        if (most_common_elements[i] not in model.vocab):
            vocab_not_in_model.append(most_common_elements[i])
    for word in vocab_not_in_model:
        most_common_elements.remove(word)
    for i in range(len(most_common_elements)):
        vectors = np.concatenate((vectors, model.get_vector(most_common_elements[i]).reshape(1, 300)), axis=0)
    return most_common_elements, vectors[1:]


def run():
    pos_contents, neg_contents, pos_sentences_list, neg_sentences_list = read_data("../data/sentiment_analysis/train")
    print("done reading")
    del pos_sentences_list[82]
    del neg_sentences_list[93]
    most_common_elements = get_unigram_counts(pos_contents, neg_contents)
    most_common_elements, most_common_elements_vectorized = get_vectors_from_file(most_common_elements)
    #most_common_elements, most_common_elements_vectorized = get_vectors(most_common_elements)
    print("got common elements")
    pos_matrix, neg_matrix = get_matrix_form(pos_sentences_list, neg_sentences_list, most_common_elements,
                                             most_common_elements_vectorized)
    print("got matrix form")
    data = add_labels(pos_matrix, neg_matrix)
    accuracy = build_model(data)
    print("accuracy = ", accuracy)
    with open("./q_3_4_result.txt", "w") as f:
        f.write("The final accuracy on the whole training set with extra feature as POS tag distribution is: " + str(accuracy))


run()