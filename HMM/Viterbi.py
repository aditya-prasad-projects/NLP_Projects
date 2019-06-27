import collections
import numpy as np
import os

def read_test_data():
    sentences = []
    with open("../data/testing/Test_File.txt.html", "r") as f:
        temp = []
        contents = f.read()
        contents = contents.split("\n")
        for i in contents:
            if "< sentence ID =" in i:
                continue
            elif("<EOS>" not in i):
                temp.append(i.split()[0])
            elif("<EOS>" in i):
                sentences.append(temp)
                temp = []
    return sentences

def read_probabilities():
    transitional_probability = {}
    with open("transitional_probability.txt", "r") as f:
        for i in f.readlines():
            temp = i.split("#")
            transitional_probability[temp[0]] = float(temp[1])
    return transitional_probability

def replace_UNK(data, data_count):
    min_frequency = set()
    for key, value in data_count.items():
        if(value <=5 ):
            min_frequency.add(key)
    for i in range(len(data)):
        index = data[i].index("/")
        word_tag = data[i]
        tag = word_tag[index+1:]
        if(data[i] in min_frequency):
            data[i] = "UNK/" + tag
    return data, collections.Counter(data)

def read_unigram_data():
    with open("unigram_tags.txt", "r") as f:
        contents = f.read()
        unigram_tag = contents.split(" ")
    unigram_tag_count = collections.Counter(unigram_tag)
    temp = set()
    for key,value in unigram_tag_count.items():
        if(value <= 5):
            temp.add(key)
    for i in range(len(unigram_tag)):
        if(unigram_tag[i] in temp):
            unigram_tag.remove(unigram_tag[i])
            unigram_tag.append("UNK")
    return unigram_tag, collections.Counter(unigram_tag)

def read_word_tags():
    contents = ""
    for _, _, file in os.walk("../data/brown_corpus/training"):
        for filename in file:
            with open("../data/brown_corpus/training/" + filename, "r", encoding="latin-1") as f:
                contents = contents + " " + f.read()
    contents = contents.split()
    remove_data = []
    for t in contents:
        if "/" not in t:
            remove_data.append(t)
    for i in remove_data:
        contents.remove(i)
    return contents, collections.Counter(contents)

def get_bigram_tag_count(unigram_tags):
    bigram_tags = []
    for i in range(1, len(unigram_tags)):
        b = " ".join(unigram_tags[i - 1:i + 1])
        bigram_tags.append(b)
    return bigram_tags, collections.Counter(bigram_tags)

def calculate_emission_probability(word_tags_count, unigram_tag_count):
    length_unigram = len(unigram_tag_count)
    emission_probability = {}
    for key, value in word_tags_count.items():
        index = key.index("/")
        tag = key[index + 1:]
        word = key[:index]
        emission_probability[key] = (value + 0.0001) / (unigram_tag_count[tag] + length_unigram)
        for key1, value1 in unigram_tag_count.items():
            em_tag = word +"/"+key1
            if(em_tag in word_tags_count):
                emission_probability[em_tag] = (word_tags_count[em_tag] + 0.0001) / (value1 + length_unigram)
            else:
                emission_probability[em_tag] = 0.0001 / (value1 + length_unigram)
    return emission_probability



def get_cell_value(transitional_probability, emission_probability, tags, s, t, column_word, len_N, viterbi):
    max_value = 0
    for i in range(len_N):
        tr_tag = tags[i] + " " + tags[s]
        em_tag = column_word + "/" + tags[s]
        if(tags[i] + " " + tags[s] in transitional_probability and column_word + "/" + tags[s] in emission_probability):
            v = viterbi[i][t-1] * (0.7 * transitional_probability[tags[i] + " " + tags[s]] + 0.3 * unigram_tag_count[tags[s]]) * emission_probability[column_word + "/" + tags[s]]
        elif(tags[i] + " " + tags[s] in transitional_probability and column_word + "/" + tags[s] not in emission_probability and "UNK/" + tags[s] in emission_probability):
            v = viterbi[i][t-1] * (0.7 * transitional_probability[tags[i] + " " + tags[s]] + 0.3 * unigram_tag_count[tags[s]]) * emission_probability["UNK/" + tags[s]]
        elif(tags[i] + " " + tags[s] not in transitional_probability and column_word + "/" + tags[s] in emission_probability):
            v = viterbi[i][t-1] * 0.3 * unigram_tag_count[tags[s]] * emission_probability[column_word + "/" + tags[s]]
        elif(tags[i] + " " + tags[s] not in transitional_probability and column_word + "/" + tags[s] not in emission_probability and "UNK/" + tags[s] in emission_probability):
            v = viterbi[i][t-1] * 0.3 * unigram_tag_count[tags[s]] * emission_probability["UNK/" + tags[s]]
        else:
            v = 0
        if(v > max_value):
            max_value = v
            max_tag = tags[i]

    return max_value, max_tag




def viterbi_algorithm(test_data,transitional_probability,emission_probability, unigram_tag_count):
    tags = []
    for key, value in unigram_tag_count.items():
        tags.append(key)
    tags.remove("<s>")
    test_sentence_tags = []
    for j in range(len(test_data)):
        len_T = len(test_data[j])
        len_N = len(tags)
        viterbi = [[0]*len_T for m in range(len_N)]
        backpointer = [[0]*len_T for n in range(len_N)]
        test_sample = test_data[j]
        temp_test_tags = []
        for i in range(len_N):
            if("<s> " + tags[i] in transitional_probability and test_sample[0] + "/" + tags[i] in emission_probability):
                viterbi[i][0] = (0.7 * transitional_probability["<s> " + tags[i]] + 0.3 * (unigram_tag_count[tags[i]])) * emission_probability[test_sample[0] + "/"  +tags[i]]
            elif("<s> " + tags[i] in transitional_probability and test_sample[0] + "/" + tags[i] not in emission_probability and "UNK/" + tags[i] in emission_probability):
                viterbi[i][0] = (0.7 * transitional_probability["<s> " + tags[i]] + 0.3 * (unigram_tag_count[tags[i]])) * emission_probability["UNK/" + tags[i]]
            elif("<s> " + tags[i] not in transitional_probability and test_sample[0] + "/" + tags[i] in emission_probability):
                viterbi[i][0] = 0.3 * unigram_tag_count[tags[i]] * emission_probability[test_sample[0] + "/"  +tags[i]]
            elif("<s> " + tags[i] not in transitional_probability and test_sample[0] + "/" + tags[i] not in emission_probability and "UNK/" + tags[i] in emission_probability):
                viterbi[i][0] = 0.3 * unigram_tag_count[tags[i]] * emission_probability["UNK/" + tags[i]]
            backpointer[i][0] = 0
        for t in range(1, len_T):
            for s in range(len_N):
                viterbi[s][t], backpointer[s][t] = get_cell_value(transitional_probability, emission_probability, tags, s, t, test_sample[t], len_N, viterbi)
        viterbi_transpose = list(zip(*viterbi))
        maximum_index = viterbi_transpose[-1].index(max(viterbi_transpose[-1]))
        temp_test_tags.append(backpointer[maximum_index][-1])
        j = 0
        for k in reversed(range(1,len_T)):
            index = tags.index(temp_test_tags[j])
            temp_test_tags.append(backpointer[index][k])
            j+=1
        temp_test_tags.reverse()
        test_sentence_tags.append(temp_test_tags)
        c = []
    return test_sentence_tags


test_data = read_test_data()
unigram_tag, unigram_tag_count = read_unigram_data()
word_tags, word_tags_count = read_word_tags()
word_tags, word_tags_count = replace_UNK(word_tags,word_tags_count)
bigram_tags, bigram_tag_count = get_bigram_tag_count(unigram_tag)
transitional_probability = read_probabilities()
emission_probability = calculate_emission_probability(word_tags_count, unigram_tag_count)
length_unigram = len(unigram_tag)
for key,value in unigram_tag_count.items():
    unigram_tag_count[key] = value / length_unigram
test_sentence_tags = viterbi_algorithm(test_data,transitional_probability,emission_probability,unigram_tag_count)
print(test_sentence_tags[124])
c = []

