import os
import collections
from HMM.Preprocess import Preprocess
import numpy as np
import random
import sys

class HMM:
    def __init__(self, directory = "../data/brown_corpus/training", replace_minimum_occurences  =  True, minimum_occurences = 5, replace_character  = "UNK/"):
        self.directory = directory
        self.replace_minimum_occurences = replace_minimum_occurences
        self.minimum_occurences = minimum_occurences
        self.replace_character = replace_character

    def run(self):
        self.training_data = Preprocess(self.directory, self.replace_minimum_occurences,
                                        self.minimum_occurences, self.replace_character).read_data()
        word_tag_count = self.__get_count(self.training_data)
        word, unigram_tags, unigram_tag_count = self.__get_unigram_tag_count(self.training_data)
        word_count = self.__get_count(word)
        bigram_tags, bigram_tag_count = self.__get_bigram_tag_count(unigram_tags)
        transitional_probability = self.__get_transitional_probability(bigram_tags, bigram_tag_count, unigram_tag_count)
        emission_probability = self.__get_emission_probability(self.training_data, word_tag_count, unigram_tag_count)
        #transitional_tags, emission_tags, probability_sentence = self.__generate_sentences(transitional_probability,emission_probability, bigram_tags,self.training_data)




    def __get_count(self, data):
        return collections.Counter(data)

    def __generate_sentences(self, transitional_probability, emission_probability, bigram_tags, training_data):
        transitional_tags, probabilities_tags = self.__get_transitional_tags(transitional_probability, bigram_tags)
        emission_tags, emission_probability_tags = self.__get_emission_tags(emission_probability, transitional_tags,
                                                                     training_data)
        probability_sentence = []
        for i in range(len(transitional_tags)):
            prob = 0
            for j in range(len(transitional_tags[i])):
                prob += -np.log2(probabilities_tags[i][j]) + (-np.log2(emission_probability_tags[i][j]))
            probability_sentence.append(prob)
        return transitional_tags, emission_tags, probability_sentence

    def __get_emission_tags(self, emission_probability, transitional_tags, training_data):
        words = []
        probabilities = []
        for tags in transitional_tags:
            temp_word = []
            temp_probs = []
            for i in range(len(tags)):
                if i == 0:
                    word, prob = self.__get_best_word(emission_probability, training_data, tags[i])
                else:
                    word, prob = self.__get_best_word(emission_probability, training_data, tags[i], tags[i - 1])
                temp_probs.append(prob)
                temp_word.append(word)
            words.append(temp_word)
            probabilities.append(temp_probs)
        return words, probabilities

    def __get_best_word(self, emission_probability, training_data, tag, tag1=None):
        temp = []
        for i in range(len(training_data)):
            if tag in training_data[i] and tag1 == training_data[i - 1]:
                temp.append((training_data[i]))
            elif tag in training_data[i] and tag1 == None:
                temp.append(training_data[i])
        if (not temp):
            for i in range(len(training_data)):
                if tag in training_data[i]:
                    temp.append(training_data[i])

        chosen_word_tag = random.choice(temp)
        index = chosen_word_tag.index("/")
        chosen_word = chosen_word_tag[:index]
        return chosen_word, emission_probability[chosen_word_tag]

    def __get_transitional_tags(self, transitional_probability, bigram_tags):
        i = 0
        word_tags = []
        probabilities = []
        while (i < 5):
            previous_tag = "<s>"
            temp_tags = []
            temp_probs = []
            while (previous_tag != "<e>"):
                temp_transitional_probabilities = self.__get_temp_transitional_probabilities(transitional_probability,
                                                                                      previous_tag)

                previous_tag, probability = self.__get_best_tag(temp_transitional_probabilities, bigram_tags)
                temp_tags.append(previous_tag)
                temp_probs.append(probability)
            word_tags.append(temp_tags)
            probabilities.append(temp_probs)
            i += 1
        return word_tags, probabilities

    def __get_transitional_probability(self,  bigram_tags, bigram_tag_count, unigram_tag_count):
        transitional_probability = {}
        for i in range(len(bigram_tags)):
            unigram_tags = bigram_tags[i].split()
            transitional_probability[bigram_tags[i]] = bigram_tag_count[bigram_tags[i]] / unigram_tag_count[
                unigram_tags[0]]
        return transitional_probability

    def __get_emission_probability(self, training_data, word_tag_count, unigram_tag_count):
        emission_probability = {}
        for t in (training_data):
            index = t.index("/")
            tag = t[index + 1:]
            emission_probability[t] = word_tag_count[t] / unigram_tag_count[tag]
        return emission_probability

    def __get_best_tag(self, temp_transitional_probabilities, bigram_tags):
        temp_tags = []
        for t in bigram_tags:
            if t in temp_transitional_probabilities:
                temp_tags.append(t)
        tag = random.choice(temp_tags)
        probability = temp_transitional_probabilities[tag]
        tags = tag.split()
        return tags[1], probability

    def __get_temp_transitional_probabilities(self, transitional_probability, previous_tag):
        temp_transitional_probabilities = {}
        for key, value in transitional_probability.items():
            tags = key.split()
            if (tags[0] == previous_tag):
                temp_transitional_probabilities[key] = value
        return temp_transitional_probabilities

    def __get_bigram_tag_count(self, unigram_tags):
        bigram_tags = []
        for i in range(1, len(unigram_tags)):
            b = " ".join(unigram_tags[i - 1:i + 1])
            bigram_tags.append(b)
        return bigram_tags, collections.Counter(bigram_tags)

    def __get_unigram_tag_count(self, training_data):
        unigram_tags = []
        word = []
        for t in training_data:
            index = t.index("/")
            unigram_tags.append(t[index + 1:])
            word.append(t[:index])

        return word, unigram_tags, collections.Counter(unigram_tags)

hmm= HMM()
hmm.run()

