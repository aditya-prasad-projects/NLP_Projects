import pandas as pd
import numpy as np
import re
from text_normalizer import factory
from nltk.corpus import stopwords
import collections
import string
import gensim
from nltk import tokenize
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import readability

class Preprocess_Fluency:
    def __init__(self, filename, remove_digit, remove_punctuation, extra_features):
        self.file_name = filename
        self.remove_digit = remove_digit
        self.table = str.maketrans({key: None for key in string.punctuation})
        self.remove_punctuation = remove_punctuation
        self.extra_features = extra_features

    def _read_data(self):
        data = pd.read_csv("../data/Summary_Evaluation/" + self.file_name)
        dataset = data.values
        return dataset

    def preprocess(self):
        data = self._read_data()
        preprocessed_summary = self._preprocess_data(data[:, 0])
        repititive_unigram  = self._get_repititive_unigram(preprocessed_summary)
        repititive_bigram = self._get_repititive_bigram(preprocessed_summary)
        flesch_score = self._get_score(preprocessed_summary, data, "FleschReadingEase")
        features = []
        if(len(self.extra_features) != 0):
            for i in range(len(self.extra_features)):
                if(self.extra_features[i] == "RIX"):
                    rix_score = self._get_score(preprocessed_summary, data, "RIX")
                    features.append(rix_score)
                if(self.extra_features[i] == 'Kincaid'):
                    kincaid_score = self._get_score(preprocessed_summary,data,'Kincaid')
                    features.append((kincaid_score))
                if (self.extra_features[i] == 'ARI'):
                    kincaid_score = self._get_score(preprocessed_summary, data, 'ARI')
                    features.append((kincaid_score))
                if (self.extra_features[i] == 'Coleman-Liau'):
                    kincaid_score = self._get_score(preprocessed_summary, data, 'Coleman-Liau')
                    features.append((kincaid_score))
                if (self.extra_features[i] == 'GunningFogIndex'):
                    kincaid_score = self._get_score(preprocessed_summary, data, 'GunningFogIndex')
                    features.append((kincaid_score))
                if (self.extra_features[i] == 'LIX'):
                    kincaid_score = self._get_score(preprocessed_summary, data, 'LIX')
                    features.append((kincaid_score))
                if (self.extra_features[i] == 'SMOGIndex'):
                    kincaid_score = self._get_score(preprocessed_summary, data, 'SMOGIndex')
                    features.append((kincaid_score))
                if (self.extra_features[i] == 'DaleChallIndex'):
                    kincaid_score = self._get_score(preprocessed_summary, data, 'DaleChallIndex')
                    features.append((kincaid_score))
        matrix = self._build_matrix(repititive_unigram,repititive_bigram,flesch_score, (list(data[:, 2])), features)

        return matrix

    def _get_score(self, preprocessed_summary, dataset, score):
        flesch_score = []
        for i in range(len(preprocessed_summary)):
            temp_sentence_list = tokenize.sent_tokenize(dataset[i, 0])
            temp_score_list = []
            for j in range(len(temp_sentence_list)):
                if(any(c.isalpha() for c in temp_sentence_list[j])):
                    results = readability.getmeasures(temp_sentence_list[j], lang='en')
                    temp_score_list.append(results['readability grades'][score])
            flesch_score.append(min(temp_score_list))
        return np.array(flesch_score).reshape(-1,1)

    def _build_matrix(self, repititive_unigram,repititive_bigram,flesch_score, fluency_score, features):
        if(len(self.extra_features) != 0     and features):
            matrix = np.c_[repititive_unigram,repititive_bigram,flesch_score]
            for i in range(len(features)):
                matrix = np.c_[matrix, features[i]]
            matrix = np.c_[matrix, fluency_score]
        else:
            matrix = np.c_[repititive_unigram,repititive_bigram,flesch_score,fluency_score]
        return matrix



    def _get_repititive_unigram(self, preprocessed_summary):
        repititive_count = []
        for i in range(len(preprocessed_summary)):
            temp_list = preprocessed_summary[i].split()
            unigram_count = collections.Counter(temp_list)
            repititive_count.append(float(sum(1 for i in unigram_count.values() if i > 1)))
        return np.array(repititive_count).reshape(-1,1)

    def _get_repititive_bigram(self,preprocessed_summary):
        repititive_count = []
        for i in range(len(preprocessed_summary)):
            temp_list = preprocessed_summary[i].split()
            bigram_list = self._get_bigram_list(temp_list)
            bigram_count = collections.Counter(bigram_list)
            repititive_count.append(float(sum(1 for i in bigram_count.values() if i > 1)))
        return np.array(repititive_count).reshape(-1,1)

    def _get_bigram_list(self, unigram_list):
        bigram_list = []
        for i in range(1, len(unigram_list)):
           bigram_list.append(unigram_list[i - 1] + " " + unigram_list[i])
        return bigram_list




    def _cleanUp(self, line):
        a = factory.EngLowercase()
        lines = a.lowercase(line)
        stopset = set(stopwords.words('english')) | set(string.punctuation)
        cleanup = " ".join(filter(lambda word: word not in stopset, lines.split()))
        if (self.remove_digit):
            cleanup = ''.join(i for i in cleanup if not i.isdigit())
        if(self.remove_punctuation):
            cleanup = cleanup.translate(self.table)
        return cleanup

    def _preprocess_data(self, summary):
        preprocessed_summary = []
        for i in range(len(summary)):
            preprocessed_summary.append(self._cleanUp(summary[i]))
        return preprocessed_summary

