import pandas as pd
import numpy as np
from text_normalizer import factory
from nltk.corpus import stopwords
import collections
import string
import gensim
from nltk import tokenize
from sklearn.metrics.pairwise import cosine_similarity
import sklearn
import nltk


class Preprocess:
    def __init__(self, file_name, remove_digit, remove_punctuation, extra_feature):
        self.file_name = file_name
        self.remove_digit = remove_digit
        self.table = str.maketrans({key: None for key in string.punctuation})
        self.remove_punctuation = remove_punctuation
        self.extra_feature = extra_feature

    def _read_data(self):
        data = pd.read_csv(self.file_name)
        dataset = data.values
        return dataset

    def preprocess(self, redundancy):
        dataset = self._read_data()
        preprocessed_summary = self._preprocess_data(dataset[:,0])
        unigram_feature = self._getUnigramFeature(preprocessed_summary)
        print("got unigram features")
        bigram_feature = self._getBigramFeature(preprocessed_summary)
        print("got bigram features")
        features = []
        if(self.extra_feature):
            for i in range(len(self.extra_feature)):
                if(self.extra_feature[i] == "uni_count"):
                    unigram_count = self._getUnigramCount(preprocessed_summary)
                    features.append(unigram_count)
                if(self.extra_feature[i] == "pos_tags"):
                    pos_tags = self._getPosTags(preprocessed_summary)
                    features.append(pos_tags)
                if(self.extra_feature[i] == "tri_count"):
                    trigram_count = self._getTrigramCount(preprocessed_summary)
                    features.append(trigram_count)
                if(self.extra_feature[i] == "word_count"):
                    word_count = self._get_word_count(preprocessed_summary)
                    features.append(word_count)
                if(self.extra_feature[i] == "bi_count"):
                    bi_count = self._get_bi_count(preprocessed_summary)
                    features.append(bi_count)

        #sentence_similarity = self._get_sentence_similarity(preprocessed_summary, dataset)
        sentence_similarity = self._get_sentence_similarity_from_file(preprocessed_summary, dataset)
        print("got sentence similarity")
        if(redundancy):
            matrix = self._build_matrix(unigram_feature, bigram_feature, sentence_similarity, (list(dataset[:, 1])), features)
        else:
            matrix = self._build_matrix(unigram_feature, bigram_feature,sentence_similarity, (list(dataset[:, 2])), features)
        #matrix = sklearn.preprocessing.normalize(matrix, axis = 0)
        return matrix
        #np.savetxt("./matrix.csv", matrix, delimiter=",")

    def _get_word_count(self, preprocessed_summary):
        word_count = []
        for i in range(len(preprocessed_summary)):
            word_count.append(float(len(preprocessed_summary[i].split())))
        return np.array(word_count).reshape(-1,1)

    def _getPosTags(self, preprocessed_summary):
        pos_tag_count = []
        for i in range(len(preprocessed_summary)):
            tokens = nltk.word_tokenize(preprocessed_summary[i])
            temp_tokens = np.array(nltk.pos_tag(tokens)).T[1]
            pos_counter = collections.Counter(temp_tokens)
            pos_tag_count.append(float(pos_counter.most_common(1)[0][1]))
        return np.array(pos_tag_count).reshape(-1,1)

    def _get_sentence_similarity_from_file(self, preprocessed_summary, dataset):
        sentence_similarity = []
        vectors = self._get_vectors_from_file()
        for i in range(len(preprocessed_summary)):
            temp_sentence_list = tokenize.sent_tokenize(dataset[i, 0])
            sentence_vector_list = self._get_sentence_vector(temp_sentence_list, vectors)
            if(sentence_vector_list.shape[0] > 2):
                sentence_similarity.append(self._get_cosine_similarity(sentence_vector_list[1:]))
            else:
                sentence_similarity.append(1)
        return np.array(sentence_similarity).reshape(-1,1)

    def _get_sentence_vector(self, temp_sentence_list, vectors):
        sentence_vector_list = np.zeros(300).reshape(1, 300)
        for j in range(len(temp_sentence_list)):
            cleanup = self._cleanUp(temp_sentence_list[j])
            cleanup = cleanup.split()
            if(cleanup):
                word_embeddings =  self._get_average_word_embedding(cleanup, vectors)
                if(not(False in np.isfinite(word_embeddings))):
                    sentence_vector_list = np.concatenate((sentence_vector_list, word_embeddings), axis=0)
        return sentence_vector_list

    def _build_matrix(self, unigram_feature, bigram_feature, sentence_similarity, redundant_score, features):
        if(self.extra_feature and features):
            matrix = np.c_[unigram_feature, bigram_feature, sentence_similarity]
            for i in range(len(features)):
                matrix = np.c_[matrix, features[i]]
            matrix = np.c_[matrix, redundant_score]
        else:
            matrix = np.c_[unigram_feature, bigram_feature, sentence_similarity, redundant_score]
        return matrix

    def _get_average_word_embedding(self, cleanup, vectors):
        temp_matrix = np.zeros(300).reshape(1,300)
        for i in range(len(cleanup)):
            if(cleanup[i] in vectors):
                temp_matrix = np.concatenate((temp_matrix, vectors[cleanup[i]].reshape(1,300)), axis = 0)
        return temp_matrix[1:,:].mean(axis = 0).reshape(1,300)

    def _get_vectors_from_file(self):
        vectors = {}
        with  open("../../../word_vectors.txt", "r") as f:
            contents = f.read()
            contents = contents.split("\n")
            for i in range(len(contents) - 1):
                temp_numpy = []
                temp_value = contents[i].split(" ")
                temp_matrix = temp_value[1].split(",")

                for i in range(len(temp_matrix) - 1):
                    temp_numpy.append(float(temp_matrix[i]))
                vectors[temp_value[0]] = np.array(temp_numpy)
        return vectors

    def _get_sentence_similarity(self, preprocessed_summary, dataset):
        model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin',
                                                                binary=True)
        print("model loaded")
        sentence_similarity = []
        for i in range(len(preprocessed_summary)):
            sentence_vector_list = np.zeros(300).reshape(1,300)
            temp_sentence_list = tokenize.sent_tokenize(dataset[i,0])
            for j in range(len(temp_sentence_list)):
                sentence_vector_list = np.concatenate((sentence_vector_list, self._get_vector(model, temp_sentence_list[j].split()).reshape(1,300)), axis = 0)
            print(i)
            sentence_similarity.append(self._get_cosine_similarity(sentence_vector_list[1:]))
        return np.array(sentence_similarity).reshape(-1,1)

    def _get_vector(self, model, sentence_word_list):
        vectors = np.zeros(300).reshape(1, 300)
        for i in range(len(sentence_word_list)):
            if(sentence_word_list[i] in model.vocab):
                vectors = np.concatenate((vectors, model.get_vector(sentence_word_list[i]).reshape(1, 300)), axis=0)
        return vectors[1:,:].mean(axis = 0).reshape(1,300)

    def _get_cosine_similarity(self, sentence_vector_list):
        cosine_similarity_sentences = cosine_similarity(sentence_vector_list)
        return np.max(cosine_similarity_sentences[np.triu_indices(sentence_vector_list.shape[0], k=1)])

    def _getUnigramCount(self, preprocessed_summary):
        unigram_count = []
        for i in range(len(preprocessed_summary)):
            temp_list = preprocessed_summary[i].split()
            unigram_counter = collections.Counter(temp_list)
            most_common = unigram_counter.most_common(1)[0][1]
            if(most_common > 1):
                counter = collections.Counter(unigram_counter.values())
                unigram_count.append(float(counter[most_common]))
            else:
                unigram_count.append(2.0)
        return np.array(unigram_count).reshape(-1,1)

    def _get_bi_count(self, preprocessed_summary):
        bigram_count = []
        for i in range(len(preprocessed_summary)):
            temp_list = preprocessed_summary[i].split()
            bigram_list = self._get_bigram_list(temp_list)
            bigram_counter = collections.Counter(bigram_list)
            most_common = bigram_counter.most_common(1)[0][1]
            if(most_common > 1):
                counter = collections.Counter(bigram_counter.values())
                bigram_count.append((float(counter[most_common])))
            else:
                bigram_count.append(1.0)
        return bigram_count


    def _getUnigramFeature(self, preprocessed_summary):
        unigram_feature = []
        for i in range(len(preprocessed_summary)):
            temp_list = preprocessed_summary[i]
            temp_list = temp_list.split()
            unigram_counter = collections.Counter(temp_list)
            unigram_feature.append(float(unigram_counter.most_common(1)[0][1]))
        return np.array(unigram_feature).reshape(-1,1)

    def _getBigramFeature(self, preprocessed_summary):
        bigram_feature = []
        for i in range(len(preprocessed_summary)):
            unigram_list = preprocessed_summary[i].split()
            bigram_list = (self._get_bigram_list(unigram_list))
            bigram_feature.append(float(collections.Counter(bigram_list).most_common(1)[0][1]))
        return np.array(bigram_feature).reshape(-1,1)

    def _getTrigramCount(self, preprocessed_summary):
        trigram_feature = []
        for i in range(len(preprocessed_summary)):
            unigram_list = preprocessed_summary[i].split()
            trigram_list = self._getTrigramList(unigram_list)
            trigram_feature.append(float(collections.Counter(trigram_list).most_common(1)[0][1]))
        return np.array(trigram_feature).reshape(-1,1)

    def _getTrigramList(self, unigram_list):
        trigram_list = []
        for i in range(2, len(unigram_list)):
            trigram_list.append(unigram_list[i-2] + " " + unigram_list[i-1] + " " + unigram_list[i])
        return trigram_list


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

