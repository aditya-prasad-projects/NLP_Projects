from collections import Counter

class character_n_gram:
    def __init__(self, data, n_gram, separator = "", return_counts = False):
        self.data = data
        self.n_gram = n_gram
        self.separator= separator
        self.return_counts = return_counts

    def get_n_gram(self):
        self.n_grams_count_dictionary = {}
        self.n_grams_dictionary ={}
        self.unigrams = list(self.data)
        self.unigrams_counter = Counter(self.unigrams)
        self.n_grams_count_dictionary[1] = self.unigrams_counter
        self.n_grams_dictionary[1] = self.unigrams
        for i in range(1,self.n_gram):
            temp_n_gram = []
            for j in range(i,len(self.unigrams)):
                temp_n_gram.append(self.separator.join(self.unigrams[j - i :j+1]))
            temp_n_gram_counter = Counter(temp_n_gram)
            self.n_grams_count_dictionary[i+1] = temp_n_gram_counter
            self.n_grams_dictionary[i+1] = temp_n_gram
        if(self.return_counts):
            return self.n_grams_dictionary, self.n_grams_count_dictionary
        else:
            return self.n_grams_dictionary





