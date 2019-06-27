from character_level_language_model.Preprocess import Preprocess
from character_level_language_model.n_grams import character_n_gram
import sys

class generate_n_grams:
    def __init__(self, directory = "gutenberg", n_grams = 3):
        self.directory = str(directory)
        self.n_grams = int(n_grams)

    def generate(self):
        preprocess = Preprocess(self.directory)
        self.preprocessed_data = preprocess.preprocess()
        n_gram = character_n_gram(self.preprocessed_data, 3, return_counts=True)
        n_grams_dictionary, n_grams_count_dictionary = n_gram.get_n_gram()
        for i in range(1, self.n_grams + 1):
            with open(str(i) + "-gram.txt", "w") as f:
                for key, value in n_grams_count_dictionary[i].items():
                    f.write(key +  " =  "  + str(value) + "\n")


n_grams = generate_n_grams()
n_grams.generate()




