import os
from character_level_language_model.n_grams import character_n_gram


class Preprocess:
    def __init__(self, directory, remove_blank_lines = True,replace_newline_characters = True, remove_duplicate_spaces  = True,
                 replace_minimum_occurences  =  True, minimum_occurences = 5, replace_character  = "#"):
        self.directory = directory
        self.remove_blank_lines = remove_blank_lines
        self.replace_newline_characters = replace_newline_characters
        self.remove_duplicate_spaces = remove_duplicate_spaces
        self.replace_minimum_occurences = replace_minimum_occurences
        self.minimum_occurences = minimum_occurences
        self.preprocessed_data = ""
        self.replace_character = replace_character

    def preprocess(self):
        for _, _, file in os.walk("../data/" + self.directory):
            for filename in file:
                with open("../data/"+ self.directory + "/" + filename, "r", encoding="latin-1") as f:
                    preprocessed_data = ""
                    if(self.remove_blank_lines):
                        preprocessed_data = "".join(line for line in f if not line.isspace())
                    if(self.remove_duplicate_spaces):
                        preprocessed_data = " ".join(preprocessed_data.split())
                    self.preprocessed_data = self.preprocessed_data + " " + preprocessed_data
            if(self.replace_minimum_occurences):
                self.preprocessed_data = self.__replace_minimum_occurences()
        return self.preprocessed_data

    def __replace_minimum_occurences(self):
        n_gram = character_n_gram(self.preprocessed_data, 1, return_counts=True)
        unigrams, unigrams_count = n_gram.get_n_gram()
        min_frequency = set()
        for key, value in unigrams_count[1].items():
            if (value <= 5):
                #self.preprocessed_data = [w.replace(key, "#") for w in self.preprocessed_data]
                min_frequency.add(key)
        for i in  range(len(self.preprocessed_data)):
            if(self.preprocessed_data[i] in min_frequency):
                self.preprocessed_data = self.preprocessed_data[:i] + self.replace_character + self.preprocessed_data[i + 1:]
        return "".join(self.preprocessed_data)







