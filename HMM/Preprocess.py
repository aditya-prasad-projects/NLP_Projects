import os
import collections


class Preprocess:
    def __init__(self, directory, replace_minimum_occurences  =  True, minimum_occurences = 5, replace_character  = "UNK/"):
        self.directory = directory
        self.replace_minimum_occurences = replace_minimum_occurences
        self.minimum_occurences = minimum_occurences
        self.replace_character = replace_character

    def read_data(self):
        contents = ""
        for _, _, file in os.walk("../data/" + self.directory):
            for filename in file:
                with open("../data/"+ self.directory + "/" + filename, "r", encoding="latin-1") as f:
                    contents = contents + " " + f.read()
        contents = contents.split()
        contents = self.__remove_missing_data(contents)
        if (self.replace_minimum_occurences):
            return self.__replace_UNK(contents)
        else:
            return contents

    def __replace_UNK(self, data):
        data_count = collections.Counter(data)
        min_frequency = set()
        for key, value in data_count.items():
            if (value <= 5):
                min_frequency.add(key)
        for i in range(len(data)):
            index = data[i].index("/")
            word_tag = data[i]
            tag = word_tag[index + 1:]
            if (data[i] in min_frequency):
                data[i] = "UNK/" + tag
        return data

    def __remove_missing_data(self, data):
        remove_data = []
        for t in data:
            if "/" not in t:
                remove_data.append(t)
        for i in remove_data:
            data.remove(i)
        return data