import os
from nltk.tokenize import RegexpTokenizer
from nltk.parse.corenlp import CoreNLPDependencyParser
dep_parser = CoreNLPDependencyParser(url='http://localhost:9000', tagtype="pos")
import collections

def read_data():
    count_preposition = []
    tokenizer = RegexpTokenizer(r'\w+')
    write_to_file = []
    for _, _, file in os.walk("../../data/parsing_corpus"):
        for filename in file:
            with open("../../data/parsing_corpus/" + filename, "r") as f:
                preposition_list = []
                contents = f.read()
                contents = contents.split("\n")
                for i in range(len(contents)):
                    temp_list = []
                    temp_tokenized_sentence = tokenizer.tokenize(contents[i])
                    if(len(temp_tokenized_sentence) <= 50):
                        if(temp_tokenized_sentence):
                            parses = dep_parser.parse(temp_tokenized_sentence)
                            for parse in parses:
                                for governor, dep, dependent in parse.triples():
                                    if(governor[1] == "IN"):
                                        if(governor not in temp_list):
                                            temp_list.append(governor)
                                    if(dependent[1] == "IN"):
                                        if(dependent not in temp_list):
                                            temp_list.append(dependent)
                    preposition_list.extend(temp_list)
                write_to_file.append(filename + ": " + str(len(preposition_list)) + "\n")
                count_preposition.extend(preposition_list)
    return write_to_file, collections.Counter(count_preposition)

file_data, preposition_counter = read_data()
common = preposition_counter.most_common(3)
with open("./Prepositions.txt", "w") as file:
    file.write("the most common prepositions found are:" + "\n")
    for count in common:
        word_tag = count[0]
        file.write(word_tag[0] + "\n")
    for i in range(len(file_data)):
        file.write(file_data[i] + "\n")


