import os
from nltk.tokenize import RegexpTokenizer
from nltk.parse.corenlp import CoreNLPDependencyParser
dep_parser = CoreNLPDependencyParser(url='http://localhost:9000', tagtype="pos")

def read_data():
    tokenizer = RegexpTokenizer(r'\w+')
    tokenized_sentences = []
    sentences = []
    for _, _, file in os.walk("../../data/parsing_corpus"):
        for filename in file:
            with open("../../data/parsing_corpus/" + filename, "r") as f:
                contents = f.read()
                contents = contents.split("\n")
                for i in range(len(contents)):
                    temp_tokenized_sentence = tokenizer.tokenize(contents[i])
                    if(len(temp_tokenized_sentence) <= 50):
                        tokenized_sentences.append(temp_tokenized_sentence)
                        sentences.append(contents[i])
    return tokenized_sentences, sentences

tokenized_sentences, sentences = read_data()
dependency_parsed = []
with open("./dependencies.txt", "w") as f:
    for i in range(len(tokenized_sentences)):
        if(tokenized_sentences[i]):
            f.write(sentences[i] + "\n")
            parses = dep_parser.parse(tokenized_sentences[i])
            for parse in parses:
                for governor, dep, dependent in parse.triples():
                    f.write("(" + governor[0] + ","  + dependent[1] +  ") " + dep + " " + "(" + dependent[0] + ","  + dependent[1] + ") " + "\n")
            f.write("\n")













