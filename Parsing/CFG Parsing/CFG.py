import os
from nltk.tokenize import RegexpTokenizer
from nltk.parse import CoreNLPParser

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

parser = CoreNLPParser(url = 'http://localhost:9000')
tree_sentences = []

for i in range(len(tokenized_sentences)):
    if(tokenized_sentences[i]):
        a = list(parser.parse(tokenized_sentences[i]))
        tree_sentences.append(a)
with open('./CFG trees.txt', 'w') as filehandle:
    for i in range(len(tree_sentences)):
        filehandle.write(sentences[i])
        filehandle.write("\n")
        filehandle.write(str(tree_sentences[i]))
        filehandle.write("\n\n")