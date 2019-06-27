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
c = []
pos_tags_sentences = []
pos_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='pos')
for i in range(len(tokenized_sentences)):
    if(tokenized_sentences[i]):
        b = list(pos_tagger.tag(tokenized_sentences[i]))
        pos_tags_sentences.append(b)
with open('./pos_sentences', 'w') as filehandle:
    for i in range(len(pos_tags_sentences)):
        filehandle.write(sentences[i] + "\n")
        filehandle.write(str(pos_tags_sentences[i]))
        filehandle.write("\n\n")