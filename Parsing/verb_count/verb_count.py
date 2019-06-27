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

pos_tagger = CoreNLPParser(url='http://localhost:9000', tagtype='pos')
pos_tags_sentences = []

verb_list = ["VB", "VBD", "VBG", "VBN","VBP", "VBZ"]
count_verbs = 0
for i in range(len(tokenized_sentences)):
    if(tokenized_sentences[i]):
        b = list(pos_tagger.tag(tokenized_sentences[i]))
        for word_tag in b:
            if(word_tag[1] in verb_list):
                count_verbs +=1
        pos_tags_sentences.append(b)

with open('./verb_count', 'w') as filehandle:
    filehandle.write("The verbs considered are:")
    for i in verb_list:
        filehandle.write(i + " ")
    filehandle.write("\nAverage number of verbs per sentence in the overall corpus: " + str(count_verbs / len(pos_tags_sentences)))
