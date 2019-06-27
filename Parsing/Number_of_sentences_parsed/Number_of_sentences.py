import os
from nltk.tokenize import RegexpTokenizer


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

count_sentences = 0
for i in range(len(tokenized_sentences)):
    if(tokenized_sentences[i]):
        count_sentences +=1
with open("./Number_of_sentences_parsed.txt", "w") as f:
    f.write("The number of sentences parsed is " + str(count_sentences))

c =[]