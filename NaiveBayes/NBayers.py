import os
import re
import numpy as np
import math

''' parse_file is reading each email, each line and storing the count for the ham and spam for a term in following format
    hamVocab has all the vocabulary of ham with structure {'term' : count_in_ham}
    spamVocab has all the vocabulary of spam with structure {'term' : count_in_spam}
'''

def filter_email(fileName,vocabulary,globVocab):
    file = open(fileName, "r")
    for line in file.readlines():
        line = line.lower()
        line = re.sub('[^A-Za-z0-9 ]+', ' ', line)
        line = re.split(" ", line)
        for i in line:
            if i != '':
                c = vocabulary.get(i)
                if c:
                    vocabulary[i] = c + 1
                else:
                    vocabulary[i] = 1
                globVocab.append(i)

def parse_file():
    vocabulary = []
    hamVocab = {}
    spamVocab = {}
    ham = 0
    spam = 0
    files = os.walk("../Training Set/", topdown=True)
    files = files.__next__()[2]
    # print(files)
    for fileName in files:
        file_path = "../Training Set/" + fileName
        fname = re.split("-",fileName)
        if fname[1] == 'ham':
            ham += 1
            filter_email(file_path, hamVocab,vocabulary)
        else:
            spam += 1
            filter_email(file_path, spamVocab,vocabulary)
    vocabulary = list(dict.fromkeys(vocabulary))
    vocabulary.sort()
    calculate_probabilities(ham, spam, vocabulary, hamVocab, spamVocab)


'''model is a dictionary where, key is the term and value is following 
    [term,  c(term, Ham), P(term | ham) , count(term, spam), P(term | spam)]'''

def calculate_probabilities(ham, spam, vocabulary, hamVocab, spamVocab):

    totalEmail = ham + spam
    PHam = ham / totalEmail
    PSpam = spam / totalEmail
    PHamWord = {}
    PSpamWord = {}
    hamWords = len(hamVocab)
    spamWords = len(spamVocab)
    vocabularySize = len(vocabulary)

    for k , v in hamVocab.items():
        PHamWord[k] = (v + 0.5) / (hamWords + (0.5 * vocabularySize))

    for k , v in spamVocab.items():
        PSpamWord[k] = (v + 0.5) / (spamWords + (0.5 * vocabularySize))

    model = {}

    for term in vocabulary:
        row = [term]
        p = PHamWord.get(term)
        c = hamVocab.get(term)
        if not p:
            c = 0
            p = 0.5 / (hamWords + (0.5 * vocabularySize))
        row.append(c)
        row.append(p)
        p = PSpamWord.get(term)
        c = spamVocab.get(term)
        if not p:
            c = 0
            p = 0.5 / (spamWords + (0.5 * vocabularySize))
        row.append(c)
        row.append(p)
        model[term] = row

    # for k, v in model.items():
    #     print(k, '->', v)
    generateModel(model)
    # print(ham)
    # print(spam)
    # print(len(hamVocab))
    # print(len(spamVocab))
    # print(len(vocabulary))
    # print(PHam)
    # print(PSpam)
    # print(len(PHamWord))
    # print(len(PSpamWord))
    # print(hamWords)
    # print(spamWords)

def generateModel(model):
    f = open("model.txt", "w+")
    i = 1
    for k , v in model.items():
        line = str(i) + "  " + v[0] + "  " + str(v[1]) + "  " + str(v[2]) + "  " + str(v[3]) + "  " + str(v[4]) + "\n"
        f.write(line)
        i += 1


parse_file()
