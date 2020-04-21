import os
import re
import numpy as np

''' parse_file is reading each email, each line and storing the count for the ham and spam for a term in following format
    hamVocab has all the vocabulary of ham with structure {'term' : count_in_ham}
    spamVocab has all the vocabulary of spam with structure {'term' : count_in_spam}
'''

def filter_email(fileName,vocabulary):
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

def parse_file():
    vocabulary = {}
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
            filter_email(file_path, hamVocab)
        else:
            spam += 1
            filter_email(file_path, spamVocab)

    print(ham)
    print(spam)
    print(hamVocab)
    print(spamVocab)

parse_file()
