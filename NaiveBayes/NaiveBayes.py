import glob
import re
import numpy as np


''' The project data structure is as follows:
the parser parses a file and tokenizes it. The tokens are then stored in a directory named vocabulary with word as key 
and a list as item. The list would contain all the word related data in the following format.
[ count of word, conditional probability in ham, conditional probability in spam]
I have already completed parsing and count calculation. The conditional probability calculations are left.  
'''

vocab = []
vocabulary = {}

def parse_file():
    path = '../Training Set/*.txt'      ## check with prof
    files = glob.glob(path)         ## check with prof
    file = open("../Training Set/train-ham-00006.txt","r")
    for line in file.readlines():
        #print(line)
        line = line.lower()
        line = re.sub('[^A-Za-z0-9 ]+', ' ', line)
        line = re.split(" ", line)
        for i in line:
            if (i != " " and i != ''):
                vocab.append(i)

    for i in vocab:
        word_data = []
        count = vocab.count(i)
        word_data.append(count)
        vocabulary[i] = word_data
        vocab.remove(i)

    print(vocabulary)




parse_file()