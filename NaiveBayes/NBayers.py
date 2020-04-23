import os
import re
import math

''' parse_file is reading each email, each line and storing the count for the ham and spam for a term in following format
    hamVocab has all the vocabulary of ham with structure {'term' : count_in_ham}
    spamVocab has all the vocabulary of spam with structure {'term' : count_in_spam}
'''
PHam = 0
PSpam = 0
model = {}

def filter_email(fileName,vocabulary,globVocab):
    file = open(fileName, "r", encoding="utf8", errors='ignore')
    for line in file.readlines():
        line = line.lower()
        line = re.sub('[^A-Za-z ]+', ' ', line)
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
        fname = re.split("-", fileName)
        if fname[1] == 'ham':
            ham += 1
            filter_email(file_path, hamVocab, vocabulary)
        else:
            spam += 1
            filter_email(file_path, spamVocab, vocabulary)
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
    hamWords = sum(hamVocab.values())
    spamWords = sum(spamVocab.values())
    vocabularySize = len(vocabulary)

    for k , v in hamVocab.items():
        PHamWord[k] = (v + 0.5) / (hamWords + (0.5 * vocabularySize))

    for k , v in spamVocab.items():
        PSpamWord[k] = (v + 0.5) / (spamWords + (0.5 * vocabularySize))

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

    generateModel(model)
    classifier(PHam,PSpam)


def generateModel(model):
    f = open("model.txt", "w+")
    i = 1
    for k , v in model.items():
        line = str(i) + "  " + v[0] + "  " + str(v[1]) + "  " + str(v[2]) + "  " + str(v[3]) + "  " + str(v[4]) + "\n"
        f.write(line)
        i += 1
    f.close()


def hamProbability(vocabulary_test,PHam):
    if(PHam != 0) :
        probability = math.log10(PHam)
    else:
        probability = 0
    for word in vocabulary_test.keys():
        if(model.__contains__(word)):
            probability = probability + (vocabulary_test[word]*math.log10(model[word][2]))
    return probability

def spamProbability(vocabulary_test,PSpam):
    if(PSpam != 0) :
        probability = math.log10(PSpam)
    else:
        probability = 0
    for word in vocabulary_test.keys():
        if(model.__contains__(word)):
            probability = probability + (vocabulary_test[word]*math.log10(model[word][4]))
    return probability

def confusion_matrix(truePositive,trueNegative,falsePositive,falseNegative):
    print("\n")
    print( "CONFUSION MATRIX")
    print("|-----------------------------|")
    print("|    |    HAM    |    SPAM    |")
    print("|----|------------------------|")
    print("|    |           |            |")
    print("|HAM |   " + str(truePositive) + "     |         " + str(falsePositive) + "  |")
    print("|    |           |            |")
    print("|----|------------------------|")
    print("|    |           |            |")
    print("|SPAM|   " + str(falseNegative) + "      |       " + str(trueNegative) + "  |")
    print("|    |           |            |")
    print("|-----------------------------|")

def display_evaluation_metrics(truePositive,trueNegative,falsePositive,falseNegative):
    accuracy = ((trueNegative + truePositive) / (trueNegative + truePositive + falseNegative + falsePositive)) * 100
    precision = (truePositive / (truePositive + falsePositive)) * 100
    recall = (truePositive / (truePositive + falseNegative)) * 100
    f1_score = ((2 * precision/100 * recall/100) / ((precision/100) + (recall/100))) * 100
    #total = trueNegative + truePositive + falseNegative + falsePositive
    print("\n")
    print("|--------------------|")
    print("| Accuracy  : " + str(accuracy) + "% |")
    print("|--------------------|")
    print("| Precision : " + str(precision) + "%  |")
    print("|--------------------|")
    print("| Recall    : " + str(round(recall,2)) + "% |")
    print("|--------------------|")
    print("| F1-Score  : " + str(round(f1_score, 2)) + "% |")
    print("|--------------------|")


def classifier(PHam,PSpam):
    files = os.walk("../Test Set/", topdown=True)
    files = files.__next__()[2]
    cntr = 1
    correctly_classified = 0
    incorrectly_classified = 0
    truePositive = 0
    trueNegative = 0
    falsePositive = 0
    falseNegative = 0

    f = open("result.txt", "w+")


    for fileName in files:
        file = fileName
        vocabulary_test = {}
        fileName = "../Test Set/" + fileName
        fname = re.split("-", fileName)
        if fname[1] == 'ham':
            correctClass = "ham"
        else:
            correctClass = "spam"

        filter_email(fileName, vocabulary_test, [])

        probHam = hamProbability(vocabulary_test,PHam)
        probSpam = spamProbability(vocabulary_test,PSpam)

        if probHam > probSpam:
            classifiedClass = "ham"
        else:
            classifiedClass = "spam"

        if classifiedClass.__eq__(correctClass):
            label = "right"
            correctly_classified = correctly_classified + 1
            if(correctClass == "ham") :
                truePositive = truePositive + 1
            else:
                trueNegative = trueNegative + 1
        else:
            label = "wrong"
            incorrectly_classified = incorrectly_classified +1
            if(correctClass == "ham"):
                falsePositive = falsePositive + 1
            else:
                falseNegative = falseNegative + 1

        line = str(cntr) + "  " + file + "  " + str(classifiedClass) + "  " + str(probHam) + "  " + \
               str(probSpam) + "  " + str(correctClass) + "  " + str(label) + "\n"
        f.write(line)
        cntr = cntr + 1

    confusion_matrix(truePositive, trueNegative, falsePositive, falseNegative)
    display_evaluation_metrics(truePositive, trueNegative, falsePositive, falseNegative)

    f.close()

parse_file()
