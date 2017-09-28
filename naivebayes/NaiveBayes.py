from numpy import *

def loadDataSet(fileName):
    #파일을 읽는다.
    fr = open(fileName, 'rt', encoding='UTF8')
    #파일의 각 줄로 배열을 만든다.
    arrayOfLines = fr.readlines()
    #0으로된 (파일줄수, k)의 matrix를 만든다.
    classVec = []
    #단어를 저장할 배열을 만든다.
    wordList = [];
    #각 라인의 데이터를 매트릭스와 라벨베열에 넣어준다.
    for line in arrayOfLines:
        #문자 양끝 공백제거. (필요에 따라 생략가능)
        line = line.strip()
        #라인을 space기준으로 스플릿하여 리스트로 만든다.
        listFromLine = line.split(' ')
        #단어리스트에 추가시켜준다.
        wordList.append(listFromLine[0:-1])
        #데이터 마지막에 있는 구분값을 구분벡터에 넣어준다..
        classVec.append(int(listFromLine[-1]))
    return wordList, classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList, sentence):
    returnVec = [0]*len(vocabList)
    for word in sentence:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print("the word: %s is not in my vocabulary" % word)
    return returnVec

def trainNaiveBayes(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])

    # 카테고리수 / 총문장갯수 => 문장이 들어왔을때 어느 카테고리일지 나타내는 확률
    pAbusive = sum(trainCategory) / float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num/ p0Denom)

    return p0Vect, p1Vect, pAbusive

def classifyNaiveBayes(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def executeNaiveBayesTraining(fileName):
    #데이터 로드하고
    listOfPosts, listClasses = loadDataSet(fileName)
    postCount = len(listOfPosts)
    #트레이닝 데이터 뽑고
    trainStartNum = int(postCount * 0.1)
    trainPost = listOfPosts[trainStartNum:]
    #검증 데이터 뽑고
    verificationPost = listOfPosts[:trainStartNum]
    #단어 리스트만들고
    myVocabList = createVocabList(listOfPosts)
    #트레이닝 시키고
    trainMat = []
    for postInDoc in trainPost:
        trainMat.append(setOfWords2Vec(myVocabList, postInDoc))

    p0V, p1V, pAb = trainNaiveBayes(trainMat, listClasses)

    answerCount = 0
    #검증단계.
    for i in range(trainStartNum):
        print(verificationPost[i])
        verificationDoc = array(setOfWords2Vec(myVocabList, verificationPost[i]))
        verificationClass = classifyNaiveBayes(verificationDoc, p0V, p1V, pAb)
        print(verificationClass)
        if verificationClass == listClasses[i]:
            answerCount += 1

    answerPercent = answerCount / postCount * 0.1;
    print("잘맞았나 확률 : %f" % answerPercent)


executeNaiveBayesTraining('data.txt')
#
# myVocabList = createVocabList(listOPosts)
#
# print(myVocabList)
#
# trainMat = []
# for postInDoc in listOPosts:
#     trainMat.append(setOfWords2Vec(myVocabList, postInDoc))
#
# print(trainMat)
#
# p0V, p1V, pAb = trainNaiveBayes(trainMat, listClasses)
#
# testEntry = ['love', 'my', 'dalmation']
# thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
#
# print(classifyNaiveBayes(thisDoc, p0V, p1V, pAb))

