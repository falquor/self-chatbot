# -*- coding: utf-8 -*-
"""
Created on Sat Sep  2 20:29:50 2017

@author: Quentin
"""

import tensorflow as tf
import json
import os
import numpy as np
from random import randint
from sklearn.utils import shuffle
import datetime

with open("messages.json", 'r', encoding = 'utf8') as file:
    messages = json.load(file)
    
with open("wordList.json",'r', encoding = 'utf8') as file:
    wList = json.load(file)
    
# Hyperparamters
batchSize = 24
maxEncoderLength = 30
maxDecoderLength = maxEncoderLength
lstmUnits = 112
embeddingDim = lstmUnits
numLayersLSTM = 3
numIterations = 500000

def wordlist(messages):
    wList = []
    string_alt = ''
    i = 0
    for conv in messages:
        string_alt += ' ' + conv[0] + ' ' + conv[1] + ' '
        i+=1
        print(i)
    allWords = list(set(string_alt.split()))
    i = 0
    for word in allWords:
        if 'http' not in word:
            wList.append(word)
        i+=1
        print(str(i) + "/" + str(len(allWords)))
    string_alt = ''
    for word in wList:
        string_alt+= ' ' + word
    string_alt = string_alt.replace(',',' ').replace('.',' ').replace('?',' ').replace(':',' ').replace('!',' ').replace('(',' ').replace(')',' ').replace('"',' ').replace('[',' ').replace(']',' ').replace('=',' ').replace('/',' ').replace('~',' ').replace(';',' ')
    wList = list(set(string_alt.split()))
    return wList

#wList = wordlist(messages)

#with open("wordList.json",'w', encoding = 'utf8') as file:
#    json.dump(wList,file)
    
vocabSize = len(wList)
wList.append('<pad>')
wList.append('<EOS>')
vocabSize += 2

def createTrainingMatrices(wList,messages,maxLen):
    numExamples = len(messages)
    xTrain = np.zeros((numExamples, maxLen), dtype='int32')
    yTrain = np.zeros((numExamples, maxLen), dtype='int32')
    i=0
    for message in messages:
        encoderMessage = np.full((maxLen), wList.index('<pad>'), dtype='int32')
        decoderMessage = np.full((maxLen), wList.index('<pad>'), dtype='int32')
        friendMessage = message[0].replace(',',' ').replace('.',' ').replace('?',' ').replace(':',' ').replace('!',' ').replace('(',' ').replace(')',' ').replace('"',' ').replace('[',' ').replace(']',' ').replace('=',' ').replace('/',' ').replace('~',' ').replace(';',' ')
        myMessage = message[1].replace(',',' ').replace('.',' ').replace('?',' ').replace(':',' ').replace('!',' ').replace('(',' ').replace(')',' ').replace('"',' ').replace('[',' ').replace(']',' ').replace('=',' ').replace('/',' ').replace('~',' ').replace(';',' ')
        friendMessageSplit = friendMessage.split()
        myMessageSplit = myMessage.split()
        friendMessageCount = len(friendMessageSplit)
        myMessageCount = len(myMessageSplit)
        #Throw out sequences that are too long
#        print(friendMessageCount)
#        print(myMessageCount)
        if (friendMessageCount > (maxLen -1) or myMessageCount > (maxLen -1)):
            continue
        #Integerize the encoder string
        for index,word in enumerate(friendMessageSplit):
            try:
                encoderMessage[index] = wList.index(word)
            except ValueError:
                #TODO : verify the error
                encoderMessage[index] = 0
        encoderMessage[index+1] = wList.index('<EOS>')
        
        #Integerize the decoder string
        for index, word in enumerate(myMessageSplit):
            try :
                decoderMessage[index] = wList.index(word)
            except ValueError:
                #TODO : verify the error
                decoderMessage[index] = 0
        decoderMessage[index+1] = wList.index('<EOS>')
        xTrain[i] = encoderMessage
        yTrain[i] = decoderMessage
        i+=1
        print(str(i) + "/" + str(numExamples))
    yTrain = yTrain[~np.all(yTrain == 0, axis=1)]
    xTrain = xTrain[~np.all(xTrain == 0, axis=1)]
    numExamples = xTrain.shape[0]
    
    return numExamples, xTrain, yTrain

def getTrainingBatch(localXTrain, localYTrain, localBatchSize, maxLen):
    num = randint(0,numTrainingExamples - localBatchSize -1)
    arr = localXTrain[num:num + localBatchSize]
    labels = localYTrain[num:num + localBatchSize]
    
    reversedList = list(arr)
    for index,example in enumerate(reversedList):
        reversedList[index] = list(reversed(example))
        
    laggedLabels = []
    EOStokenIndex = wList.index('<EOS>')
    padTokenIndex = wList.index('<pad>')
    for example in labels:
        eosFound = np.argwhere(example==EOStokenIndex)[0]
        shiftedExample = np.roll(example,1)
        shiftedExample[0] = EOStokenIndex
        
        if (eosFound != (maxLen-1)):
            shiftedExample[eosFound+1] = padTokenIndex
        laggedLabels.append(shiftedExample)
        
    reversedList = np.asarray(reversedList).T.tolist()
    labels = labels.T.tolist()
    laggedLabels = np.asarray(laggedLabels).T.tolist()
    
    return reversedList, labels, laggedLabels

if (os.path.isfile('seq2seqXTrain.npy') and os.path.isfile('seq2seqYTrain.npy')):
    xTrain = np.load('seq2seqXTrain.npy')
    yTrain = np.load('seq2seqYTrain.npy')
    print('matrices loaded')
    numTrainingExamples = xTrain.shape[0]
else: 
    numTrainingExamples, xTrain, yTrain = createTrainingMatrices(wList, messages, maxEncoderLength)
    np.save('seq2seqXTrain.npy', xTrain)
    np.save('seq2seqYTrain.npy', yTrain)
    print('matrices created')

