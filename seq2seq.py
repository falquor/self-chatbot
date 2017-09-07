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
#from sklearn.utils import shuffle
import datetime

with open("messages.json", 'r', encoding = 'utf8') as file:
    messages = json.load(file)
    
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

def getTestInput(inputMessage, wList, maxLen):
    encoderMessage = np.full((maxLen), wList.index('<pad>'), dtype='int32')
    inputSplit = inputMessage.lower().split()
    for index, word in enumerate(inputSplit):
        try:
            encoderMessage[index] = wList.index(word)
        except ValueError:
            continue
    encoderMessage[index + 1] = wList.index('<EOS>')
    encoderMessage = encoderMessage[::-1]
    encoderMessageList=[]
    for num in encoderMessage:
        encoderMessageList.append([num])
    
    return encoderMessageList

def idsToSentence(ids, wList):
    EOStokenIndex = wList.index('<EOS>')
    padTokenIndex = wList.index('<pad>')
    myStr = ""
    listOfResponses = []
    for num in ids:
        if (num[0] == EOStokenIndex or num[0] == padTokenIndex):
            listOfResponses.append(myStr)
            myStr = ""
        else:
            myStr = myStr + wList[num[0]] + " "
    if myStr:
        listOfResponses.append(myStr)
    listOfResponses = [i for i in listOfResponses if i]
    
    return listOfResponses

# Hyperparamters
batchSize = 24
maxEncoderLength = 30
maxDecoderLength = maxEncoderLength
lstmUnits = 112
embeddingDim = lstmUnits
numLayersLSTM = 3
numIterations = 500000


if (os.path.isfile('wordList.json')):
    with open("wordList.json",'r', encoding = 'utf8') as file:
        wList = json.load(file)
else:
    wList = wordlist(messages)

    with open("wordList.json",'w', encoding = 'utf8') as file:
        json.dump(wList,file)
        

vocabSize = len(wList)
wList.append('<pad>')
wList.append('<EOS>')
vocabSize += 2

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

tf.reset_default_graph()

# Create the placeholders
encoderInputs = [tf.placeholder(tf.int32, shape=(None,)) for i in range(maxEncoderLength)]
decoderLabels = [tf.placeholder(tf.int32, shape=(None,)) for i in range(maxDecoderLength)]
decoderInputs = [tf.placeholder(tf.int32, shape=(None,)) for i in range(maxDecoderLength)]
feedPrevious = tf.placeholder(tf.bool)

encoderLSTM = tf.nn.rnn_cell.BasicLSTMCell(lstmUnits, state_is_tuple=True)

#encoderLSTM = tf.nn.rnn_cell.MultiRNNCell([singleCell]*numLayersLSTM, state_is_tuple=True)
# Architectural choice of of whether or not to include ^

decoderOutputs, decoderFinalState = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(encoderInputs, decoderInputs, encoderLSTM, 
															vocabSize, vocabSize, embeddingDim, feed_previous=feedPrevious)

decoderPrediction = tf.argmax(decoderOutputs, 2)

lossWeights = [tf.ones_like(l, dtype = tf.float32) for l in decoderLabels]
loss = tf.contrib.legacy_seq2seq.sequence_loss(decoderOutputs, decoderLabels, lossWeights, vocabSize)
optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

sess = tf.Session()
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

#if loading a saved model, use :
restored = True
if restored :
#    saver = tf.train.import_meta_graph('models/pretrained_seq2seq.ckpt-40000.meta')
    saver.restore(sess, tf.train.latest_checkpoint('models/'))
    

#uploading results to Tensorboard
tf.summary.scalar("Loss", loss)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

#
encoderTestStrings = ['oh shit', 'here come dat boi', 'fuccboi', 'harro', 'sup', 'LUL', 'salut comment ça va ?']

zeroVector = np.zeros((1), dtype = 'int32')

def askBot(question):
    inputVector = getTestInput(question, wList, maxEncoderLength)
    feedDict = {encoderInputs[t]: inputVector[t] for t in range(maxEncoderLength)}
    feedDict.update({decoderLabels[t]: zeroVector for t in range(maxDecoderLength)})
    feedDict.update({decoderInputs[t]: zeroVector for t in range(maxDecoderLength)})
    feedDict.update({feedPrevious: True})
    
    ids = (sess.run(decoderPrediction, feed_dict=feedDict))
    
    return (idsToSentence(ids,wList))

for i in range(80001,numIterations):
    
    encoderTrain, decoderTargetTrain, decoderInputTrain = getTrainingBatch(xTrain, yTrain, batchSize, maxEncoderLength)
    feedDict = {encoderInputs[t]: encoderTrain[t] for t in range(maxEncoderLength)}
    feedDict.update({decoderLabels[t]: decoderTargetTrain[t] for t in range(maxDecoderLength)})
    feedDict.update({decoderInputs[t]: decoderInputTrain[t] for t in range(maxDecoderLength)})
    feedDict.update({feedPrevious: False})
    
    curLoss, _, pred = sess.run([loss, optimizer, decoderPrediction], feed_dict=feedDict)
    print(i)
    
    if (i % 50 == 0):
        print('Current loss:', curLoss, 'at iteration', i)
        summary = sess.run(merged, feed_dict=feedDict)
        writer.add_summary(summary, i)
    
    if (i % 25 == 0 and i != 0):
        num = randint(0,len(encoderTestStrings) -1)
        print(encoderTestStrings[num])
        inputVector = getTestInput(encoderTestStrings[num], wList, maxEncoderLength);
        feedDict = {encoderInputs[t]: inputVector[t] for t in range(maxEncoderLength)}
        feedDict.update({decoderLabels[t]: zeroVector for t in range(maxDecoderLength)})
        feedDict.update({decoderInputs[t]: zeroVector for t in range(maxDecoderLength)})
        feedDict.update({feedPrevious: True})
        ids = (sess.run(decoderPrediction, feed_dict=feedDict))
        print(idsToSentence(ids, wList))
        
    if (i % 10000 == 0 and i != 0):
        savePath = saver.save(sess, "models/pretrained_seq2seq.ckpt", global_step=i)
    
    
    
    