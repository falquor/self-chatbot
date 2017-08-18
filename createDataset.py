# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 22:33:23 2017

@author: Quentin
"""

import pandas as pd
import numpy as np
import os
import re
from datetime import datetime

def getFacebookData():
	responseDictionary = dict()
	fbFile = open('fbMessages.txt', 'r') 
	allLines = fbFile.readlines()
	myMessage, otherPersonsMessage, currentSpeaker = "","",""
	for index,lines in enumerate(allLines):
	    rightBracket = lines.find(']') + 2
	    justMessage = lines[rightBracket:]
	    colon = justMessage.find(':')
	    # Find messages that I sent
	    if (justMessage[:colon] == personName):
	        if not myMessage:
	            # Want to find the first message that I send (if I send multiple in a row)
	            startMessageIndex = index - 1
	        myMessage += justMessage[colon+2:]
	        
	    elif myMessage:
	        # Now go and see what message the other person sent by looking at previous messages
	        for counter in range(startMessageIndex, 0, -1):
	            currentLine = allLines[counter]
	            rightBracket = currentLine.find(']') + 2
	            justMessage = currentLine[rightBracket:]
	            colon = justMessage.find(':')
	            if not currentSpeaker:
	                # The first speaker not named me
	                currentSpeaker = justMessage[:colon]
	            elif (currentSpeaker != justMessage[:colon] and otherPersonsMessage):
	                # A different person started speaking, so now I know that the first person's message is done
	                otherPersonsMessage = cleanMessage(otherPersonsMessage)
	                myMessage = cleanMessage(myMessage)
	                responseDictionary[otherPersonsMessage] = myMessage
	                break
	            otherPersonsMessage = justMessage[colon+2:] + otherPersonsMessage
	        myMessage, otherPersonsMessage, currentSpeaker = "","",""    
	return responseDictionary
 
fbFile = open('conv.txt','r')
allLines = fbFile.readlines()
#n = len(allLines)
#for k in range(n):
#    if k%3 == 1 :
#        allLines.pop(k)
name = "Quentin Bouniot"
list_messages = []
message = ''
response = ''
conversation = []
n = len(allLines)
k = 0
for line in allLines:
    if k%3 == 0:
        